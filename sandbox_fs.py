#!/usr/bin/env python3
"""
Sandbox FUSE filesystem daemon with dynamic mount management.

Provides a virtual filesystem that selectively exposes host directories
with fine-grained read/write control and path hiding. Supports runtime
reconfiguration via a Unix domain socket without restarting.

Usage:
   # Start the daemon
   sandbox_fs.py daemon /mnt/sandbox \\
       --rw-mount /home/user/project:/home/user/project \\
       --ro-mount /home/user/reference:/home/user/reference \\
      --hide /home/user/project:.env

  # Add a mount at runtime
  sandbox_fs.py mount -s /mnt/sandbox.sock /host/path /virtual/path -w

  # Hide a path within a mount
  sandbox_fs.py hide -s /mnt/sandbox.sock /virtual/mount .secrets

  # Unhide a path
  sandbox_fs.py unhide -s /mnt/sandbox.sock /virtual/mount .secrets

  # List current state
  sandbox_fs.py list -s /mnt/sandbox.sock

  # Save and restore state
  sandbox_fs.py save -s /mnt/sandbox.sock sandbox-state.json
  sandbox_fs.py load -s /mnt/sandbox.sock sandbox-state.json

   # Remove a mount
   sandbox_fs.py unmount -s /mnt/sandbox.sock /virtual/path

   # Inspect or update cache timeouts (seconds; defaults are 5)
   sandbox_fs.py timeouts -s /mnt/sandbox.sock
   sandbox_fs.py timeouts -s /mnt/sandbox.sock --entry-timeout 1 --attr-timeout 1

   # Immediately invalidate cached attributes, contents, and directory entries
   sandbox_fs.py invalidate-cache -s /mnt/sandbox.sock

Requires: pyfuse3, trio
Install:  pip install pyfuse3 trio
"""

import argparse
import errno
import json
import logging
import math
import os
import socket
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pyfuse3
import trio

log = logging.getLogger("sandboxfs")

_ROOT_INO = pyfuse3.ROOT_INODE
_DEFAULT_CACHE_TIMEOUT = 5.0


def _validate_timeout(value) -> float:
    message = "Timeout must be a finite, non-negative number of seconds"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(message)
    return timeout


def _timeout_argument(value: str) -> float:
    try:
        return _validate_timeout(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _normalize_mount_target(target: str) -> str:
    if not isinstance(target, str):
        raise ValueError("Mount target must be a string")
    parts = PurePosixPath(target.strip("/")).parts
    if not parts:
        raise ValueError("Cannot mount at root")
    if ".." in parts:
        raise ValueError(f"Mount target cannot contain '..': {target}")
    if any("\0" in part for part in parts):
        raise ValueError("Mount target cannot contain NUL")
    return "/" + "/".join(parts)


def _normalize_hidden_path(rel_path: str) -> str:
    if not isinstance(rel_path, str):
        raise ValueError("Hidden path must be a string")
    parts = PurePosixPath(rel_path.strip("/")).parts
    if not parts:
        raise ValueError("Hidden path cannot be empty")
    if ".." in parts:
        raise ValueError(f"Hidden path cannot contain '..': {rel_path}")
    if any("\0" in part for part in parts):
        raise ValueError("Hidden path cannot contain NUL")
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Inode:
    ino: int
    real_path: str | None = None  # None => virtual directory
    mount_target: str | None = None
    writable: bool = False
    active: bool = True
    lookup_count: int = 0
    children: dict[str, int] = field(default_factory=dict)

    @property
    def is_virtual(self) -> bool:
        return self.real_path is None


@dataclass
class _Handle:
    ino: int
    fd: int = -1
    dir_entries: list | None = None


# ---------------------------------------------------------------------------
# FUSE operations
# ---------------------------------------------------------------------------

class SandboxFS(pyfuse3.Operations):

    def __init__(self, entry_timeout: float = _DEFAULT_CACHE_TIMEOUT,
                 attr_timeout: float = _DEFAULT_CACHE_TIMEOUT):
        super().__init__()
        self._lock = threading.RLock()
        self._cache_control_lock = threading.Lock()
        self._ino_counter = _ROOT_INO
        self._inodes: dict[int, _Inode] = {}
        self._path_to_ino: dict[tuple[str, str], int] = {}
        self._handles: dict[int, _Handle] = {}
        self._fh_counter = 0
        self._mounts: dict[str, dict] = {}     # target -> mount info
        self._hidden: set[tuple[str, str]] = set()
        self._entry_timeout = _validate_timeout(entry_timeout)
        self._attr_timeout = _validate_timeout(attr_timeout)

        root = _Inode(ino=_ROOT_INO, lookup_count=1)
        self._inodes[_ROOT_INO] = root

    # -- inode helpers ------------------------------------------------------

    def _next_ino(self) -> int:
        self._ino_counter += 1
        return self._ino_counter

    def _next_fh(self) -> int:
        self._fh_counter += 1
        return self._fh_counter

    def _get(self, ino: int) -> _Inode:
        with self._lock:
            try:
                node = self._inodes[ino]
            except KeyError:
                raise pyfuse3.FUSEError(errno.ENOENT)
            if not node.active:
                raise pyfuse3.FUSEError(errno.ENOENT)
            return node

    def _alloc(self, real_path: str | None = None,
               mount_target: str | None = None,
               writable: bool = False) -> _Inode:
        ino = self._next_ino()
        node = _Inode(
            ino=ino,
            real_path=real_path,
            mount_target=mount_target,
            writable=writable,
        )
        self._inodes[ino] = node
        if real_path is not None and mount_target is not None:
            self._path_to_ino[(mount_target, real_path)] = ino
        return node

    def _is_hidden(self, mount_target: str, real_path: str) -> bool:
        with self._lock:
            for target, hidden_path in self._hidden:
                if target != mount_target:
                    continue
                if (
                    real_path == hidden_path
                    or real_path.startswith(hidden_path + "/")
                ):
                    return True
        return False

    def _rebuild_hidden(self):
        self._hidden = {
            (mount["target"], os.path.join(mount["source"], rel_path))
            for mount in self._mounts.values()
            for rel_path in mount["hidden"]
        }

    # -- attribute helpers --------------------------------------------------

    def _set_attr_timeouts(self, entry: pyfuse3.EntryAttributes):
        with self._lock:
            entry.entry_timeout = self._entry_timeout
            entry.attr_timeout = self._attr_timeout

    def _fill_attr(self, entry: pyfuse3.EntryAttributes,
                   st: os.stat_result, ino: int, writable: bool):
        entry.st_ino = ino
        entry.st_mode = st.st_mode
        entry.st_nlink = st.st_nlink
        entry.st_uid = st.st_uid
        entry.st_gid = st.st_gid
        entry.st_size = st.st_size
        entry.st_atime_ns = st.st_atime_ns
        entry.st_mtime_ns = st.st_mtime_ns
        entry.st_ctime_ns = st.st_ctime_ns
        entry.st_blksize = getattr(st, "st_blksize", 4096)
        entry.st_blocks = getattr(st, "st_blocks", 0)
        entry.st_rdev = getattr(st, "st_rdev", 0)
        if not writable:
            entry.st_mode &= ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        self._set_attr_timeouts(entry)

    def _virtual_attr(self, ino: int) -> pyfuse3.EntryAttributes:
        e = pyfuse3.EntryAttributes()
        e.st_ino = ino
        e.st_mode = stat.S_IFDIR | 0o555
        e.st_nlink = 2
        e.st_uid = os.getuid()
        e.st_gid = os.getgid()
        e.st_size = 0
        e.st_atime_ns = 0
        e.st_mtime_ns = 0
        e.st_ctime_ns = 0
        self._set_attr_timeouts(e)
        return e

    # -- cache management ---------------------------------------------------

    def get_timeouts(self) -> dict:
        with self._lock:
            return {
                "entry_timeout": self._entry_timeout,
                "attr_timeout": self._attr_timeout,
            }

    def set_timeouts(self, entry_timeout: float | None = None,
                     attr_timeout: float | None = None) -> dict:
        # Validate both before changing either, so a bad request is atomic.
        entry = None if entry_timeout is None else _validate_timeout(entry_timeout)
        attr = None if attr_timeout is None else _validate_timeout(attr_timeout)
        with self._cache_control_lock:
            with self._lock:
                if entry is not None:
                    self._entry_timeout = entry
                if attr is not None:
                    self._attr_timeout = attr

            if entry is None and attr is None:
                return {"ok": True, "timeouts": self.get_timeouts()}

            # Old kernel TTLs must be invalidated as well as updating future
            # replies. Notifications run without _lock, on the control thread.
            result = self._invalidate_cache()
            result["timeouts"] = self.get_timeouts()
            if not result["ok"]:
                result["error"] = (
                    "Timeouts updated, but some cache entries could not be invalidated"
                )
            log.info("Cache timeouts updated: %s", result["timeouts"])
            return result

    def invalidate_cache(self) -> dict:
        with self._cache_control_lock:
            return self._invalidate_cache()

    def _invalidate_cache(self) -> dict:
        with self._lock:
            inodes = list(self._inodes)
            entries = {
                (node.ino, os.fsencode(name))
                for node in self._inodes.values()
                for name in node.children
            }
            for (target, real_path) in self._path_to_ino:
                parent_ino = self._path_to_ino.get(
                    (target, os.path.dirname(real_path))
                )
                if parent_ino is not None:
                    entries.add((parent_ino, os.fsencode(os.path.basename(real_path))))

        result = {"ok": True, "inodes": 0, "entries": 0, "failed": []}
        # Notifications can wait for FUSE requests. Never hold _lock here:
        # those requests need it to complete. ENOENT means already uncached.
        for ino in inodes:
            try:
                pyfuse3.invalidate_inode(ino)
                result["inodes"] += 1
            except OSError as e:
                if e.errno != errno.ENOENT:
                    result["failed"].append(f"inode {ino}: {e}")

        for parent_ino, name in sorted(entries):
            try:
                pyfuse3.invalidate_entry(parent_ino, name)
                result["entries"] += 1
            except OSError as e:
                if e.errno != errno.ENOENT:
                    result["failed"].append(
                        f"entry {parent_ino}/{os.fsdecode(name)}: {e}"
                    )

        result["ok"] = not result["failed"]
        if not result["ok"]:
            result["error"] = "Some cache entries could not be invalidated"
        log.info("Cache invalidation: %s", result)
        return result

    # -- mount management ---------------------------------------------------

    def add_mount(self, source: str, target: str, writable: bool = False):
        source = os.path.realpath(source)
        if not os.path.exists(source):
            raise ValueError(f"Source does not exist: {source}")
        target = _normalize_mount_target(target)

        with self._lock:
            if target in self._mounts:
                raise ValueError(f"Already mounted: {target}")

            parts = PurePosixPath(target).parts[1:]
            parent = self._inodes[_ROOT_INO]

            # Create intermediate virtual directories
            for i, part in enumerate(parts[:-1]):
                if part in parent.children:
                    child = self._inodes[parent.children[part]]
                    if not child.is_virtual:
                        raise ValueError(
                            f"Intermediate '{'/' + '/'.join(parts[:i+1])}' "
                            f"is already a passthrough mount"
                        )
                    parent = child
                else:
                    child = self._alloc()
                    parent.children[part] = child.ino
                    parent = child

            # Create mount point inode
            name = parts[-1]
            if name in parent.children:
                raise ValueError(
                    f"Path already exists: {target} "
                    f"(virtual directory with children)"
                )

            node = self._alloc(
                real_path=source,
                mount_target=target,
                writable=writable,
            )
            parent.children[name] = node.ino
            parent_ino = parent.ino

            self._mounts[target] = {
                "source": source,
                "target": target,
                "writable": writable,
                "inode": node.ino,
                "hidden": set(),
            }

        # Drop both the cached directory entry and any cached readdir data.
        try:
            pyfuse3.invalidate_entry(parent_ino, name.encode())
            pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Mounted %s -> %s (writable=%s)", source, target, writable)

    def remove_mount(self, target: str):
        target = _normalize_mount_target(target)

        with self._lock:
            if target not in self._mounts:
                raise ValueError(f"Not mounted: {target}")

            info = self._mounts.pop(target)

            self._rebuild_hidden()

            # Remove from parent's children
            parts = PurePosixPath(target).parts[1:]
            parent = self._inodes[_ROOT_INO]
            for part in parts[:-1]:
                parent = self._inodes[parent.children[part]]
            parent_ino = parent.ino

            del parent.children[parts[-1]]

            # Detach every cached inode from this mount generation. Inodes still
            # referenced by the kernel remain until forget(), but cannot be
            # reused if the same target is mounted again with different policy.
            for path_key, ino in list(self._path_to_ino.items()):
                if path_key[0] != target:
                    continue
                del self._path_to_ino[path_key]
                node = self._inodes.get(ino)
                if node:
                    node.active = False
                    if node.lookup_count <= 0:
                        del self._inodes[ino]

            # Clean up empty virtual parents (deepest first)
            for depth in range(len(parts) - 2, -1, -1):
                cur = self._inodes[_ROOT_INO]
                for part in parts[:depth]:
                    cur = self._inodes[cur.children[part]]
                name = parts[depth]
                if name in cur.children:
                    child = self._inodes[cur.children[name]]
                    if child.is_virtual and not child.children:
                        del cur.children[name]
                        if child.lookup_count <= 0:
                            del self._inodes[child.ino]
                    else:
                        break

        try:
            pyfuse3.invalidate_entry(
                parent_ino, parts[-1].encode(), deleted=info["inode"]
            )
            pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Unmounted %s", target)

    def add_hide(self, mount_target: str, rel_path: str):
        mount_target = _normalize_mount_target(mount_target)
        rel_path = _normalize_hidden_path(rel_path)

        with self._lock:
            if mount_target not in self._mounts:
                raise ValueError(f"Not mounted: {mount_target}")

            info = self._mounts[mount_target]
            abs_path = os.path.join(info["source"], rel_path)

            if not os.path.exists(abs_path):
                raise ValueError(f"Path does not exist: {abs_path}")

            info["hidden"].add(rel_path)
            self._rebuild_hidden()

        # Invalidate the parent directory of the hidden path so readdir
        # refreshes and lookup returns ENOENT.
        try:
            parent_real = os.path.dirname(abs_path)
            parent_key = (mount_target, parent_real)
            if parent_key in self._path_to_ino:
                parent_ino = self._path_to_ino[parent_key]
            elif parent_real == info["source"]:
                parent_ino = info["inode"]
            else:
                parent_ino = None
            if parent_ino is not None:
                pyfuse3.invalidate_entry(
                    parent_ino, os.path.basename(abs_path).encode()
                )
                pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Hidden %s in mount %s", rel_path, mount_target)

    def remove_hide(self, mount_target: str, rel_path: str):
        mount_target = _normalize_mount_target(mount_target)
        rel_path = _normalize_hidden_path(rel_path)

        with self._lock:
            if mount_target not in self._mounts:
                raise ValueError(f"Not mounted: {mount_target}")

            info = self._mounts[mount_target]
            abs_path = os.path.join(info["source"], rel_path)

            info["hidden"].discard(rel_path)
            self._rebuild_hidden()

        try:
            parent_real = os.path.dirname(abs_path)
            parent_key = (mount_target, parent_real)
            if parent_key in self._path_to_ino:
                parent_ino = self._path_to_ino[parent_key]
            elif parent_real == info["source"]:
                parent_ino = info["inode"]
            else:
                parent_ino = None
            if parent_ino is not None:
                pyfuse3.invalidate_entry(
                    parent_ino, os.path.basename(abs_path).encode()
                )
                pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Unhidden %s in mount %s", rel_path, mount_target)

    def list_mounts(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "source": m["source"],
                    "target": m["target"],
                    "writable": m["writable"],
                    "hidden": sorted(m["hidden"]),
                }
                for m in self._mounts.values()
            ]

    @staticmethod
    def _normalize_mount_config(config: dict, index: int) -> dict:
        if not isinstance(config, dict):
            raise ValueError(f"Mount {index} must be an object")

        source = config.get("source")
        target = config.get("target")
        writable = config.get("writable", False)
        hidden = config.get("hidden", [])

        if not isinstance(source, str) or not source:
            raise ValueError(f"Mount {index} has an invalid source")
        if not isinstance(target, str):
            raise ValueError(f"Mount {index} has an invalid target")
        if not isinstance(writable, bool):
            raise ValueError(f"Mount {index} has an invalid writable value")
        if not isinstance(hidden, list) or not all(
            isinstance(path, str) for path in hidden
        ):
            raise ValueError(f"Mount {index} has an invalid hidden list")

        return {
            "source": os.path.realpath(source),
            "target": _normalize_mount_target(target),
            "writable": writable,
            "hidden": sorted({_normalize_hidden_path(path) for path in hidden}),
        }

    @staticmethod
    def _same_mount(current: dict, desired: dict) -> bool:
        return (
            current["source"] == desired["source"]
            and current["writable"] == desired["writable"]
            and current["hidden"] == desired["hidden"]
        )

    @staticmethod
    def _targets_overlap(first: str, second: str) -> bool:
        return (
            first == second
            or first.startswith(second + "/")
            or second.startswith(first + "/")
        )

    @staticmethod
    def _validate_mount_paths(config: dict):
        if not os.path.exists(config["source"]):
            raise ValueError(f"Source does not exist: {config['source']}")
        for rel_path in config["hidden"]:
            abs_path = os.path.join(config["source"], rel_path)
            if not os.path.exists(abs_path):
                raise ValueError(f"Hidden path does not exist: {abs_path}")

    def _add_mount_config(self, config: dict):
        self.add_mount(
            config["source"], config["target"], config["writable"]
        )
        try:
            for rel_path in config["hidden"]:
                self.add_hide(config["target"], rel_path)
        except Exception:
            self.remove_mount(config["target"])
            raise

    def load_mounts(self, mounts: list, force: bool = False,
                    replace: bool = False) -> dict:
        if not isinstance(mounts, list):
            raise ValueError("State field 'mounts' must be a list")
        if not isinstance(force, bool) or not isinstance(replace, bool):
            raise ValueError("Load options must be boolean values")

        desired = [
            self._normalize_mount_config(config, index)
            for index, config in enumerate(mounts, start=1)
        ]
        desired_by_target = {m["target"]: m for m in desired}
        if len(desired_by_target) != len(desired):
            raise ValueError("State contains duplicate mount targets")

        targets = list(desired_by_target)
        for index, target in enumerate(targets):
            for other in targets[index + 1:]:
                if self._targets_overlap(target, other):
                    raise ValueError(
                        f"State contains overlapping mount targets: "
                        f"{target}, {other}"
                    )

        result = {
            "loaded": [],
            "unchanged": [],
            "removed": [],
            "conflicts": [],
            "failed": [],
        }

        invalid = {}
        for config in desired:
            try:
                self._validate_mount_paths(config)
            except ValueError as e:
                invalid[config["target"]] = str(e)

        with self._lock:
            current = {
                m["target"]: {
                    "source": m["source"],
                    "target": m["target"],
                    "writable": m["writable"],
                    "hidden": sorted(m["hidden"]),
                }
                for m in self._mounts.values()
            }

            # Replacing with an incomplete state would be destructive, so leave
            # the current state untouched if a mount that needs changing can no
            # longer be restored.
            if replace:
                blocking_invalid = {
                    target: error
                    for target, error in invalid.items()
                    if target not in current
                    or not self._same_mount(
                        current[target], desired_by_target[target]
                    )
                }
                if blocking_invalid:
                    result["failed"] = [
                        {"target": target, "error": error}
                        for target, error in blocking_invalid.items()
                    ]
                    return result

            if replace:
                for target in list(current):
                    if target not in desired_by_target:
                        self.remove_mount(target)
                        result["removed"].append(target)
                        del current[target]

            for config in desired:
                target = config["target"]
                existing = current.get(target)

                if existing and self._same_mount(existing, config):
                    result["unchanged"].append(target)
                    continue

                if target in invalid:
                    result["failed"].append({
                        "target": target,
                        "error": invalid[target],
                    })
                    continue

                overlapping = [
                    mount for current_target, mount in current.items()
                    if self._targets_overlap(current_target, target)
                ]
                if overlapping and not (force or replace):
                    result["conflicts"].append(target)
                    continue

                for mount in overlapping:
                    self.remove_mount(mount["target"])
                    del current[mount["target"]]

                try:
                    self._add_mount_config(config)
                    result["loaded"].append(target)
                    current[target] = config
                    result["removed"].extend(
                        mount["target"] for mount in overlapping
                        if mount["target"] != target
                    )
                except (OSError, ValueError) as e:
                    result["failed"].append({
                        "target": target,
                        "error": str(e),
                    })
                    for mount in overlapping:
                        try:
                            self._add_mount_config(mount)
                            current[mount["target"]] = mount
                        except (OSError, ValueError):
                            log.exception(
                                "Failed to restore mount %s after load error",
                                mount["target"],
                            )

        return result

    # -- FUSE: metadata -----------------------------------------------------

    async def getattr(self, ino, ctx=None):
        node = self._get(ino)
        if node.is_virtual:
            return self._virtual_attr(ino)

        if node.real_path and self._is_hidden(
            node.mount_target, node.real_path
        ):
            raise pyfuse3.FUSEError(errno.ENOENT)

        try:
            st = os.lstat(node.real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        entry = pyfuse3.EntryAttributes()
        self._fill_attr(entry, st, ino, node.writable)
        return entry

    async def lookup(self, parent_ino, name, ctx=None):
        name_s = name.decode() if isinstance(name, bytes) else name
        parent = self._get(parent_ino)

        if parent.is_virtual:
            with self._lock:
                child_ino = parent.children.get(name_s)
                child = self._inodes.get(child_ino)
                if child is None or not child.active:
                    raise pyfuse3.FUSEError(errno.ENOENT)
                child.lookup_count += 1
            if child.is_virtual:
                return self._virtual_attr(child_ino)
            if child.real_path and self._is_hidden(
                child.mount_target, child.real_path
            ):
                raise pyfuse3.FUSEError(errno.ENOENT)
            return await self.getattr(child_ino, ctx)

        # Passthrough directory
        real_path = os.path.join(parent.real_path, name_s)

        if self._is_hidden(parent.mount_target, real_path):
            raise pyfuse3.FUSEError(errno.ENOENT)

        try:
            st = os.lstat(real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            path_key = (parent.mount_target, real_path)
            if path_key in self._path_to_ino:
                child_ino = self._path_to_ino[path_key]
                child = self._inodes[child_ino]
            else:
                child = self._alloc(
                    real_path=real_path,
                    mount_target=parent.mount_target,
                    writable=parent.writable,
                )
                child_ino = child.ino
            child.lookup_count += 1

        entry = pyfuse3.EntryAttributes()
        self._fill_attr(entry, st, child_ino, child.writable)
        return entry

    async def forget(self, inode_list):
        for ino, nlookup in inode_list:
            if ino == _ROOT_INO:
                continue
            with self._lock:
                node = self._inodes.get(ino)
                if node is None:
                    continue
                node.lookup_count -= nlookup
                if node.lookup_count <= 0 and node.real_path is not None:
                    # Don't reclaim mount-point inodes that are still registered
                    is_mount_root = any(
                        m["inode"] == ino for m in self._mounts.values()
                    )
                    if not is_mount_root:
                        path_key = (node.mount_target, node.real_path)
                        if self._path_to_ino.get(path_key) == ino:
                            del self._path_to_ino[path_key]
                        del self._inodes[ino]

    async def setattr(self, ino, attr, fields, fh, ctx):
        node = self._get(ino)
        if node.is_virtual:
            raise pyfuse3.FUSEError(errno.EPERM)
        if not node.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        path = node.real_path
        try:
            if fields.update_size:
                h = self._handles.get(fh) if fh else None
                if h and h.fd >= 0:
                    os.ftruncate(h.fd, attr.st_size)
                else:
                    os.truncate(path, attr.st_size)
            if fields.update_mode:
                os.chmod(path, stat.S_IMODE(attr.st_mode))
            if fields.update_uid or fields.update_gid:
                uid = attr.st_uid if fields.update_uid else -1
                gid = attr.st_gid if fields.update_gid else -1
                os.lchown(path, uid, gid)
            if fields.update_atime or fields.update_mtime:
                st = os.lstat(path)
                atime = attr.st_atime_ns if fields.update_atime else st.st_atime_ns
                mtime = attr.st_mtime_ns if fields.update_mtime else st.st_mtime_ns
                os.utime(path, ns=(atime, mtime), follow_symlinks=False)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        return await self.getattr(ino, ctx)

    async def statfs(self, ctx):
        s = pyfuse3.StatvfsData()
        with self._lock:
            mounts = list(self._mounts.values())
        for m in mounts:
            try:
                st = os.statvfs(m["source"])
                for a in ("f_bsize", "f_frsize", "f_blocks", "f_bfree",
                          "f_bavail", "f_files", "f_ffree", "f_favail",
                          "f_namemax"):
                    setattr(s, a, getattr(st, a))
                return s
            except OSError:
                continue
        return s

    # -- FUSE: directories --------------------------------------------------

    async def opendir(self, ino, ctx):
        node = self._get(ino)
        entries: list[tuple[bytes, pyfuse3.EntryAttributes]] = []

        if node.is_virtual:
            with self._lock:
                children = sorted(node.children.items())
            for name, child_ino in children:
                with self._lock:
                    child = self._inodes.get(child_ino)
                    if child is None or not child.active:
                        continue
                    if child.real_path and self._is_hidden(
                        child.mount_target, child.real_path
                    ):
                        continue
                    # READDIRPLUS caches these as full lookup results.
                    child.lookup_count += 1
                if child.is_virtual:
                    attr = self._virtual_attr(child_ino)
                else:
                    try:
                        st = os.lstat(child.real_path)
                    except OSError:
                        with self._lock:
                            child.lookup_count -= 1
                        continue
                    attr = pyfuse3.EntryAttributes()
                    self._fill_attr(attr, st, child_ino, child.writable)
                entries.append((name.encode(), attr))
        else:
            try:
                with os.scandir(node.real_path) as it:
                    for de in sorted(it, key=lambda e: e.name):
                        rp = os.path.join(node.real_path, de.name)
                        if self._is_hidden(node.mount_target, rp):
                            continue
                        try:
                            st = de.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        # READDIRPLUS: allocate a real FUSE inode so the
                        # kernel can use it for subsequent open/getattr.
                        with self._lock:
                            path_key = (node.mount_target, rp)
                            if path_key in self._path_to_ino:
                                child = self._inodes[
                                    self._path_to_ino[path_key]
                                ]
                            else:
                                child = self._alloc(
                                    real_path=rp,
                                    mount_target=node.mount_target,
                                    writable=node.writable,
                                )
                            child.lookup_count += 1
                        attr = pyfuse3.EntryAttributes()
                        self._fill_attr(attr, st, child.ino, child.writable)
                        entries.append((de.name.encode(), attr))
            except OSError as e:
                raise pyfuse3.FUSEError(e.errno)

        fh = self._next_fh()
        self._handles[fh] = _Handle(ino=ino, dir_entries=entries)
        return fh

    async def readdir(self, fh, start_id, token):
        h = self._handles[fh]
        for idx in range(start_id, len(h.dir_entries)):
            name, attr = h.dir_entries[idx]
            # Keep names/cookies stable for this directory handle, but fetch
            # current metadata and TTLs instead of republishing stale snapshots.
            try:
                attr = await self.getattr(attr.st_ino)
            except pyfuse3.FUSEError as e:
                if e.errno == errno.ENOENT:
                    continue
                raise
            if not pyfuse3.readdir_reply(token, name, attr, idx + 1):
                return

    async def releasedir(self, fh):
        self._handles.pop(fh, None)

    async def mkdir(self, parent_ino, name, mode, ctx):
        name_s = name.decode() if isinstance(name, bytes) else name
        parent = self._get(parent_ino)
        if parent.is_virtual or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        real_path = os.path.join(parent.real_path, name_s)
        try:
            os.mkdir(real_path, mode)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            child = self._alloc(
                real_path=real_path,
                mount_target=parent.mount_target,
                writable=True,
            )
            child.lookup_count += 1
        return await self.getattr(child.ino, ctx)

    async def rmdir(self, parent_ino, name, ctx):
        name_s = name.decode() if isinstance(name, bytes) else name
        parent = self._get(parent_ino)
        if parent.is_virtual or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        real_path = os.path.join(parent.real_path, name_s)
        try:
            os.rmdir(real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

    # -- FUSE: files --------------------------------------------------------

    async def open(self, ino, flags, ctx):
        node = self._get(ino)
        if node.is_virtual:
            raise pyfuse3.FUSEError(errno.EISDIR)

        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_TRUNC)
        if writing and not node.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        try:
            fd = os.open(node.real_path, flags & ~os.O_CREAT)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        fh = self._next_fh()
        self._handles[fh] = _Handle(ino=ino, fd=fd)
        return pyfuse3.FileInfo(fh=fh)

    async def create(self, parent_ino, name, mode, flags, ctx):
        name_s = name.decode() if isinstance(name, bytes) else name
        parent = self._get(parent_ino)
        if parent.is_virtual or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        real_path = os.path.join(parent.real_path, name_s)
        try:
            fd = os.open(real_path, flags | os.O_CREAT, mode)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            child = self._alloc(
                real_path=real_path,
                mount_target=parent.mount_target,
                writable=True,
            )
            child.lookup_count += 1

        fh = self._next_fh()
        self._handles[fh] = _Handle(ino=child.ino, fd=fd)
        entry = await self.getattr(child.ino, ctx)
        return (pyfuse3.FileInfo(fh=fh), entry)

    async def read(self, fh, off, size):
        h = self._handles[fh]
        try:
            os.lseek(h.fd, off, os.SEEK_SET)
            return os.read(h.fd, size)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

    async def write(self, fh, off, buf):
        h = self._handles[fh]
        node = self._inodes.get(h.ino)
        if not node or not node.active or not node.writable:
            raise pyfuse3.FUSEError(errno.EACCES)
        try:
            os.lseek(h.fd, off, os.SEEK_SET)
            return os.write(h.fd, buf)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

    async def release(self, fh):
        h = self._handles.pop(fh, None)
        if h and h.fd >= 0:
            os.close(h.fd)

    async def flush(self, fh):
        h = self._handles.get(fh)
        if h and h.fd >= 0:
            try:
                os.fsync(h.fd)
            except OSError:
                pass

    async def fsync(self, fh, datasync):
        h = self._handles.get(fh)
        if h and h.fd >= 0:
            try:
                (os.fdatasync if datasync else os.fsync)(h.fd)
            except OSError as e:
                raise pyfuse3.FUSEError(e.errno)

    # -- FUSE: path operations ----------------------------------------------

    async def unlink(self, parent_ino, name, ctx):
        name_s = name.decode() if isinstance(name, bytes) else name
        parent = self._get(parent_ino)
        if parent.is_virtual or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        real_path = os.path.join(parent.real_path, name_s)
        try:
            os.unlink(real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

    async def rename(self, p_old_ino, name_old, p_new_ino, name_new, flags, ctx):
        name_old_s = name_old.decode() if isinstance(name_old, bytes) else name_old
        name_new_s = name_new.decode() if isinstance(name_new, bytes) else name_new

        p_old = self._get(p_old_ino)
        p_new = self._get(p_new_ino)
        if p_old.is_virtual or p_new.is_virtual:
            raise pyfuse3.FUSEError(errno.EACCES)
        if not p_old.writable or not p_new.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        old_path = os.path.join(p_old.real_path, name_old_s)
        new_path = os.path.join(p_new.real_path, name_new_s)

        try:
            os.rename(old_path, new_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            old_key = (p_old.mount_target, old_path)
            if old_key in self._path_to_ino:
                ino = self._path_to_ino.pop(old_key)
                node = self._inodes.get(ino)
                if node:
                    node.real_path = new_path
                    node.mount_target = p_new.mount_target
                    self._path_to_ino[(p_new.mount_target, new_path)] = ino

    async def symlink(self, parent_ino, name, target, ctx):
        name_s = name.decode() if isinstance(name, bytes) else name
        target_s = target.decode() if isinstance(target, bytes) else target

        parent = self._get(parent_ino)
        if parent.is_virtual or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        real_path = os.path.join(parent.real_path, name_s)
        try:
            os.symlink(target_s, real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            child = self._alloc(
                real_path=real_path,
                mount_target=parent.mount_target,
                writable=True,
            )
            child.lookup_count += 1
        return await self.getattr(child.ino, ctx)

    async def readlink(self, ino, ctx):
        node = self._get(ino)
        if node.is_virtual:
            raise pyfuse3.FUSEError(errno.EINVAL)
        try:
            return os.readlink(node.real_path).encode()
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

    async def link(self, ino, new_parent_ino, new_name, ctx):
        name_s = new_name.decode() if isinstance(new_name, bytes) else new_name
        node = self._get(ino)
        parent = self._get(new_parent_ino)

        if node.is_virtual or parent.is_virtual:
            raise pyfuse3.FUSEError(errno.EACCES)
        if not node.writable or not parent.writable:
            raise pyfuse3.FUSEError(errno.EACCES)

        new_path = os.path.join(parent.real_path, name_s)
        try:
            os.link(node.real_path, new_path, follow_symlinks=False)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        node.lookup_count += 1
        return await self.getattr(ino, ctx)


# ---------------------------------------------------------------------------
# Control socket server
# ---------------------------------------------------------------------------

class ControlServer:

    def __init__(self, fs: SandboxFS, socket_path: str):
        self._fs = fs
        self._path = socket_path
        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if os.path.exists(self._path):
            os.unlink(self._path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self._path)
        os.chmod(self._path, 0o600)
        self._sock.listen(5)
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Control socket: %s", self._path)

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=3)
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def _loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True
            ).start()

    def _handle(self, conn: socket.socket):
        try:
            conn.settimeout(10.0)
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            if not data.strip():
                return
            req = json.loads(data)
            resp = self._dispatch(req)
            conn.sendall(json.dumps(resp).encode() + b"\n")
        except Exception as e:
            try:
                conn.sendall(
                    json.dumps({"ok": False, "error": str(e)}).encode() + b"\n"
                )
            except Exception:
                pass
        finally:
            conn.close()

    def _dispatch(self, req: dict) -> dict:
        action = req.get("action", "")
        try:
            if action == "mount":
                self._fs.add_mount(
                    req["source"], req["target"], req.get("writable", False)
                )
                return {"ok": True}
            elif action == "unmount":
                self._fs.remove_mount(req["target"])
                return {"ok": True}
            elif action == "hide":
                self._fs.add_hide(req["mount"], req["path"])
                return {"ok": True}
            elif action == "unhide":
                self._fs.remove_hide(req["mount"], req["path"])
                return {"ok": True}
            elif action == "list":
                return {"ok": True, "mounts": self._fs.list_mounts()}
            elif action == "timeouts":
                for key in ("entry_timeout", "attr_timeout"):
                    if key in req:
                        _validate_timeout(req[key])
                return self._fs.set_timeouts(
                    entry_timeout=req.get("entry_timeout"),
                    attr_timeout=req.get("attr_timeout"),
                )
            elif action == "invalidate-cache":
                return self._fs.invalidate_cache()
            elif action == "load":
                result = self._fs.load_mounts(
                    req["mounts"],
                    force=req.get("force", False),
                    replace=req.get("replace", False),
                )
                result["ok"] = not result["conflicts"] and not result["failed"]
                if not result["ok"]:
                    result["error"] = "Some mounts could not be loaded"
                return result
            else:
                return {"ok": False, "error": f"Unknown action: {action}"}
        except (KeyError, ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Control socket client
# ---------------------------------------------------------------------------

def _send(socket_path: str, command: dict) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(10.0)
        sock.connect(socket_path)
        sock.sendall(json.dumps(command).encode() + b"\n")
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return json.loads(data)
    finally:
        sock.close()


def _print_result(resp: dict):
    if resp.get("ok"):
        if "mounts" in resp:
            mounts = resp["mounts"]
            if not mounts:
                print("No mounts")
            for m in mounts:
                rw = "rw" if m["writable"] else "ro"
                hidden = ""
                if m.get("hidden"):
                    hidden = f"  hidden: {', '.join(m['hidden'])}"
                print(f"  {m['source']} -> {m['target']} ({rw}){hidden}")
        else:
            print("OK")
    else:
        print(f"Error: {resp.get('error')}", file=sys.stderr)
        sys.exit(1)


def _print_cache_result(resp: dict):
    if "timeouts" in resp:
        timeouts = resp["timeouts"]
        print(f"Entry timeout: {timeouts['entry_timeout']:g} seconds")
        print(f"Attribute timeout: {timeouts['attr_timeout']:g} seconds")
    if "inodes" in resp:
        print(f"Invalidated: {resp['inodes']} inodes, "
              f"{resp['entries']} directory entries")
    if not resp.get("ok"):
        for failure in resp.get("failed", []):
            print(f"Failed: {failure}", file=sys.stderr)
        _print_result(resp)


def _read_state(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot read state file {path}: {e}")

    if not isinstance(state, dict):
        raise ValueError("State file must contain a JSON object")
    if "mounts" not in state:
        raise ValueError("State file is missing the 'mounts' field")
    return state["mounts"]


def _write_state(path: str, mounts: list):
    try:
        with open(path, "w", encoding="utf-8") as state_file:
            json.dump({"mounts": mounts}, state_file, indent=2)
            state_file.write("\n")
    except OSError as e:
        raise ValueError(f"Cannot write state file {path}: {e}")


def _print_load_result(resp: dict):
    labels = (
        ("loaded", "Loaded"),
        ("unchanged", "Unchanged"),
        ("removed", "Removed"),
        ("conflicts", "Conflict"),
    )
    for key, label in labels:
        for target in resp.get(key, []):
            stream = sys.stderr if key == "conflicts" else sys.stdout
            print(f"{label}: {target}", file=stream)
    for failure in resp.get("failed", []):
        print(
            f"Failed: {failure['target']}: {failure['error']}",
            file=sys.stderr,
        )
    if not resp.get("ok"):
        if not resp.get("conflicts") and not resp.get("failed"):
            print(f"Error: {resp.get('error')}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _only_keyboard_interrupts(error: BaseException) -> bool:
    if isinstance(error, KeyboardInterrupt):
        return True
    nested = getattr(error, "exceptions", ())
    return bool(nested) and all(
        _only_keyboard_interrupts(child) for child in nested
    )


def main():
    p = argparse.ArgumentParser(
        description="Sandbox FUSE filesystem with dynamic mount management"
    )
    sub = p.add_subparsers(dest="cmd")

    # -- daemon --
    d = sub.add_parser("daemon", help="Run the FUSE daemon")
    d.add_argument("mountpoint", help="FUSE mount point")
    d.add_argument("--control-socket", "-s", default=None,
                   help="Control socket path (default: <mountpoint>.sock)")
    d.add_argument("--ro-mount", action="append", default=[],
                   metavar="SRC[:TGT]",
                   help="Read-only mount (repeatable)")
    d.add_argument("--rw-mount", action="append", default=[],
                   metavar="SRC[:TGT]",
                   help="Read-write mount (repeatable)")
    d.add_argument("--hide", action="append", default=[],
                   metavar="TGT:RELPATH",
                   help="Initial hide (repeatable)")
    d.add_argument("--entry-timeout", type=_timeout_argument,
                   default=_DEFAULT_CACHE_TIMEOUT,
                   help="Directory-entry cache TTL in seconds (default: 5; 0 disables)")
    d.add_argument("--attr-timeout", type=_timeout_argument,
                   default=_DEFAULT_CACHE_TIMEOUT,
                   help="Attribute cache TTL in seconds (default: 5; 0 disables)")
    d.add_argument("--debug", action="store_true")

    # -- mount --
    m = sub.add_parser("mount", help="Add a mount to running daemon")
    m.add_argument("-s", "--socket", required=True)
    m.add_argument("source")
    m.add_argument("target", nargs="?", default=None)
    m.add_argument("-w", "--writable", action="store_true")

    # -- unmount --
    u = sub.add_parser("unmount", help="Remove a mount")
    u.add_argument("-s", "--socket", required=True)
    u.add_argument("target")

    # -- hide --
    h = sub.add_parser("hide", help="Hide a path within a mount")
    h.add_argument("-s", "--socket", required=True)
    h.add_argument("mount_target", metavar="mount",
                   help="Mount target path")
    h.add_argument("path", help="Relative path to hide")

    # -- unhide --
    uh = sub.add_parser("unhide", help="Unhide a path")
    uh.add_argument("-s", "--socket", required=True)
    uh.add_argument("mount_target", metavar="mount")
    uh.add_argument("path")

    # -- list --
    ls = sub.add_parser("list", help="List mounts and hides")
    ls.add_argument("-s", "--socket", required=True)

    # -- cache controls --
    timeouts = sub.add_parser(
        "timeouts", help="Show or change cache timeouts in a running daemon"
    )
    timeouts.add_argument("-s", "--socket", required=True)
    timeouts.add_argument(
        "--entry-timeout", type=_timeout_argument, default=None,
        help="Directory-entry cache TTL in seconds (omit to keep current value)",
    )
    timeouts.add_argument(
        "--attr-timeout", type=_timeout_argument, default=None,
        help="Attribute cache TTL in seconds (omit to keep current value)",
    )
    invalidate = sub.add_parser(
        "invalidate-cache", help="Immediately invalidate all cached entries, attributes, and data"
    )
    invalidate.add_argument("-s", "--socket", required=True)

    # -- save --
    save = sub.add_parser("save", help="Save mounts and hides to a JSON file")
    save.add_argument("-s", "--socket", required=True)
    save.add_argument("state_file", metavar="FILE")

    # -- load --
    load = sub.add_parser("load", help="Load mounts and hides from a JSON file")
    load.add_argument("-s", "--socket", required=True)
    load.add_argument("state_file", metavar="FILE")
    load_mode = load.add_mutually_exclusive_group()
    load_mode.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite mounts whose targets conflict",
    )
    load_mode.add_argument(
        "--replace", action="store_true",
        help="Replace all current mounts with the saved mounts",
    )

    args = p.parse_args()

    if args.cmd == "daemon":
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        fs = SandboxFS(
            entry_timeout=args.entry_timeout,
            attr_timeout=args.attr_timeout,
        )

        # Parse initial mounts
        for rw, specs in [(False, args.ro_mount), (True, args.rw_mount)]:
            for spec in specs:
                parts = spec.split(":", 1)
                src = parts[0]
                tgt = parts[1] if len(parts) > 1 else os.path.basename(src)
                try:
                    fs.add_mount(src, tgt, rw)
                except Exception as e:
                    p.error(f"Mount failed ({spec}): {e}")

        # Parse initial hides: TGT:RELPATH
        for spec in args.hide:
            parts = spec.split(":", 1)
            if len(parts) != 2:
                p.error(f"Bad hide spec (need TGT:RELPATH): {spec}")
            try:
                fs.add_hide(parts[0], parts[1])
            except Exception as e:
                p.error(f"Hide failed ({spec}): {e}")

        sock_path = args.control_socket or args.mountpoint.rstrip("/") + ".sock"
        ctl = ControlServer(fs, sock_path)

        fuse_opts = set(pyfuse3.default_options)
        fuse_opts.add("fsname=sandboxfs")
        fuse_opts.add("default_permissions")
        if args.debug:
            fuse_opts.add("debug")

        pyfuse3.init(fs, args.mountpoint, fuse_opts)
        try:
            ctl.start()
            log.info("sandboxfs ready at %s", args.mountpoint)
            trio.run(pyfuse3.main)
        except KeyboardInterrupt:
            pass
        except BaseException as e:
            if not _only_keyboard_interrupts(e):
                raise
        finally:
            ctl.stop()
            pyfuse3.close(unmount=True)

    elif args.cmd == "mount":
        target = args.target or os.path.basename(os.path.realpath(args.source))
        _print_result(_send(args.socket, {
            "action": "mount",
            "source": args.source,
            "target": target,
            "writable": args.writable,
        }))

    elif args.cmd == "unmount":
        _print_result(_send(args.socket, {
            "action": "unmount", "target": args.target,
        }))

    elif args.cmd == "hide":
        _print_result(_send(args.socket, {
            "action": "hide",
            "mount": args.mount_target,
            "path": args.path,
        }))

    elif args.cmd == "unhide":
        _print_result(_send(args.socket, {
            "action": "unhide",
            "mount": args.mount_target,
            "path": args.path,
        }))

    elif args.cmd == "list":
        _print_result(_send(args.socket, {"action": "list"}))

    elif args.cmd == "timeouts":
        command = {"action": "timeouts"}
        for key in ("entry_timeout", "attr_timeout"):
            value = getattr(args, key)
            if value is not None:
                command[key] = value
        _print_cache_result(_send(args.socket, command))

    elif args.cmd == "invalidate-cache":
        _print_cache_result(_send(args.socket, {"action": "invalidate-cache"}))

    elif args.cmd == "save":
        resp = _send(args.socket, {"action": "list"})
        if not resp.get("ok"):
            _print_result(resp)
        try:
            _write_state(args.state_file, resp["mounts"])
        except (KeyError, ValueError) as e:
            p.error(str(e))
        print(f"Saved {len(resp['mounts'])} mount(s) to {args.state_file}")

    elif args.cmd == "load":
        try:
            mounts = _read_state(args.state_file)
        except ValueError as e:
            p.error(str(e))
        _print_load_result(_send(args.socket, {
            "action": "load",
            "mounts": mounts,
            "force": args.force,
            "replace": args.replace,
        }))

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

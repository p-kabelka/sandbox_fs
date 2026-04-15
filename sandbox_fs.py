#!/usr/bin/env python3
"""
Sandbox FUSE filesystem daemon with dynamic mount management.

Provides a virtual filesystem that selectively exposes host directories
with fine-grained read/write control and path hiding. Supports runtime
reconfiguration via a Unix domain socket without restarting.

Usage:
  # Start the daemon
  sandbox_fs.py daemon /mnt/sandbox \\
      -m /home/user/project:/home/user/project:rw \\
      -m /home/user/reference:/home/user/reference \\
      --hide /home/user/project:.env

  # Add a mount at runtime
  sandbox_fs.py mount -s /mnt/sandbox.sock /host/path /virtual/path -w

  # Hide a path within a mount
  sandbox_fs.py hide -s /mnt/sandbox.sock /virtual/mount .secrets

  # Unhide a path
  sandbox_fs.py unhide -s /mnt/sandbox.sock /virtual/mount .secrets

  # List current state
  sandbox_fs.py list -s /mnt/sandbox.sock

  # Remove a mount
  sandbox_fs.py unmount -s /mnt/sandbox.sock /virtual/path

Requires: pyfuse3, trio
Install:  pip install pyfuse3 trio
"""

import argparse
import errno
import json
import logging
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Inode:
    ino: int
    real_path: str | None = None  # None => virtual directory
    writable: bool = False
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

    def __init__(self, entry_timeout: float = 300, attr_timeout: float = 300):
        super().__init__()
        self._lock = threading.Lock()
        self._ino_counter = _ROOT_INO
        self._inodes: dict[int, _Inode] = {}
        self._path_to_ino: dict[str, int] = {}
        self._handles: dict[int, _Handle] = {}
        self._fh_counter = 0
        self._mounts: dict[str, dict] = {}     # target -> mount info
        self._hidden: set[str] = set()          # absolute real paths to hide
        self._entry_timeout = entry_timeout
        self._attr_timeout = attr_timeout

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
        try:
            return self._inodes[ino]
        except KeyError:
            raise pyfuse3.FUSEError(errno.ENOENT)

    def _alloc(self, real_path: str | None = None, writable: bool = False) -> _Inode:
        ino = self._next_ino()
        node = _Inode(ino=ino, real_path=real_path, writable=writable)
        self._inodes[ino] = node
        if real_path is not None:
            self._path_to_ino[real_path] = ino
        return node

    def _is_hidden(self, real_path: str) -> bool:
        for h in self._hidden:
            if real_path == h or real_path.startswith(h + "/"):
                return True
        return False

    # -- attribute helpers --------------------------------------------------

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
        entry.entry_timeout = self._entry_timeout
        entry.attr_timeout = self._attr_timeout

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
        e.entry_timeout = self._entry_timeout
        e.attr_timeout = self._attr_timeout
        return e

    # -- mount management ---------------------------------------------------

    def add_mount(self, source: str, target: str, writable: bool = False):
        source = os.path.realpath(source)
        if not os.path.exists(source):
            raise ValueError(f"Source does not exist: {source}")
        target = "/" + target.strip("/")
        if target == "/":
            raise ValueError("Cannot mount at root")

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

            node = self._alloc(real_path=source, writable=writable)
            parent.children[name] = node.ino
            parent_ino = parent.ino

            self._mounts[target] = {
                "source": source,
                "target": target,
                "writable": writable,
                "inode": node.ino,
                "hidden": set(),
            }

        # Invalidate parent directory in kernel cache
        try:
            pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Mounted %s -> %s (writable=%s)", source, target, writable)

    def remove_mount(self, target: str):
        target = "/" + target.strip("/")

        with self._lock:
            if target not in self._mounts:
                raise ValueError(f"Not mounted: {target}")

            info = self._mounts.pop(target)

            # Remove hidden paths for this mount
            for rel in info["hidden"]:
                self._hidden.discard(os.path.join(info["source"], rel))

            # Remove from parent's children
            parts = PurePosixPath(target).parts[1:]
            parent = self._inodes[_ROOT_INO]
            for part in parts[:-1]:
                parent = self._inodes[parent.children[part]]
            parent_ino = parent.ino

            del parent.children[parts[-1]]

            # Clean up mount inode if not referenced by kernel
            mount_ino = info["inode"]
            node = self._inodes.get(mount_ino)
            if node:
                if node.real_path:
                    self._path_to_ino.pop(node.real_path, None)
                if node.lookup_count <= 0:
                    del self._inodes[mount_ino]

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
            pyfuse3.invalidate_inode(parent_ino)
        except Exception:
            pass
        log.info("Unmounted %s", target)

    def add_hide(self, mount_target: str, rel_path: str):
        mount_target = "/" + mount_target.strip("/")

        with self._lock:
            if mount_target not in self._mounts:
                raise ValueError(f"Not mounted: {mount_target}")

            info = self._mounts[mount_target]
            rel_path = rel_path.strip("/")
            abs_path = os.path.join(info["source"], rel_path)

            if not os.path.exists(abs_path):
                raise ValueError(f"Path does not exist: {abs_path}")

            info["hidden"].add(rel_path)
            self._hidden.add(abs_path)

        # Invalidate the parent directory of the hidden path so readdir
        # refreshes and lookup returns ENOENT.
        try:
            parent_real = os.path.dirname(abs_path)
            if parent_real in self._path_to_ino:
                pyfuse3.invalidate_inode(self._path_to_ino[parent_real])
            elif parent_real == info["source"]:
                pyfuse3.invalidate_inode(info["inode"])
        except Exception:
            pass
        log.info("Hidden %s in mount %s", rel_path, mount_target)

    def remove_hide(self, mount_target: str, rel_path: str):
        mount_target = "/" + mount_target.strip("/")

        with self._lock:
            if mount_target not in self._mounts:
                raise ValueError(f"Not mounted: {mount_target}")

            info = self._mounts[mount_target]
            rel_path = rel_path.strip("/")
            abs_path = os.path.join(info["source"], rel_path)

            info["hidden"].discard(rel_path)
            self._hidden.discard(abs_path)

        try:
            parent_real = os.path.dirname(abs_path)
            if parent_real in self._path_to_ino:
                pyfuse3.invalidate_inode(self._path_to_ino[parent_real])
            elif parent_real == info["source"]:
                pyfuse3.invalidate_inode(info["inode"])
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

    # -- FUSE: metadata -----------------------------------------------------

    async def getattr(self, ino, ctx=None):
        node = self._get(ino)
        if node.is_virtual:
            return self._virtual_attr(ino)

        if node.real_path and self._is_hidden(node.real_path):
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
            if name_s not in parent.children:
                raise pyfuse3.FUSEError(errno.ENOENT)
            child_ino = parent.children[name_s]
            child = self._inodes[child_ino]
            child.lookup_count += 1
            if child.is_virtual:
                return self._virtual_attr(child_ino)
            if child.real_path and self._is_hidden(child.real_path):
                raise pyfuse3.FUSEError(errno.ENOENT)
            return await self.getattr(child_ino, ctx)

        # Passthrough directory
        real_path = os.path.join(parent.real_path, name_s)

        if self._is_hidden(real_path):
            raise pyfuse3.FUSEError(errno.ENOENT)

        try:
            st = os.lstat(real_path)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno)

        with self._lock:
            if real_path in self._path_to_ino:
                child_ino = self._path_to_ino[real_path]
                child = self._inodes[child_ino]
            else:
                child = self._alloc(real_path=real_path, writable=parent.writable)
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
                        self._path_to_ino.pop(node.real_path, None)
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
        for m in self._mounts.values():
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
            for name, child_ino in sorted(node.children.items()):
                child = self._inodes.get(child_ino)
                if child is None:
                    continue
                if child.real_path and self._is_hidden(child.real_path):
                    continue
                # READDIRPLUS: kernel caches these as full lookup results,
                # so we must return proper FUSE inodes and full attributes.
                child.lookup_count += 1
                if child.is_virtual:
                    attr = self._virtual_attr(child_ino)
                else:
                    try:
                        st = os.lstat(child.real_path)
                    except OSError:
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
                        if self._is_hidden(rp):
                            continue
                        try:
                            st = de.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        # READDIRPLUS: allocate a real FUSE inode so the
                        # kernel can use it for subsequent open/getattr.
                        with self._lock:
                            if rp in self._path_to_ino:
                                child = self._inodes[self._path_to_ino[rp]]
                            else:
                                child = self._alloc(
                                    real_path=rp, writable=node.writable
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
            child = self._alloc(real_path=real_path, writable=True)
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
            child = self._alloc(real_path=real_path, writable=True)
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
        if not node or not node.writable:
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
            if old_path in self._path_to_ino:
                ino = self._path_to_ino.pop(old_path)
                node = self._inodes.get(ino)
                if node:
                    node.real_path = new_path
                    self._path_to_ino[new_path] = ino

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
            child = self._alloc(real_path=real_path, writable=True)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    d.add_argument("--mount", "-m", action="append", default=[],
                   metavar="SRC:TGT[:ro|rw]",
                   help="Initial mount (repeatable)")
    d.add_argument("--hide", action="append", default=[],
                   metavar="TGT:RELPATH",
                   help="Initial hide (repeatable)")
    d.add_argument("--entry-timeout", type=float, default=300)
    d.add_argument("--attr-timeout", type=float, default=300)
    d.add_argument("--debug", action="store_true")

    # -- mount --
    m = sub.add_parser("mount", help="Add a mount to running daemon")
    m.add_argument("-s", "--socket", required=True)
    m.add_argument("source")
    m.add_argument("target")
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

        # Parse initial mounts: SRC:TGT[:ro|rw]
        for spec in args.mount:
            parts = spec.split(":")
            if len(parts) < 2:
                p.error(f"Bad mount spec (need SRC:TGT[:ro|rw]): {spec}")
            src, tgt = parts[0], parts[1]
            rw = len(parts) > 2 and parts[2] == "rw"
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
        finally:
            ctl.stop()
            pyfuse3.close(unmount=True)

    elif args.cmd == "mount":
        _print_result(_send(args.socket, {
            "action": "mount",
            "source": args.source,
            "target": args.target,
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

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

import asyncio
import errno
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import pyfuse3  # noqa: F401
except ModuleNotFoundError:
    pyfuse3 = types.ModuleType("pyfuse3")
    pyfuse3.Operations = object
    pyfuse3.ROOT_INODE = 1
    pyfuse3.EntryAttributes = type("EntryAttributes", (), {})
    pyfuse3.FileInfo = types.SimpleNamespace

    class FUSEError(Exception):
        def __init__(self, error):
            self.errno = error
            super().__init__(os.strerror(error))

    pyfuse3.FUSEError = FUSEError
    pyfuse3.default_options = set()
    pyfuse3.invalidate_entry = lambda inode, name, deleted=0: None
    pyfuse3.invalidate_inode = lambda ino: None
    sys.modules["pyfuse3"] = pyfuse3

try:
    import trio  # noqa: F401
except ModuleNotFoundError:
    sys.modules["trio"] = types.ModuleType("trio")

from sandbox_fs import ControlServer, SandboxFS, _read_state, _write_state, main


class StatePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.sources = []
        for name in ("one", "two", "three"):
            source = os.path.join(self.temp_dir.name, name)
            os.mkdir(source)
            self.sources.append(source)

        self.hidden_path = "secret.txt"
        with open(
            os.path.join(self.sources[1], self.hidden_path),
            "w",
            encoding="utf-8",
        ) as hidden_file:
            hidden_file.write("secret")

    def config(self, source, target, writable=False, hidden=None):
        return {
            "source": source,
            "target": target,
            "writable": writable,
            "hidden": hidden or [],
        }

    def test_default_load_merges_and_is_idempotent(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/existing")
        desired = [
            self.config(self.sources[1], "/existing", writable=True),
            self.config(self.sources[1], "/new", hidden=[self.hidden_path]),
        ]

        first = fs.load_mounts(desired)
        state_after_first = fs.list_mounts()
        second = fs.load_mounts(desired)

        self.assertEqual(first["loaded"], ["/new"])
        self.assertEqual(first["conflicts"], ["/existing"])
        self.assertEqual(second["unchanged"], ["/new"])
        self.assertEqual(second["conflicts"], ["/existing"])
        self.assertEqual(fs.list_mounts(), state_after_first)

    def test_force_overwrites_conflicting_target(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/mount")
        desired = [
            self.config(
                self.sources[1],
                "/mount",
                writable=True,
                hidden=[self.hidden_path],
            )
        ]

        first = fs.load_mounts(desired, force=True)
        second = fs.load_mounts(desired, force=True)

        self.assertEqual(first["loaded"], ["/mount"])
        self.assertEqual(second["unchanged"], ["/mount"])
        self.assertEqual(fs.list_mounts(), desired)

    def test_force_does_not_reuse_inodes_with_old_writable_policy(self):
        shared_file = os.path.join(self.sources[0], "shared.txt")
        with open(shared_file, "w", encoding="utf-8") as source_file:
            source_file.write("data")

        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/mount", writable=True)
        old_attr = asyncio.run(fs.lookup(fs._mounts["/mount"]["inode"], b"shared.txt"))
        desired = [self.config(self.sources[0], "/mount")]

        fs.load_mounts(desired, force=True)
        new_attr = asyncio.run(fs.lookup(fs._mounts["/mount"]["inode"], b"shared.txt"))

        self.assertFalse(fs._inodes[old_attr.st_ino].active)
        self.assertNotEqual(old_attr.st_ino, new_attr.st_ino)
        self.assertFalse(fs._inodes[new_attr.st_ino].writable)

        asyncio.run(fs.forget([(old_attr.st_ino, 1)]))
        repeated_attr = asyncio.run(
            fs.lookup(fs._mounts["/mount"]["inode"], b"shared.txt")
        )
        self.assertEqual(repeated_attr.st_ino, new_attr.st_ino)

    def test_replace_removes_mounts_not_in_saved_state(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/old")
        fs.add_mount(self.sources[1], "/keep", writable=True)
        desired = [self.config(self.sources[2], "/replacement")]

        result = fs.load_mounts(desired, replace=True)

        self.assertEqual(result["removed"], ["/old", "/keep"])
        self.assertEqual(result["loaded"], ["/replacement"])
        self.assertEqual(fs.list_mounts(), desired)

    def test_replace_with_invalid_source_leaves_state_unchanged(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/existing")
        original = fs.list_mounts()
        missing = os.path.join(self.temp_dir.name, "missing")

        result = fs.load_mounts([self.config(missing, "/replacement")], replace=True)

        self.assertEqual(result["failed"][0]["target"], "/replacement")
        self.assertEqual(fs.list_mounts(), original)

    def test_replace_rejects_invalid_target_before_removing_mounts(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/existing")
        original = fs.list_mounts()

        with self.assertRaisesRegex(ValueError, "Cannot mount at root"):
            fs.load_mounts([self.config(self.sources[1], ".")], replace=True)

        self.assertEqual(fs.list_mounts(), original)

    def test_force_overwrites_hierarchical_conflict(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/parent/child")
        desired = [self.config(self.sources[1], "/parent")]

        conflict = fs.load_mounts(desired)
        original = fs.list_mounts()
        forced = fs.load_mounts(desired, force=True)
        repeated = fs.load_mounts(desired, force=True)

        self.assertEqual(conflict["conflicts"], ["/parent"])
        self.assertEqual(original[0]["target"], "/parent/child")
        self.assertEqual(forced["removed"], ["/parent/child"])
        self.assertEqual(forced["loaded"], ["/parent"])
        self.assertEqual(repeated["unchanged"], ["/parent"])
        self.assertEqual(fs.list_mounts(), desired)

    def test_replace_preserves_shared_source_hide(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[1], "/remove")
        fs.add_hide("/remove", self.hidden_path)
        fs.add_mount(self.sources[1], "/keep")
        fs.add_hide("/keep", self.hidden_path)
        desired = [self.config(self.sources[1], "/keep", hidden=[self.hidden_path])]

        result = fs.load_mounts(desired, replace=True)

        self.assertEqual(result["removed"], ["/remove"])
        self.assertEqual(result["unchanged"], ["/keep"])
        self.assertTrue(
            fs._is_hidden("/keep", os.path.join(self.sources[1], self.hidden_path))
        )
        self.assertEqual(fs.list_mounts(), desired)

    def test_shared_source_mounts_keep_independent_policy(self):
        shared_file = os.path.join(self.sources[0], "shared.txt")
        with open(shared_file, "w", encoding="utf-8") as source_file:
            source_file.write("data")

        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/read-only")
        fs.add_hide("/read-only", "shared.txt")
        fs.add_mount(self.sources[0], "/read-write", writable=True)

        self.assertTrue(fs._is_hidden("/read-only", shared_file))
        self.assertFalse(fs._is_hidden("/read-write", shared_file))

        fs.remove_hide("/read-only", "shared.txt")
        ro_attr = asyncio.run(
            fs.lookup(fs._mounts["/read-only"]["inode"], b"shared.txt")
        )
        rw_attr = asyncio.run(
            fs.lookup(fs._mounts["/read-write"]["inode"], b"shared.txt")
        )
        self.assertNotEqual(ro_attr.st_ino, rw_attr.st_ino)
        self.assertFalse(fs._inodes[ro_attr.st_ino].writable)
        self.assertTrue(fs._inodes[rw_attr.st_ino].writable)

    def test_dynamic_changes_invalidate_kernel_entries(self):
        fs = SandboxFS()
        with (
            mock.patch.object(pyfuse3, "invalidate_entry") as invalidate_entry,
            mock.patch.object(pyfuse3, "invalidate_inode"),
        ):
            fs.add_mount(self.sources[1], "/mount")
            mount_ino = fs._mounts["/mount"]["inode"]
            invalidate_entry.assert_called_with(pyfuse3.ROOT_INODE, b"mount")

            fs.add_hide("/mount", self.hidden_path)
            invalidate_entry.assert_called_with(mount_ino, self.hidden_path.encode())

            fs.remove_hide("/mount", self.hidden_path)
            invalidate_entry.assert_called_with(mount_ino, self.hidden_path.encode())

            fs.remove_mount("/mount")
            invalidate_entry.assert_called_with(
                pyfuse3.ROOT_INODE, b"mount", deleted=mount_ino
            )

    def test_load_rejects_empty_hidden_path(self):
        fs = SandboxFS()

        with self.assertRaisesRegex(ValueError, "Hidden path cannot be empty"):
            fs.load_mounts([self.config(self.sources[0], "/mount", hidden=["."])])

    def test_state_file_is_two_space_indented_json(self):
        state_path = os.path.join(self.temp_dir.name, "state.json")
        mounts = [self.config(self.sources[0], "/mount")]

        _write_state(state_path, mounts)

        with open(state_path, encoding="utf-8") as state_file:
            contents = state_file.read()
        self.assertEqual(
            contents,
            json.dumps({"mounts": mounts}, indent=2) + "\n",
        )
        self.assertEqual(_read_state(state_path), mounts)

    def test_invalid_utf8_state_file_has_controlled_error(self):
        state_path = os.path.join(self.temp_dir.name, "state.json")
        with open(state_path, "wb") as state_file:
            state_file.write(b"\xff")

        with self.assertRaisesRegex(ValueError, "Cannot read state file"):
            _read_state(state_path)


class PathWritePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.source = self.temp_dir.name
        os.mkdir(os.path.join(self.source, ".git"))
        os.mkdir(os.path.join(self.source, ".git", "objects"))
        for path in (".git/config", ".git/objects/data", ".git-backup", "work.txt"):
            with open(os.path.join(self.source, path), "w", encoding="utf-8") as f:
                f.write("original")
        # Notifications require a live FUSE session when real pyfuse3 is installed.
        inode_patch = mock.patch.object(pyfuse3, "invalidate_inode")
        entry_patch = mock.patch.object(pyfuse3, "invalidate_entry")
        self.inodes = inode_patch.start()
        self.entries = entry_patch.start()
        self.addCleanup(inode_patch.stop)
        self.addCleanup(entry_patch.stop)
        self.fs = SandboxFS(entry_timeout=300, attr_timeout=300)
        self.fs.add_mount(self.source, "/project", writable=True)
        self.root = self.fs._mounts["/project"]["inode"]
        self.control = ControlServer(self.fs, os.path.join(self.source, "control.sock"))

    def lookup(self, path, fs=None, root=None):
        fs = fs or self.fs
        ino = self.root if root is None else root
        for name in path.split("/"):
            ino = asyncio.run(fs.lookup(ino, os.fsencode(name))).st_ino
        return ino

    def assert_denied(self, operation):
        with self.assertRaises(pyfuse3.FUSEError) as raised:
            asyncio.run(operation)
        self.assertEqual(raised.exception.errno, errno.EACCES)

    def open_handle(self, ino, flags=os.O_RDWR):
        fh = asyncio.run(self.fs.open(ino, flags, None)).fh
        self.addCleanup(lambda: asyncio.run(self.fs.release(fh)))
        return fh

    def test_rule_updates_cached_and_new_descendants_but_not_prefix_siblings(self):
        config = self.lookup(".git/config")
        self.assertTrue(asyncio.run(self.fs.getattr(config)).st_mode & 0o222)
        self.fs.add_readonly("/project", ".git")
        for path in (".git", ".git/config", ".git/objects/data"):
            ino = self.lookup(path)
            self.assertFalse(asyncio.run(self.fs.getattr(ino)).st_mode & 0o222)
            self.assertFalse(self.fs._inodes[ino].writable)
        for path in ("work.txt", ".git-backup"):
            ino = self.lookup(path)
            fh = self.open_handle(ino)
            self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"edited")), 6)
        self.assertTrue(
            os.stat(os.path.join(self.source, ".git/config")).st_mode & 0o222
        )

    def test_directory_handles_publish_current_readonly_attributes(self):
        fh = asyncio.run(self.fs.opendir(self.root, None))
        self.addCleanup(lambda: asyncio.run(self.fs.releasedir(fh)))
        self.fs.add_readonly("/project", ".git")
        replies = {}

        def reply(token, name, attr, cookie):
            replies[name] = attr
            return True

        with mock.patch.object(
            pyfuse3, "readdir_reply", side_effect=reply, create=True
        ):
            asyncio.run(self.fs.readdir(fh, 0, object()))
        self.assertFalse(replies[b".git"].st_mode & 0o222)
        self.assertTrue(replies[b"work.txt"].st_mode & 0o222)

        git = self.lookup(".git")
        git_fh = asyncio.run(self.fs.opendir(git, None))
        self.addCleanup(lambda: asyncio.run(self.fs.releasedir(git_fh)))
        with mock.patch.object(
            pyfuse3, "readdir_reply", side_effect=reply, create=True
        ):
            asyncio.run(self.fs.readdir(git_fh, 0, object()))
        self.assertFalse(replies[b"config"].st_mode & 0o222)

    def test_readonly_blocks_new_and_existing_writers_but_allows_reads(self):
        ino = self.lookup(".git/config")
        fh = self.open_handle(ino)
        self.fs.add_readonly("/project", ".git")
        self.assert_denied(self.fs.write(fh, 0, b"blocked"))
        for flags in (os.O_WRONLY, os.O_RDWR, os.O_RDONLY | os.O_TRUNC, os.O_APPEND):
            with self.subTest(flags=flags):
                self.assert_denied(self.fs.open(ino, flags, None))
        reader = self.open_handle(ino, os.O_RDONLY)
        self.assertEqual(asyncio.run(self.fs.read(reader, 0, 100)), b"original")
        self.fs.add_writable("/project", ".git")
        self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"allowed!")), 8)
        self.assertTrue(asyncio.run(self.fs.getattr(ino)).st_mode & 0o222)

    def test_metadata_changes_and_fd_truncation_are_blocked(self):
        ino = self.lookup(".git/config")
        fh = self.open_handle(ino)
        self.fs.add_readonly("/project", ".git")
        attr = types.SimpleNamespace(
            st_size=0,
            st_mode=0o777,
            st_uid=os.getuid(),
            st_gid=os.getgid(),
            st_atime_ns=0,
            st_mtime_ns=0,
        )
        for field in ("size", "mode", "uid", "gid", "atime", "mtime"):
            fields = types.SimpleNamespace(
                **{
                    "update_" + name: name == field
                    for name in ("size", "mode", "uid", "gid", "atime", "mtime")
                }
            )
            with self.subTest(field=field):
                self.assert_denied(self.fs.setattr(ino, attr, fields, fh, None))
        with open(os.path.join(self.source, ".git/config"), "rb") as f:
            self.assertEqual(f.read(), b"original")

    def test_creation_links_deletion_and_renames_cannot_modify_subtree(self):
        git = self.lookup(".git")
        config = self.lookup(".git/config")
        work = self.lookup("work.txt")
        self.fs.add_readonly("/project", ".git")
        operations = {
            "create": lambda: self.fs.create(git, b"new", 0o644, os.O_WRONLY, None),
            "mkdir": lambda: self.fs.mkdir(git, b"new", 0o755, None),
            "symlink": lambda: self.fs.symlink(git, b"new", b"../work.txt", None),
            "link in": lambda: self.fs.link(work, git, b"new", None),
            "link out": lambda: self.fs.link(config, self.root, b"alias", None),
            "unlink": lambda: self.fs.unlink(git, b"config", None),
            "rmdir": lambda: self.fs.rmdir(git, b"objects", None),
            "remove root": lambda: self.fs.rmdir(self.root, b".git", None),
            "rename out": lambda: self.fs.rename(
                git, b"config", self.root, b"moved", 0, None
            ),
            "rename in": lambda: self.fs.rename(
                self.root, b"work.txt", git, b"new", 0, None
            ),
            "rename root": lambda: self.fs.rename(
                self.root, b".git", self.root, b"moved", 0, None
            ),
            "replace root": lambda: self.fs.rename(
                self.root, b"work.txt", self.root, b".git", 0, None
            ),
        }
        for name, operation in operations.items():
            with self.subTest(operation=name):
                self.assert_denied(operation())
        self.assertTrue(os.path.exists(os.path.join(self.source, ".git/config")))
        self.assertTrue(os.path.exists(os.path.join(self.source, "work.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.source, "alias")))

    def test_file_rule_cannot_be_bypassed_through_its_writable_parent(self):
        ino = self.lookup("work.txt")
        self.fs.add_readonly("/project", "work.txt")
        self.assert_denied(
            self.fs.create(
                self.root, b"work.txt", 0o644, os.O_WRONLY | os.O_TRUNC, None
            )
        )
        self.assert_denied(self.fs.unlink(self.root, b"work.txt", None))
        self.assert_denied(
            self.fs.rename(self.root, b"work.txt", self.root, b"moved", 0, None)
        )
        self.assert_denied(
            self.fs.rename(self.root, b".git-backup", self.root, b"work.txt", 0, None)
        )
        self.assert_denied(self.fs.link(ino, self.root, b"alias", None))

        # Rules stay anchored even if the host removes the protected path.
        os.unlink(os.path.join(self.source, "work.txt"))
        other = self.lookup(".git-backup")
        for operation in (
            lambda: self.fs.create(self.root, b"work.txt", 0o644, os.O_WRONLY, None),
            lambda: self.fs.mkdir(self.root, b"work.txt", 0o755, None),
            lambda: self.fs.symlink(self.root, b"work.txt", b".git-backup", None),
            lambda: self.fs.link(other, self.root, b"work.txt", None),
        ):
            self.assert_denied(operation())

    def test_renaming_ancestor_cannot_move_or_replace_protected_path(self):
        os.mkdir(os.path.join(self.source, "parent"))
        os.mkdir(os.path.join(self.source, "parent", "protected"))
        os.mkdir(os.path.join(self.source, "replacement"))
        self.fs.add_readonly("/project", "parent/protected")
        self.assert_denied(
            self.fs.rename(self.root, b"parent", self.root, b"moved", 0, None)
        )
        self.assert_denied(
            self.fs.rename(self.root, b"replacement", self.root, b"parent", 0, None)
        )

    def test_writable_directory_rename_keeps_cached_descendants_under_policy(self):
        ino = self.lookup(".git/objects/data")
        fh = self.open_handle(ino)
        asyncio.run(self.fs.rename(self.root, b".git", self.root, b"moved", 0, None))
        self.assertEqual(self.lookup("moved/objects/data"), ino)
        self.fs.add_readonly("/project", "moved")
        self.assert_denied(self.fs.write(fh, 0, b"blocked"))
        self.assertFalse(asyncio.run(self.fs.getattr(ino)).st_mode & 0o222)

    def test_child_override_wins_and_parent_updates_preserve_child_rules(self):
        self.fs.add_readonly("/project", ".git")
        self.fs.add_writable("/project", ".git/config")
        self.open_handle(self.lookup(".git/config"))
        self.assert_denied(
            self.fs.open(self.lookup(".git/objects/data"), os.O_RDWR, None)
        )
        self.fs.add_readonly("/project", ".git")
        self.open_handle(self.lookup(".git/config"))
        self.fs.add_readonly("/project", ".git/config")
        self.fs.add_writable("/project", ".git")
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))
        self.open_handle(self.lookup(".git/objects/data"))

    def test_mount_defaults_preserve_subpath_rules_and_other_mounts(self):
        self.fs.add_mount(self.source, "/other", writable=True)
        other = self.fs._mounts["/other"]["inode"]
        other_work = self.lookup("work.txt", root=other)
        work = self.lookup("work.txt")
        work_fh = self.open_handle(work)
        self.fs.add_readonly("/project", ".git")
        self.fs.add_writable("/project", ".git/objects")
        self.fs.add_hide("/project", ".git-backup")
        config = self.lookup(".git/config")
        data = self.lookup(".git/objects/data")
        data_fh = self.open_handle(data)
        original = self.fs.list_mounts()

        def invalidate(ino):
            self.assertFalse(self.fs._lock._is_owned())

        self.inodes.reset_mock()
        self.entries.reset_mock()
        self.inodes.side_effect = invalidate
        for action in ("readonly", "writable", "writable", "readonly"):
            with self.subTest(action=action):
                result = self.control._dispatch({"action": action, "mount": "project/"})
                self.assertTrue(result["ok"])
                writable = action == "writable"
                self.assertEqual(
                    self.fs.list_mounts(),
                    [{**original[0], "writable": writable}, original[1]],
                )
                self.assertEqual(self.fs._mounts["/project"]["inode"], self.root)
                for ino in (self.root, work):
                    self.assertEqual(self.fs._inodes[ino].writable, writable)
                    self.assertEqual(
                        bool(asyncio.run(self.fs.getattr(ino)).st_mode & 0o222),
                        writable,
                    )
                self.assert_denied(self.fs.open(config, os.O_RDWR, None))
                self.assertEqual(asyncio.run(self.fs.write(data_fh, 0, b"allowed!")), 8)
                self.assertEqual(self.lookup(".git/objects/data"), data)
                if writable:
                    self.assertEqual(asyncio.run(self.fs.write(work_fh, 0, b"edit")), 4)
                else:
                    self.assert_denied(self.fs.write(work_fh, 0, b"blocked"))
                    self.assert_denied(self.fs.open(work, os.O_RDWR, None))
                self.open_handle(other_work)
                with self.assertRaises(pyfuse3.FUSEError) as raised:
                    self.lookup(".git-backup")
                self.assertEqual(raised.exception.errno, errno.ENOENT)
        for ino in (self.root, work, config, data):
            self.inodes.assert_any_call(ino)
        self.entries.assert_any_call(pyfuse3.ROOT_INODE, b"project")
        self.entries.assert_any_call(self.root, b"work.txt")

    def test_commands_are_opposites_and_override_either_mount_default(self):
        for default in (False, True):
            target = f"/default-{default}"
            self.fs.add_mount(self.source, target, writable=default)
            root = self.fs._mounts[target]["inode"]
            ino = self.lookup(".git/config", root=root)
            for action in ("writable", "readonly", "writable", "readonly"):
                with self.subTest(default=default, action=action):
                    result = self.control._dispatch(
                        {
                            "action": action,
                            "mount": target,
                            "path": "./.git//config/",
                        }
                    )
                    self.assertTrue(result["ok"])
                    state = next(
                        m for m in self.fs.list_mounts() if m["target"] == target
                    )
                    writable = action == "writable"
                    self.assertEqual(
                        bool(asyncio.run(self.fs.getattr(ino)).st_mode & 0o222),
                        writable,
                    )
                    self.assertEqual(
                        state.get("readwrite", []), [".git/config"] if writable else []
                    )
                    self.assertEqual(
                        state.get("readonly", []), [] if writable else [".git/config"]
                    )
                    if writable:
                        fh = self.open_handle(ino)
                        self.assertEqual(
                            asyncio.run(self.fs.write(fh, 0, b"allowed!")), 8
                        )
                    else:
                        self.assert_denied(self.fs.open(ino, os.O_RDWR, None))
                    sibling = self.lookup(".git-backup", root=root)
                    self.assertEqual(
                        bool(asyncio.run(self.fs.getattr(sibling)).st_mode & 0o222),
                        default,
                    )

    def test_writable_subtree_on_readonly_mount_allows_mutations_inside_only(self):
        self.fs.add_mount(self.source, "/ro")
        ro = self.fs._mounts["/ro"]["inode"]
        git = self.lookup(".git", root=ro)
        config = self.lookup(".git/config", root=ro)
        self.assert_denied(self.fs.open(config, os.O_RDWR, None))
        self.fs.add_writable("/ro", ".git")
        self.assertTrue(asyncio.run(self.fs.getattr(git)).st_mode & 0o222)
        fh = self.open_handle(config)
        self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"allowed!")), 8)
        self.open_handle(self.lookup(".git/objects/data", root=ro))

        info, attr = asyncio.run(self.fs.create(git, b"new", 0o644, os.O_RDWR, None))
        self.addCleanup(lambda: asyncio.run(self.fs.release(info.fh)))
        self.assertTrue(attr.st_mode & 0o222)
        self.assertEqual(asyncio.run(self.fs.write(info.fh, 0, b"created")), 7)
        asyncio.run(self.fs.mkdir(git, b"directory", 0o755, None))
        asyncio.run(self.fs.symlink(git, b"symbolic", b"config", None))
        asyncio.run(self.fs.link(config, git, b"hardlink", None))
        asyncio.run(self.fs.rename(git, b"new", git, b"renamed", 0, None))
        asyncio.run(self.fs.unlink(git, b"renamed", None))
        asyncio.run(self.fs.unlink(git, b"symbolic", None))
        asyncio.run(self.fs.unlink(git, b"hardlink", None))
        asyncio.run(self.fs.rmdir(git, b"directory", None))

        self.assert_denied(self.fs.create(ro, b"outside", 0o644, os.O_RDWR, None))
        self.assert_denied(
            self.fs.open(self.lookup("work.txt", root=ro), os.O_RDWR, None)
        )
        self.assert_denied(self.fs.rename(ro, b".git", ro, b"moved", 0, None))
        self.assert_denied(self.fs.rmdir(ro, b".git", None))
        self.assertFalse(asyncio.run(self.fs.getattr(ro)).st_mode & 0o222)
        self.fs.add_readonly("/ro", ".git")
        self.assert_denied(self.fs.write(fh, 0, b"blocked"))

    def test_writable_file_exception_can_be_edited_but_parent_remains_readonly(self):
        self.fs.add_mount(self.source, "/ro")
        ro = self.fs._mounts["/ro"]["inode"]
        ino = self.lookup("work.txt", root=ro)
        self.fs.add_writable("/ro", "work.txt")
        fh = self.open_handle(ino)
        self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"allowed!")), 8)
        fields = types.SimpleNamespace(
            **{
                "update_" + name: name == "size"
                for name in ("size", "mode", "uid", "gid", "atime", "mtime")
            }
        )
        asyncio.run(
            self.fs.setattr(ino, types.SimpleNamespace(st_size=1), fields, fh, None)
        )
        self.assertEqual(asyncio.run(self.fs.getattr(ino)).st_size, 1)
        self.assert_denied(self.fs.unlink(ro, b"work.txt", None))
        self.assert_denied(self.fs.rename(ro, b"work.txt", ro, b"moved", 0, None))

    def test_alternating_nested_overrides_use_most_specific_rule(self):
        self.fs.add_mount(self.source, "/ro")
        ro = self.fs._mounts["/ro"]["inode"]
        data = self.lookup(".git/objects/data", root=ro)
        self.fs.add_writable("/ro", ".git")
        self.fs.add_readonly("/ro", ".git/objects")
        self.assert_denied(self.fs.open(data, os.O_RDWR, None))
        self.fs.add_writable("/ro", ".git/objects/data")
        self.open_handle(data)
        self.fs.add_readonly("/ro", ".git")
        self.open_handle(data)
        self.assert_denied(
            self.fs.open(self.lookup(".git/config", root=ro), os.O_RDWR, None)
        )
        self.fs.add_writable("/ro", ".git")
        objects = self.lookup(".git/objects", root=ro)
        self.assert_denied(self.fs.mkdir(objects, b"new", 0o755, None))
        self.assert_denied(
            self.fs.rename(
                self.lookup(".git", root=ro),
                b"objects",
                self.lookup(".git", root=ro),
                b"moved",
                0,
                None,
            )
        )
        self.open_handle(data)

    def test_same_source_mounts_keep_independent_readonly_rules(self):
        self.fs.add_mount(self.source, "/other", writable=True)
        other = self.fs._mounts["/other"]["inode"]
        self.fs.add_readonly("/project", ".git")
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))
        ino = self.lookup(".git/config", root=other)
        fh = self.open_handle(ino)
        self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"other")), 5)
        self.fs.remove_mount("/project")
        self.fs.add_mount(self.source, "/project", writable=True)
        self.root = self.fs._mounts["/project"]["inode"]
        self.open_handle(self.lookup(".git/config"))

    def test_unhide_readonly_is_atomic_and_invalidates_uncached_hidden_entry(self):
        self.fs.add_hide("/project", ".git")

        def invalidate(ino):
            self.assertFalse(self.fs._lock._is_owned())
            config = self.lookup(".git/config")
            self.assert_denied(self.fs.open(config, os.O_RDWR, None))

        self.inodes.side_effect = invalidate
        result = self.fs.remove_hide("/project", ".git", readonly=True)
        self.assertTrue(result["ok"])
        self.entries.assert_any_call(self.root, b".git")
        self.assertEqual(self.fs.list_mounts()[0]["hidden"], [])
        self.assertEqual(self.fs.list_mounts()[0]["readonly"], [".git"])
        self.fs.add_hide("/project", ".git")
        self.fs.remove_hide("/project", ".git")
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))

    def test_failed_unhide_readonly_leaves_hide_and_policy_unchanged(self):
        self.fs.add_hide("/project", "work.txt")
        original = self.fs.list_mounts()
        os.unlink(os.path.join(self.source, "work.txt"))
        for policy in ("readonly", "writable"):
            result = self.control._dispatch(
                {
                    "action": "unhide",
                    "mount": "/project",
                    "path": "work.txt",
                    policy: True,
                }
            )
            self.assertFalse(result["ok"])
            self.assertEqual(self.fs.list_mounts(), original)

    def test_unhide_writable_is_atomic_and_plain_hide_unhide_preserves_override(self):
        self.fs.add_mount(self.source, "/ro")
        ro = self.fs._mounts["/ro"]["inode"]
        self.fs.add_hide("/ro", ".git")

        def invalidate(ino):
            self.assertFalse(self.fs._lock._is_owned())
            self.open_handle(self.lookup(".git/config", root=ro))

        self.inodes.side_effect = invalidate
        result = self.control._dispatch(
            {
                "action": "unhide",
                "mount": "/ro",
                "path": ".git",
                "writable": True,
            }
        )
        self.assertTrue(result["ok"])
        self.entries.assert_any_call(ro, b".git")
        self.fs.add_hide("/ro", ".git")
        self.fs.remove_hide("/ro", ".git")
        self.open_handle(self.lookup(".git/config", root=ro))

    def test_cache_notification_failure_reports_error_but_keeps_rule_enforced(self):
        ino = self.lookup("work.txt")
        self.inodes.side_effect = OSError(errno.EIO, "notification failed")
        result = self.control._dispatch(
            {
                "action": "readonly",
                "mount": "/project",
                "path": "work.txt",
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("policy updated", result["error"])
        self.assert_denied(self.fs.open(ino, os.O_RDWR, None))
        result = self.control._dispatch(
            {
                "action": "writable",
                "mount": "/project",
                "path": "work.txt",
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("policy updated", result["error"])
        self.open_handle(ino)

    def test_mount_notification_failure_keeps_new_default_enforced(self):
        work = self.lookup("work.txt")
        fh = self.open_handle(work)
        self.fs.add_readonly("/project", ".git")
        config = self.lookup(".git/config")
        self.inodes.side_effect = OSError(errno.EIO, "notification failed")
        for action in ("readonly", "writable"):
            with self.subTest(action=action):
                result = self.control._dispatch({"action": action, "mount": "/project"})
                self.assertFalse(result["ok"])
                self.assertIn("Mount write policy updated", result["error"])
                self.assert_denied(self.fs.open(config, os.O_RDWR, None))
                if action == "writable":
                    self.assertEqual(asyncio.run(self.fs.write(fh, 0, b"allowed!")), 8)
                else:
                    self.assert_denied(self.fs.write(fh, 0, b"blocked"))
                    self.assert_denied(self.fs.open(work, os.O_RDWR, None))

    def test_invalid_rules_are_rejected_without_changing_state(self):
        original = self.fs.list_mounts()
        for action in ("readonly", "writable"):
            result = self.control._dispatch({"action": action, "mount": "/missing"})
            self.assertFalse(result["ok"])
            self.assertEqual(self.fs.list_mounts(), original)
            result = self.control._dispatch(
                {"action": "unhide", "mount": "/project", "path": None, action: True}
            )
            self.assertFalse(result["ok"])
            self.assertEqual(self.fs.list_mounts(), original)
            for path in (
                "",
                ".",
                "../work.txt",
                ".git/../work.txt",
                "\0",
                123,
                "missing",
            ):
                with self.subTest(action=action, path=path):
                    result = self.control._dispatch(
                        {
                            "action": action,
                            "mount": "/project",
                            "path": path,
                        }
                    )
                    self.assertFalse(result["ok"])
                    self.assertEqual(self.fs.list_mounts(), original)
            for value in ("true", 1, None):
                result = self.control._dispatch(
                    {
                        "action": "unhide",
                        "mount": "/project",
                        "path": "work.txt",
                        action: value,
                    }
                )
                self.assertFalse(result["ok"])
                self.assertEqual(self.fs.list_mounts(), original)
        result = self.control._dispatch(
            {
                "action": "unhide",
                "mount": "/project",
                "path": "work.txt",
                "readonly": True,
                "writable": True,
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(self.fs.list_mounts(), original)

    def test_cli_commands_round_trip_and_state_restores_readonly_rules(self):
        self.control.start()
        self.addCleanup(self.control.stop)
        state_path = os.path.join(self.source, "state.json")

        def cli(*args):
            output = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "sandbox_fs.py",
                        args[0],
                        "-s",
                        self.control._path,
                        *args[1:],
                    ],
                ),
                mock.patch.object(sys, "stdout", output),
            ):
                main()
            return output.getvalue()

        cli("readonly", "/project", ".git")
        self.assertIn("readonly: .git", cli("list"))
        cli("writable", "/project", ".git")
        self.assertIn("readwrite: .git", cli("list"))
        self.open_handle(self.lookup(".git/config"))
        cli("hide", "/project", ".git")
        cli("unhide", "/project", ".git", "--readonly")
        cli("save", state_path)
        self.assertEqual(_read_state(state_path)[0]["readonly"], [".git"])
        cli("writable", "/project", ".git")
        self.assertIn("Loaded: /project", cli("load", state_path, "--force"))
        self.root = self.fs._mounts["/project"]["inode"]
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))
        self.assertIn("Unchanged: /project", cli("load", state_path))

    def test_cli_unhide_writable_and_state_round_trip_on_readonly_mount(self):
        self.fs.add_mount(self.source, "/reference")
        self.control.start()
        self.addCleanup(self.control.stop)
        state_path = os.path.join(self.source, "state.json")

        def cli(*args):
            output = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "sandbox_fs.py",
                        args[0],
                        "-s",
                        self.control._path,
                        *args[1:],
                    ],
                ),
                mock.patch.object(sys, "stdout", output),
            ):
                main()
            return output.getvalue()

        cli("hide", "/reference", ".git")
        cli("unhide", "/reference", ".git", "--writable")
        cli("readonly", "/reference", ".git/config")
        cli("save", state_path)
        saved = next(m for m in _read_state(state_path) if m["target"] == "/reference")
        self.assertFalse(saved["writable"])
        self.assertEqual(saved["readwrite"], [".git"])
        self.assertEqual(saved["readonly"], [".git/config"])
        cli("readonly", "/reference", ".git")
        self.assertIn("Loaded: /reference", cli("load", state_path, "--force"))
        ro = self.fs._mounts["/reference"]["inode"]
        self.assertFalse(asyncio.run(self.fs.getattr(ro)).st_mode & 0o222)
        self.open_handle(self.lookup(".git/objects/data", root=ro))
        self.assert_denied(
            self.fs.open(self.lookup(".git/config", root=ro), os.O_RDWR, None)
        )
        self.assertIn("Unchanged: /reference", cli("load", state_path))
        self.assertIn("readwrite: .git", cli("list"))

    def test_cli_mount_defaults_preserve_hidden_and_loaded_write_rules(self):
        self.control.start()
        self.addCleanup(self.control.stop)
        state_path = os.path.join(self.source, "state.json")

        def cli(*args):
            output = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["sandbox_fs.py", args[0], "-s", self.control._path, *args[1:]],
                ),
                mock.patch.object(sys, "stdout", output),
            ):
                main()
            return output.getvalue()

        cli("readonly", "project")
        cli("hide", "project", ".git")
        cli("writable", "/project")
        self.assertIn("(rw)", cli("list"))
        self.assertIn("hidden: .git", cli("list"))
        with self.assertRaises(pyfuse3.FUSEError) as raised:
            self.lookup(".git")
        self.assertEqual(raised.exception.errno, errno.ENOENT)

        # Restore a writable mount with a hidden .git, then unhide it read-only.
        cli("save", state_path)
        cli("readonly", "/project")
        self.assertIn("Loaded: /project", cli("load", state_path, "--force"))
        self.root = self.fs._mounts["/project"]["inode"]
        cli("unhide", "--readonly", "project", ".git")
        cli("writable", "/project")
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))
        self.assertEqual(self.fs.list_mounts()[0]["hidden"], [])

        # Both kinds of explicit overrides survive changing and saving the default.
        cli("writable", "/project", ".git/objects")
        cli("readonly", "/project")
        cli("save", state_path)
        saved = _read_state(state_path)
        self.assertFalse(saved[0]["writable"])
        self.assertEqual(saved[0]["readonly"], [".git"])
        self.assertEqual(saved[0]["readwrite"], [".git/objects"])
        cli("writable", "/project")
        cli("load", state_path, "--force")
        self.root = self.fs._mounts["/project"]["inode"]
        self.assertEqual(self.fs.list_mounts(), saved)
        self.assert_denied(self.fs.open(self.lookup("work.txt"), os.O_RDWR, None))
        self.assert_denied(self.fs.open(self.lookup(".git/config"), os.O_RDWR, None))
        self.open_handle(self.lookup(".git/objects/data"))

    def test_invalid_saved_write_rules_leave_existing_mounts_untouched(self):
        original = self.fs.list_mounts()
        for key in ("readonly", "readwrite"):
            for paths in (".git", None, [1], [".."], ["."]):
                with self.subTest(key=key, paths=paths), self.assertRaises(ValueError):
                    self.fs.load_mounts(
                        [
                            {
                                **original[0],
                                key: paths,
                            }
                        ],
                        replace=True,
                    )
                self.assertEqual(self.fs.list_mounts(), original)
            result = self.fs.load_mounts(
                [
                    {
                        **original[0],
                        key: ["missing"],
                    }
                ],
                replace=True,
            )
            self.assertTrue(result["failed"])
            self.assertEqual(self.fs.list_mounts(), original)
        with self.assertRaisesRegex(
            ValueError, "Conflicting read-only and writable rules"
        ):
            self.fs.load_mounts(
                [
                    {
                        **original[0],
                        "readonly": [".git"],
                        "readwrite": ["./.git/"],
                    }
                ],
                replace=True,
            )
        self.assertEqual(self.fs.list_mounts(), original)

    def test_initial_daemon_rule_is_installed_before_fuse_starts(self):
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "sandbox_fs.py",
                    "daemon",
                    "/unused",
                    "--rw-mount",
                    f"{self.source}:/project",
                    "--readonly",
                    "/project:.git",
                    "--readonly",
                    "/project:work.txt",
                    "--ro-mount",
                    f"{self.source}:/reference",
                    "--writable",
                    "/reference:.git",
                    "--readonly",
                    "/reference:.git/config",
                ],
            ),
            mock.patch.object(pyfuse3, "init", create=True) as init,
            mock.patch.object(pyfuse3, "main", create=True),
            mock.patch.object(pyfuse3, "close", create=True),
            mock.patch("sandbox_fs.trio.run", create=True),
            mock.patch("sandbox_fs.logging.basicConfig"),
            mock.patch.object(ControlServer, "start"),
            mock.patch.object(ControlServer, "stop"),
        ):
            main()
        fs = init.call_args.args[0]
        state = {m["target"]: m for m in fs.list_mounts()}
        self.assertEqual(state["/project"]["readonly"], [".git", "work.txt"])
        self.assertEqual(state["/reference"]["readwrite"], [".git"])
        ino = self.lookup(".git/config", fs=fs, root=fs._mounts["/project"]["inode"])
        self.assert_denied(fs.open(ino, os.O_RDWR, None))
        reference = fs._mounts["/reference"]["inode"]
        ino = self.lookup(".git/config", fs=fs, root=reference)
        self.assert_denied(fs.open(ino, os.O_RDWR, None))
        ino = self.lookup(".git/objects/data", fs=fs, root=reference)
        info = asyncio.run(fs.open(ino, os.O_RDWR, None))
        asyncio.run(fs.release(info.fh))

    def test_cli_rejects_conflicting_unhide_flags_and_initial_rules(self):
        for args in (
            ["unhide", "-s", "/unused", "/project", ".git", "--readonly", "--writable"],
            [
                "daemon",
                "/unused",
                "--ro-mount",
                f"{self.source}:/project",
                "--readonly",
                "/project:.git",
                "--writable",
                "/project:./.git/",
            ],
        ):
            with (
                self.subTest(args=args),
                mock.patch.object(sys, "argv", ["sandbox_fs.py", *args]),
                mock.patch.object(sys, "stderr", io.StringIO()),
                mock.patch("sandbox_fs._send") as send,
                mock.patch("sandbox_fs.logging.basicConfig"),
                mock.patch.object(pyfuse3, "init", create=True) as init,
                self.assertRaises(SystemExit) as raised,
            ):
                main()
            self.assertEqual(raised.exception.code, 2)
            send.assert_not_called()
            init.assert_not_called()


class CacheControlTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.source = os.path.join(self.temp_dir.name, "source")
        os.mkdir(self.source)
        self.file_path = os.path.join(self.source, "file.txt")
        with open(self.file_path, "w", encoding="utf-8") as source_file:
            source_file.write("original")
        self.fs = SandboxFS(entry_timeout=300, attr_timeout=300)
        self.fs.add_mount(self.source, "/parent/mount", writable=True)
        self.mount_ino = self.fs._mounts["/parent/mount"]["inode"]
        self.file_ino = asyncio.run(self.fs.lookup(self.mount_ino, b"file.txt")).st_ino
        self.control = ControlServer(
            self.fs, os.path.join(self.temp_dir.name, "control.sock")
        )

    def test_five_second_defaults_apply_to_virtual_and_real_attributes(self):
        fs = SandboxFS()
        fs.add_mount(self.source, "/mount")
        for ino in (pyfuse3.ROOT_INODE, fs._mounts["/mount"]["inode"]):
            attr = asyncio.run(fs.getattr(ino))
            self.assertEqual(attr.entry_timeout, 5)
            self.assertEqual(attr.attr_timeout, 5)

    def test_timeout_update_invalidates_old_ttls_and_preserves_other_value(self):
        observed_timeouts = []

        def invalidate_inode(ino):
            # A notification may block until a request obtains this lock.
            self.assertFalse(self.fs._lock._is_owned())
            observed_timeouts.append(self.fs.get_timeouts())

        with (
            mock.patch.object(
                pyfuse3, "invalidate_inode", side_effect=invalidate_inode
            ) as invalidate,
            mock.patch.object(pyfuse3, "invalidate_entry") as entries,
        ):
            result = self.fs.set_timeouts(attr_timeout=0.25)

        self.assertTrue(result["ok"])
        expected = {"entry_timeout": 300, "attr_timeout": 0.25}
        self.assertEqual(result["timeouts"], expected)
        self.assertTrue(all(value == expected for value in observed_timeouts))
        self.assertIn(mock.call(self.file_ino), invalidate.call_args_list)
        parent_ino = self.fs._inodes[pyfuse3.ROOT_INODE].children["parent"]
        for entry in (
            (pyfuse3.ROOT_INODE, b"parent"),
            (parent_ino, b"mount"),
            (self.mount_ino, b"file.txt"),
        ):
            self.assertIn(mock.call(*entry), entries.call_args_list)
        attr = asyncio.run(self.fs.getattr(self.file_ino))
        self.assertEqual(attr.attr_timeout, 0.25)
        self.assertEqual(attr.entry_timeout, 300)

    def test_invalid_update_leaves_both_timeouts_unchanged(self):
        for invalid in (-1, float("nan"), float("inf"), True, None, "bad"):
            with (
                self.subTest(value=invalid),
                mock.patch.object(pyfuse3, "invalidate_inode") as invalidate,
            ):
                result = self.control._dispatch(
                    {
                        "action": "timeouts",
                        "entry_timeout": 1,
                        "attr_timeout": invalid,
                    }
                )
                self.assertFalse(result["ok"])
                self.assertEqual(
                    self.fs.get_timeouts(),
                    {
                        "entry_timeout": 300,
                        "attr_timeout": 300,
                    },
                )
                invalidate.assert_not_called()

    def test_query_does_not_invalidate_cache(self):
        with mock.patch.object(pyfuse3, "invalidate_inode") as invalidate:
            result = self.control._dispatch({"action": "timeouts"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["timeouts"], self.fs.get_timeouts())
        invalidate.assert_not_called()

    def test_invalidation_ignores_uncached_inodes_but_reports_other_errors(self):
        def invalidate_inode(ino):
            if ino == pyfuse3.ROOT_INODE:
                raise OSError(errno.ENOENT, "not cached")
            if ino == self.file_ino:
                raise OSError(errno.EIO, "notification failed")

        with (
            mock.patch.object(
                pyfuse3, "invalidate_inode", side_effect=invalidate_inode
            ),
            mock.patch.object(pyfuse3, "invalidate_entry") as entries,
        ):
            result = self.control._dispatch(
                {
                    "action": "timeouts",
                    "attr_timeout": 0,
                }
            )

        self.assertFalse(result["ok"])
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("notification failed", result["failed"][0])
        self.assertIn("Timeouts updated", result["error"])
        self.assertEqual(self.fs.get_timeouts()["attr_timeout"], 0)
        entries.assert_any_call(self.mount_ino, b"file.txt")

    def test_invalidation_reports_directory_notification_failures(self):
        with (
            mock.patch.object(pyfuse3, "invalidate_inode"),
            mock.patch.object(
                pyfuse3,
                "invalidate_entry",
                side_effect=OSError(errno.ENOSYS, "unsupported"),
            ),
        ):
            result = self.fs.invalidate_cache()
        self.assertFalse(result["ok"])
        self.assertEqual(result["entries"], 0)
        self.assertTrue(result["failed"])

    def test_readdir_does_not_republish_old_attributes_or_timeouts(self):
        fh = asyncio.run(self.fs.opendir(self.mount_ino, None))
        with open(self.file_path, "w", encoding="utf-8") as source_file:
            source_file.write("updated and longer")
        with (
            mock.patch.object(pyfuse3, "invalidate_inode"),
            mock.patch.object(pyfuse3, "invalidate_entry"),
        ):
            self.fs.set_timeouts(entry_timeout=1, attr_timeout=0)

        replies = []

        def reply(token, name, attr, next_id):
            replies.append((name, attr, next_id))
            return True

        with mock.patch.object(
            pyfuse3, "readdir_reply", side_effect=reply, create=True
        ):
            asyncio.run(self.fs.readdir(fh, 0, object()))
            os.unlink(self.file_path)
            asyncio.run(self.fs.readdir(fh, 0, object()))

        self.assertEqual(len(replies), 1)
        name, attr, cookie = replies[0]
        self.assertEqual(name, b"file.txt")
        self.assertEqual(attr.st_size, len("updated and longer"))
        self.assertEqual(attr.entry_timeout, 1)
        self.assertEqual(attr.attr_timeout, 0)
        self.assertEqual(cookie, 1)
        asyncio.run(self.fs.releasedir(fh))

    def test_cli_controls_round_trip_through_running_control_socket(self):
        self.control.start()
        self.addCleanup(self.control.stop)
        with (
            mock.patch.object(pyfuse3, "invalidate_inode") as inodes,
            mock.patch.object(pyfuse3, "invalidate_entry") as entries,
        ):
            for command, options, expected in (
                ("timeouts", [], "Entry timeout: 300 seconds"),
                (
                    "timeouts",
                    ["--entry-timeout", "0", "--attr-timeout", "1"],
                    "Attribute timeout: 1 seconds",
                ),
                ("invalidate-cache", [], "Invalidated:"),
                ("timeouts", [], "Entry timeout: 0 seconds"),
            ):
                output = io.StringIO()
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "sandbox_fs.py",
                            command,
                            "-s",
                            self.control._path,
                            *options,
                        ],
                    ),
                    mock.patch.object(sys, "stdout", output),
                ):
                    main()
                self.assertIn(expected, output.getvalue())
            inodes.assert_any_call(self.file_ino)
            entries.assert_any_call(self.mount_ino, b"file.txt")

    def test_cli_rejects_invalid_timeouts_before_contacting_daemon(self):
        for command in ("daemon", "timeouts"):
            for value in ("-1", "nan", "inf"):
                args = ["/unused"] if command == "daemon" else ["-s", "/unused"]
                with (
                    self.subTest(command=command, value=value),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "sandbox_fs.py",
                            command,
                            *args,
                            "--attr-timeout",
                            value,
                        ],
                    ),
                    mock.patch.object(sys, "stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

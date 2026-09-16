import asyncio
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
    pyfuse3.FileInfo = type("FileInfo", (), {})
    pyfuse3.default_options = set()
    pyfuse3.invalidate_entry = lambda inode, name, deleted=0: None
    pyfuse3.invalidate_inode = lambda ino: None
    sys.modules["pyfuse3"] = pyfuse3

try:
    import trio  # noqa: F401
except ModuleNotFoundError:
    sys.modules["trio"] = types.ModuleType("trio")

from sandbox_fs import SandboxFS, _read_state, _write_state


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
            self.config(
                self.sources[1], "/new", hidden=[self.hidden_path]
            ),
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
        desired = [self.config(
            self.sources[1], "/mount", writable=True,
            hidden=[self.hidden_path],
        )]

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
        old_attr = asyncio.run(fs.lookup(
            fs._mounts["/mount"]["inode"], b"shared.txt"
        ))
        desired = [self.config(self.sources[0], "/mount")]

        fs.load_mounts(desired, force=True)
        new_attr = asyncio.run(fs.lookup(
            fs._mounts["/mount"]["inode"], b"shared.txt"
        ))

        self.assertFalse(fs._inodes[old_attr.st_ino].active)
        self.assertNotEqual(old_attr.st_ino, new_attr.st_ino)
        self.assertFalse(fs._inodes[new_attr.st_ino].writable)

        asyncio.run(fs.forget([(old_attr.st_ino, 1)]))
        repeated_attr = asyncio.run(fs.lookup(
            fs._mounts["/mount"]["inode"], b"shared.txt"
        ))
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

        result = fs.load_mounts(
            [self.config(missing, "/replacement")], replace=True
        )

        self.assertEqual(result["failed"][0]["target"], "/replacement")
        self.assertEqual(fs.list_mounts(), original)

    def test_replace_rejects_invalid_target_before_removing_mounts(self):
        fs = SandboxFS()
        fs.add_mount(self.sources[0], "/existing")
        original = fs.list_mounts()

        with self.assertRaisesRegex(ValueError, "Cannot mount at root"):
            fs.load_mounts(
                [self.config(self.sources[1], ".")], replace=True
            )

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
        desired = [self.config(
            self.sources[1], "/keep", hidden=[self.hidden_path]
        )]

        result = fs.load_mounts(desired, replace=True)

        self.assertEqual(result["removed"], ["/remove"])
        self.assertEqual(result["unchanged"], ["/keep"])
        self.assertTrue(fs._is_hidden(
            "/keep", os.path.join(self.sources[1], self.hidden_path)
        ))
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
        ro_attr = asyncio.run(fs.lookup(
            fs._mounts["/read-only"]["inode"], b"shared.txt"
        ))
        rw_attr = asyncio.run(fs.lookup(
            fs._mounts["/read-write"]["inode"], b"shared.txt"
        ))
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
            invalidate_entry.assert_called_with(
                pyfuse3.ROOT_INODE, b"mount"
            )

            fs.add_hide("/mount", self.hidden_path)
            invalidate_entry.assert_called_with(
                mount_ino, self.hidden_path.encode()
            )

            fs.remove_hide("/mount", self.hidden_path)
            invalidate_entry.assert_called_with(
                mount_ino, self.hidden_path.encode()
            )

            fs.remove_mount("/mount")
            invalidate_entry.assert_called_with(
                pyfuse3.ROOT_INODE, b"mount", deleted=mount_ino
            )

    def test_load_rejects_empty_hidden_path(self):
        fs = SandboxFS()

        with self.assertRaisesRegex(ValueError, "Hidden path cannot be empty"):
            fs.load_mounts([
                self.config(self.sources[0], "/mount", hidden=["."])
            ])

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


if __name__ == "__main__":
    unittest.main()

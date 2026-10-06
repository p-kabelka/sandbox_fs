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
    pyfuse3.FileInfo = type("FileInfo", (), {})
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
        self.file_ino = asyncio.run(self.fs.lookup(
            self.mount_ino, b"file.txt"
        )).st_ino
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
            mock.patch.object(pyfuse3, "invalidate_inode",
                              side_effect=invalidate_inode) as invalidate,
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
            with self.subTest(value=invalid), mock.patch.object(
                pyfuse3, "invalidate_inode"
            ) as invalidate:
                result = self.control._dispatch({
                    "action": "timeouts",
                    "entry_timeout": 1,
                    "attr_timeout": invalid,
                })
                self.assertFalse(result["ok"])
                self.assertEqual(self.fs.get_timeouts(), {
                    "entry_timeout": 300, "attr_timeout": 300,
                })
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
            mock.patch.object(pyfuse3, "invalidate_inode",
                              side_effect=invalidate_inode),
            mock.patch.object(pyfuse3, "invalidate_entry") as entries,
        ):
            result = self.control._dispatch({
                "action": "timeouts", "attr_timeout": 0,
            })

        self.assertFalse(result["ok"])
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("notification failed", result["failed"][0])
        self.assertIn("Timeouts updated", result["error"])
        self.assertEqual(self.fs.get_timeouts()["attr_timeout"], 0)
        entries.assert_any_call(self.mount_ino, b"file.txt")

    def test_invalidation_reports_directory_notification_failures(self):
        with (
            mock.patch.object(pyfuse3, "invalidate_inode"),
            mock.patch.object(pyfuse3, "invalidate_entry",
                              side_effect=OSError(errno.ENOSYS, "unsupported")),
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

        with mock.patch.object(pyfuse3, "readdir_reply", side_effect=reply,
                               create=True):
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
                ("timeouts", ["--entry-timeout", "0", "--attr-timeout", "1"],
                 "Attribute timeout: 1 seconds"),
                ("invalidate-cache", [], "Invalidated:"),
                ("timeouts", [], "Entry timeout: 0 seconds"),
            ):
                output = io.StringIO()
                with mock.patch.object(sys, "argv", [
                    "sandbox_fs.py", command, "-s", self.control._path, *options,
                ]), mock.patch.object(sys, "stdout", output):
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
                    mock.patch.object(sys, "argv", [
                        "sandbox_fs.py", command, *args, "--attr-timeout", value,
                    ]),
                    mock.patch.object(sys, "stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

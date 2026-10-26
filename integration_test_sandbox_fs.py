"""FUSE integration tests.

Run with:
    python -m unittest -v integration_test_sandbox_fs

The tests skip when pyfuse3, trio, fusermount3, or an accessible /dev/fuse
device is unavailable.
"""

import errno
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("sandbox_fs.py")


class FuseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != "posix":
            raise unittest.SkipTest("FUSE integration requires POSIX")

        fuse_device = Path("/dev/fuse")
        if not fuse_device.exists() or not os.access(fuse_device, os.R_OK | os.W_OK):
            raise unittest.SkipTest("an accessible /dev/fuse is required")

        cls.fusermount = shutil.which("fusermount3")
        if cls.fusermount is None:
            raise unittest.SkipTest("fusermount3 is required")

        dependency_check = subprocess.run(
            [sys.executable, "-c", "import pyfuse3, trio"],
            capture_output=True,
            text=True,
        )
        if dependency_check.returncode != 0:
            raise unittest.SkipTest("pyfuse3 and trio are required")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)
        self.mountpoint = self.base / "mount"
        self.source_ro = self.base / "source-ro"
        self.source_rw = self.base / "source-rw"
        self.socket_path = self.base / "control.sock"
        self.state_path = self.base / "state.json"
        self.mountpoint.mkdir()
        self.source_ro.mkdir()
        self.source_rw.mkdir()
        (self.source_ro / "reference.txt").write_text(
            "reference data\n", encoding="utf-8"
        )
        (self.source_rw / "editable.txt").write_text(
            "original data\n", encoding="utf-8"
        )
        (self.source_rw / "secret.txt").write_text("hidden data\n", encoding="utf-8")

        self.process = None
        self.addCleanup(self._cleanup_daemon)
        self._start_daemon()

    def _start_daemon(self):
        self.assertIsNone(self.process)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "daemon",
                str(self.mountpoint),
                "--control-socket",
                str(self.socket_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                self.process = None
                details = (stderr or stdout).strip()
                unavailable = (
                    "operation not permitted",
                    "permission denied",
                    "failed to open /dev/fuse",
                    "fuse device not found",
                )
                if any(message in details.lower() for message in unavailable):
                    self.skipTest(f"FUSE mount unavailable: {details}")
                self.fail(f"daemon exited during startup:\n{details}")
            if self.socket_path.exists() and os.path.ismount(self.mountpoint):
                return
            time.sleep(0.05)

        self.fail("daemon did not create its mount and control socket")

    def _stop_daemon(self):
        process = self.process
        self.process = None
        self.assertIsNotNone(process)
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(f"daemon did not stop after SIGINT:\n{stderr or stdout}")

        self.assertEqual(process.returncode, 0, stderr or stdout)
        self.assertFalse(self.socket_path.exists())
        self.assertFalse(os.path.ismount(self.mountpoint))

    def _cleanup_daemon(self):
        if self.process is not None:
            process = self.process
            self.process = None
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=3)

        # A disconnected FUSE mount can make ismount() return False when its
        # root attributes are uncached. Always attempt cleanup of our mount.
        subprocess.run(
            [self.fusermount, "-u", "-z", str(self.mountpoint)],
            capture_output=True,
            text=True,
        )

    def _cli(self, *args, check=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check and result.returncode != 0:
            self.fail(
                f"command failed: {' '.join(map(str, args))}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def _configure_and_save(self):
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            self.source_ro,
            "/readonly",
        )
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            "--read-write",
            self.source_rw,
            "/writable",
        )
        self._cli("hide", "-s", self.socket_path, "/writable", "secret.txt")
        self._cli("save", "-s", self.socket_path, self.state_path)

    def _assert_read_only(self, path):
        with self.assertRaises(OSError) as raised:
            path.write_text("blocked", encoding="utf-8")
        self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))

    def test_save_stop_restart_and_load(self):
        self._configure_and_save()
        root = self.mountpoint

        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self._assert_read_only(root / "readonly/blocked.txt")
        self.assertFalse((root / "writable/secret.txt").exists())

        created = root / "writable/created.txt"
        created.write_text("created through FUSE\n", encoding="utf-8")
        self.assertEqual(
            (self.source_rw / "created.txt").read_text(encoding="utf-8"),
            "created through FUSE\n",
        )

        self._stop_daemon()
        self._start_daemon()
        self.assertIn(
            "No mounts",
            self._cli("list", "-s", self.socket_path).stdout,
        )

        loaded = self._cli("load", "-s", self.socket_path, self.state_path)
        self.assertIn("Loaded: /readonly", loaded.stdout)
        self.assertIn("Loaded: /writable", loaded.stdout)
        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self.assertFalse((root / "writable/secret.txt").exists())

        repeated = self._cli("load", "-s", self.socket_path, self.state_path)
        self.assertIn("Unchanged: /readonly", repeated.stdout)
        self.assertIn("Unchanged: /writable", repeated.stdout)

    def test_conflict_force_and_replace_refresh_kernel_entries(self):
        self._configure_and_save()
        root = self.mountpoint

        # Populate the kernel cache before replacing the mount at this target.
        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self._cli("unmount", "-s", self.socket_path, "/readonly")
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            "-w",
            self.source_rw,
            "/readonly",
        )
        conflicting_file = root / "readonly/editable.txt"
        self.assertEqual(
            conflicting_file.read_text(encoding="utf-8"),
            "original data\n",
        )
        conflicting_file.write_text("conflicting data\n", encoding="utf-8")

        conflict = self._cli(
            "load", "-s", self.socket_path, self.state_path, check=False
        )
        self.assertEqual(conflict.returncode, 1)
        self.assertIn("Conflict: /readonly", conflict.stderr)
        self.assertEqual(
            conflicting_file.read_text(encoding="utf-8"),
            "conflicting data\n",
        )

        forced = self._cli("load", "-s", self.socket_path, "--force", self.state_path)
        self.assertIn("Loaded: /readonly", forced.stdout)
        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self.assertFalse((root / "readonly/editable.txt").exists())
        self._assert_read_only(root / "readonly/blocked.txt")

        self._cli("mount", "-s", self.socket_path, self.source_rw, "/extra")
        self.assertEqual(
            (root / "extra/editable.txt").read_text(encoding="utf-8"),
            "conflicting data\n",
        )
        replaced = self._cli(
            "load", "-s", self.socket_path, "--replace", self.state_path
        )
        self.assertIn("Removed: /extra", replaced.stdout)
        self.assertFalse((root / "extra").exists())

    def test_runtime_hide_and_unhide_refresh_kernel_entries(self):
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            "-w",
            self.source_rw,
            "/writable",
        )
        secret = self.mountpoint / "writable/secret.txt"

        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")
        self._cli("hide", "-s", self.socket_path, "/writable", "secret.txt")
        self.assertFalse(secret.exists())
        self.assertNotIn(
            "secret.txt",
            {entry.name for entry in secret.parent.iterdir()},
        )

        self._cli("unhide", "-s", self.socket_path, "/writable", "secret.txt")
        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")

    def test_mount_defaults_refresh_cached_permissions_and_preserve_subpath_state(self):
        self._cli("mount", "-s", self.socket_path, self.source_rw, "/project")
        self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--entry-timeout",
            "300",
            "--attr-timeout",
            "300",
        )
        project = self.mountpoint / "project"
        editable = project / "editable.txt"
        secret = project / "secret.txt"
        self._cli("hide", "-s", self.socket_path, "/project", "secret.txt")
        self.assertFalse(project.stat().st_mode & 0o222)
        self.assertFalse(editable.stat().st_mode & 0o222)
        self.assertFalse(secret.exists())

        self._cli("writable", "-s", self.socket_path, "/project")
        self.assertTrue(project.stat().st_mode & 0o222)
        self.assertTrue(editable.stat().st_mode & 0o222)
        self.assertFalse(secret.exists())
        self.assertNotIn("secret.txt", {entry.name for entry in project.iterdir()})
        self._cli(
            "unhide", "-s", self.socket_path, "-R", "/project", "secret.txt"
        )

        with open(editable, "r+b", buffering=0) as writer:
            for action in ("writable", "readonly", "writable"):
                with self.subTest(action=action):
                    self._cli(action, "-s", self.socket_path, "/project")
                    writable = action == "writable"
                    self.assertEqual(bool(project.stat().st_mode & 0o222), writable)
                    self.assertEqual(bool(editable.stat().st_mode & 0o222), writable)
                    self.assertFalse(secret.stat().st_mode & 0o222)
                    self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")
                    self._assert_read_only(secret)
                    if writable:
                        writer.seek(0)
                        self.assertEqual(writer.write(b"allowed!"), 8)
                        created = project / "created.txt"
                        created.write_text("created", encoding="utf-8")
                        created.unlink()
                    else:
                        with self.assertRaises(OSError) as raised:
                            writer.write(b"blocked")
                        self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))
                        self._assert_read_only(project / "blocked.txt")
        self.assertEqual(
            (self.source_rw / "secret.txt").read_text(encoding="utf-8"), "hidden data\n"
        )

    def test_runtime_timeouts_replace_long_lived_cached_attributes(self):
        defaults = self._cli("timeouts", "-s", self.socket_path)
        self.assertIn("Entry timeout: 5 seconds", defaults.stdout)
        self.assertIn("Attribute timeout: 5 seconds", defaults.stdout)
        self._cli("mount", "-s", self.socket_path, self.source_rw, "/shared")
        shared = self.mountpoint / "shared/editable.txt"
        self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--entry-timeout",
            "300",
            "--attr-timeout",
            "300",
        )
        self.assertEqual(shared.stat().st_size, len("original data\n"))
        updated = "longer externally updated contents\n"
        (self.source_rw / "editable.txt").write_text(updated, encoding="utf-8")
        # Establish that this test really populated the old long-lived TTL.
        self.assertEqual(shared.stat().st_size, len("original data\n"))

        changed = self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--attr-timeout",
            "0",
        )
        self.assertIn("Entry timeout: 300 seconds", changed.stdout)
        self.assertEqual(shared.stat().st_size, len(updated))
        self.assertEqual(shared.read_text(encoding="utf-8"), updated)
        updated_again = "short\n"
        (self.source_rw / "editable.txt").write_text(updated_again, encoding="utf-8")
        self.assertEqual(shared.stat().st_size, len(updated_again))

    def test_readonly_subtree_blocks_cached_open_handles_and_path_mutations(self):
        git_source = self.source_rw / ".git"
        git_source.mkdir()
        (git_source / "config").write_text("original", encoding="utf-8")
        (git_source / "objects").mkdir()
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            "-w",
            self.source_rw,
            "/project",
        )
        self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--entry-timeout",
            "300",
            "--attr-timeout",
            "300",
        )
        project = self.mountpoint / "project"
        git = project / ".git"
        config = git / "config"
        self.assertTrue(config.stat().st_mode & 0o222)
        with open(config, "r+b", buffering=0) as writer:
            self._cli("readonly", "-s", self.socket_path, "/project", ".git")
            self.assertFalse(config.stat().st_mode & 0o222)
            self.assertFalse(git.stat().st_mode & 0o222)
            self.assertEqual(writer.read(), b"original")
            with self.assertRaises(OSError) as raised:
                writer.write(b"blocked")
            self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))

            for mutation in (
                lambda: config.write_text("blocked", encoding="utf-8"),
                lambda: config.unlink(),
                lambda: config.chmod(0o777),
                lambda: (git / "new").mkdir(),
                lambda: (git / "objects").rmdir(),
                lambda: git.rename(project / "moved"),
                lambda: (project / "editable.txt").rename(config),
                lambda: os.link(config, project / "alias"),
            ):
                with self.assertRaises(OSError) as raised:
                    mutation()
                self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))
            self._assert_read_only(git / "new.txt")
            (project / "editable.txt").write_text("editable", encoding="utf-8")
            self.assertEqual(
                (git_source / "config").read_text(encoding="utf-8"), "original"
            )

            self._cli("writable", "-s", self.socket_path, "/project", ".git")
            self.assertTrue(config.stat().st_mode & 0o222)
            writer.seek(0)
            self.assertEqual(writer.write(b"allowed!"), 8)
        self.assertEqual(
            (git_source / "config").read_text(encoding="utf-8"), "allowed!"
        )

    def test_unhide_readonly_and_save_restart_load(self):
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            "-w",
            self.source_rw,
            "/project",
        )
        secret = self.mountpoint / "project/secret.txt"
        self._cli("hide", "-s", self.socket_path, "/project", "secret.txt")
        # Establish a negative lookup before unhiding a never-allocated inode.
        self.assertFalse(secret.exists())
        self._cli(
            "unhide",
            "-s",
            self.socket_path,
            "/project",
            "secret.txt",
            "--ro-path",
        )
        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")
        self._assert_read_only(secret)
        self._cli("save", "-s", self.socket_path, self.state_path)
        self._stop_daemon()
        self._start_daemon()
        self._cli("load", "-s", self.socket_path, self.state_path)
        self.assertIn(
            "readonly: secret.txt", self._cli("list", "-s", self.socket_path).stdout
        )
        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")
        self._assert_read_only(secret)
        with self.assertRaises(OSError) as raised:
            secret.unlink()
        self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))

    def test_writable_override_on_readonly_mount_and_nested_rules(self):
        editable_source = self.source_ro / "editable"
        editable_source.mkdir()
        (editable_source / "file.txt").write_text("original", encoding="utf-8")
        self._cli("mount", "-s", self.socket_path, self.source_ro, "/reference")
        self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--entry-timeout",
            "300",
            "--attr-timeout",
            "300",
        )
        reference = self.mountpoint / "reference"
        editable = reference / "editable"
        file_path = editable / "file.txt"
        self.assertFalse(file_path.stat().st_mode & 0o222)
        self._assert_read_only(file_path)
        self._cli("writable", "-s", self.socket_path, "/reference", "editable")
        self.assertTrue(file_path.stat().st_mode & 0o222)
        self.assertTrue(editable.stat().st_mode & 0o222)
        self.assertFalse(reference.stat().st_mode & 0o222)
        self._assert_read_only(reference / "reference.txt")
        self._assert_read_only(reference / "outside.txt")

        (editable / "created.txt").write_text("created", encoding="utf-8")
        directory = editable / "directory"
        directory.mkdir()
        directory.rmdir()
        file_path.chmod(0o600)
        (editable / "symbolic").symlink_to("file.txt")
        (editable / "symbolic").unlink()
        os.link(file_path, editable / "hardlink")
        (editable / "hardlink").unlink()
        (editable / "created.txt").rename(editable / "renamed.txt")
        (editable / "renamed.txt").unlink()

        with open(file_path, "r+b", buffering=0) as writer:
            self.assertEqual(writer.write(b"allowed!"), 8)
            self._cli(
                "readonly", "-s", self.socket_path, "/reference", "editable/file.txt"
            )
            self.assertFalse(file_path.stat().st_mode & 0o222)
            with self.assertRaises(OSError) as raised:
                writer.write(b"blocked")
            self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))
            self._cli(
                "writable", "-s", self.socket_path, "/reference", "editable/file.txt"
            )
            self._cli("readonly", "-s", self.socket_path, "/reference", "editable")
            # The child remains writable even after its parent's rule changes.
            self.assertTrue(file_path.stat().st_mode & 0o222)
            writer.seek(0)
            self.assertEqual(writer.write(b"child-rw"), 8)
            self._assert_read_only(editable / "new.txt")
            with self.assertRaises(OSError) as raised:
                file_path.unlink()
            self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))
            self._cli(
                "readonly", "-s", self.socket_path, "/reference", "editable/file.txt"
            )
            self._cli("writable", "-s", self.socket_path, "/reference", "editable")
            self._assert_read_only(file_path)
            with self.assertRaises(OSError) as raised:
                writer.write(b"blocked")
            self.assertIn(raised.exception.errno, (errno.EACCES, errno.EROFS))
        self.assertEqual(
            (editable_source / "file.txt").read_text(encoding="utf-8"), "child-rw"
        )

    def test_mixed_write_overrides_survive_save_restart_load(self):
        cache_source = self.source_ro / "cache"
        cache_source.mkdir()
        (cache_source / "index").write_text("index", encoding="utf-8")
        (cache_source / "data").write_text("data", encoding="utf-8")
        self._cli(
            "mount",
            "-s",
            self.socket_path,
            self.source_ro,
            "/reference",
            "-W",
            "cache",
            "--ro-path",
            "cache/index",
        )
        reference = self.mountpoint / "reference"
        cache = reference / "cache"
        (cache / "data").write_text("initial update", encoding="utf-8")
        self._assert_read_only(cache / "index")
        self._assert_read_only(reference / "reference.txt")
        self._cli("save", "-s", self.socket_path, self.state_path)
        self._stop_daemon()
        self._start_daemon()
        self._cli("load", "-s", self.socket_path, self.state_path)

        listing = self._cli("list", "-s", self.socket_path).stdout
        self.assertIn("(ro)", listing)
        self.assertIn("readwrite: cache", listing)
        self.assertIn("readonly: cache/index", listing)
        (cache / "data").write_text("updated", encoding="utf-8")
        (cache / "new").write_text("created", encoding="utf-8")
        self.assertEqual((cache_source / "data").read_text(encoding="utf-8"), "updated")
        self.assertEqual((cache_source / "new").read_text(encoding="utf-8"), "created")
        self._assert_read_only(cache / "index")
        self._assert_read_only(reference / "reference.txt")
        self.assertIn(
            "Unchanged: /reference",
            self._cli("load", "-s", self.socket_path, self.state_path).stdout,
        )

    def test_unhide_writable_file_inside_readonly_mount(self):
        self._cli("mount", "-s", self.socket_path, self.source_ro, "/reference")
        file_path = self.mountpoint / "reference/reference.txt"
        self._cli("hide", "-s", self.socket_path, "/reference", "reference.txt")
        self.assertFalse(file_path.exists())
        self._cli(
            "unhide",
            "-s",
            self.socket_path,
            "/reference",
            "reference.txt",
            "-W",
        )
        file_path.write_text("writable", encoding="utf-8")
        self.assertEqual(
            (self.source_ro / "reference.txt").read_text(encoding="utf-8"), "writable"
        )
        self.assertFalse(file_path.parent.stat().st_mode & 0o222)
        self._cli("hide", "-s", self.socket_path, "/reference", "reference.txt")
        self._cli("unhide", "-s", self.socket_path, "/reference", "reference.txt")
        file_path.write_text("still writable", encoding="utf-8")
        self._cli("readonly", "-s", self.socket_path, "/reference", "reference.txt")
        self._assert_read_only(file_path)

    def test_invalidate_cache_refreshes_open_file_data_and_directory_entries(self):
        self._cli("mount", "-s", self.socket_path, self.source_rw, "/shared")
        self._cli(
            "timeouts",
            "-s",
            self.socket_path,
            "--entry-timeout",
            "300",
            "--attr-timeout",
            "300",
        )
        shared = self.mountpoint / "shared/editable.txt"
        shared.stat()
        original = "original data\n"
        updated = "modified data\n"
        self.assertEqual(len(original), len(updated))
        with open(shared, "rb", buffering=0) as reader:
            self.assertEqual(reader.read(), original.encode())
            (self.source_rw / "editable.txt").write_text(updated, encoding="utf-8")
            reader.seek(0)
            self.assertEqual(reader.read(), original.encode())
            self._cli("invalidate-cache", "-s", self.socket_path)
            reader.seek(0)
            self.assertEqual(reader.read(), updated.encode())

        deleted = self.mountpoint / "shared/secret.txt"
        self.assertTrue(deleted.exists())
        (self.source_rw / "secret.txt").unlink()
        self.assertTrue(deleted.exists())
        refreshed = self._cli("invalidate-cache", "-s", self.socket_path)
        self.assertIn("Invalidated:", refreshed.stdout)
        self.assertFalse(deleted.exists())
        new_file = self.source_rw / "new.txt"
        new_file.write_text("new file\n", encoding="utf-8")
        self._cli("invalidate-cache", "-s", self.socket_path)
        self.assertIn("new.txt", {entry.name for entry in shared.parent.iterdir()})


if __name__ == "__main__":
    unittest.main()

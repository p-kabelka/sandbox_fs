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
        if not fuse_device.exists() or not os.access(
            fuse_device, os.R_OK | os.W_OK
        ):
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
        (self.source_rw / "secret.txt").write_text(
            "hidden data\n", encoding="utf-8"
        )

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
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=3)

        if os.path.ismount(self.mountpoint):
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
            "mount", "-s", self.socket_path,
            self.source_ro, "/readonly",
        )
        self._cli(
            "mount", "-s", self.socket_path, "-w",
            self.source_rw, "/writable",
        )
        self._cli(
            "hide", "-s", self.socket_path, "/writable", "secret.txt"
        )
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

        loaded = self._cli(
            "load", "-s", self.socket_path, self.state_path
        )
        self.assertIn("Loaded: /readonly", loaded.stdout)
        self.assertIn("Loaded: /writable", loaded.stdout)
        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self.assertFalse((root / "writable/secret.txt").exists())

        repeated = self._cli(
            "load", "-s", self.socket_path, self.state_path
        )
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
            "mount", "-s", self.socket_path, "-w",
            self.source_rw, "/readonly",
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

        forced = self._cli(
            "load", "-s", self.socket_path, "--force", self.state_path
        )
        self.assertIn("Loaded: /readonly", forced.stdout)
        self.assertEqual(
            (root / "readonly/reference.txt").read_text(encoding="utf-8"),
            "reference data\n",
        )
        self.assertFalse((root / "readonly/editable.txt").exists())
        self._assert_read_only(root / "readonly/blocked.txt")

        self._cli(
            "mount", "-s", self.socket_path, self.source_rw, "/extra"
        )
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
            "mount", "-s", self.socket_path, "-w",
            self.source_rw, "/writable",
        )
        secret = self.mountpoint / "writable/secret.txt"

        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")
        self._cli(
            "hide", "-s", self.socket_path, "/writable", "secret.txt"
        )
        self.assertFalse(secret.exists())
        self.assertNotIn(
            "secret.txt",
            {entry.name for entry in secret.parent.iterdir()},
        )

        self._cli(
            "unhide", "-s", self.socket_path, "/writable", "secret.txt"
        )
        self.assertEqual(secret.read_text(encoding="utf-8"), "hidden data\n")


if __name__ == "__main__":
    unittest.main()

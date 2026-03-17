import os
import subprocess
import shutil
import signal
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
import psutil
import sys

sys.path.append("experiment-runner")
from Plugins.Profilers.JoularCore import JoularCore


# -----------------------------
# Helpers
# -----------------------------
# Check if joularcore is on PATH
def _which_joularcore() -> bool:
    return shutil.which("joularcore") is not None

def _assert_csv_has_rows(testcase: unittest.TestCase, csv_path: Path):
    testcase.assertTrue(os.path.exists(csv_path), f"Expected CSV to exist: {csv_path}")
    testcase.assertGreater(os.path.getsize(csv_path), 0, f"Expected CSV to be non-empty: {csv_path}")
    
    df = pd.read_csv(csv_path, delimiter=",")
    testcase.assertGreaterEqual(len(df.columns), 1, "Expected at least 1 column in CSV")
    testcase.assertGreater(len(df), 0, "Expected at least 1 data row in CSV")


# -----------------------------
# LOGIC TESTS (mocked)
# Run with:
#   python3 -m unittest test_JoularCore.TestJoularCoreLogic
# -----------------------------
from unittest.mock import patch


class TestJoularCoreLogic(unittest.TestCase):
    def tearDown(self):
        if getattr(self, "plugin", None) is None:
            return
        try:
            if getattr(self.plugin, "logfile", None) and os.path.exists(self.plugin.logfile):
                os.remove(self.plugin.logfile)
        except Exception:
            pass
        self.plugin = None

    def test_update(self):
        self.plugin = JoularCore()

        original_args = self.plugin.args.copy()

        self.plugin.update_parameters(add={"-c": "cpu"})
        self.assertIn(("-c", "cpu"), self.plugin.args.items())

        self.plugin.update_parameters(add={"-a": "gnome-shell"})
        self.assertIn(("-a", "gnome-shell"), self.plugin.args.items())

        self.plugin.update_parameters(add={"-p": 4462})
        self.assertIn(("-p", 4462), self.plugin.args.items())

    def test_invalid_update(self):
        self.plugin = JoularCore()

        with self.assertRaises(RuntimeError):
            self.plugin.update_parameters(add={"--not-a-valid-parameter": None})

        original_args = self.plugin.args.copy()

        # Null-op remove of non-existent key
        self.plugin.update_parameters(remove=["--not-a-valid-parameter"])
        self.assertDictEqual(original_args, self.plugin.args)

        with self.assertRaises(RuntimeError):
            self.plugin.update_parameters(add={"-p": "not an int"})

    @patch("Plugins.Profilers.DataSource.CLISource.start", autospec=True)
    @patch("Plugins.Profilers.DataSource.CLISource.stop", autospec=True)
    @patch("subprocess.Popen")
    def test_mode_app(self, popen_mock, stop_mock, start_mock):
        self.plugin = JoularCore(app="gnome-shell")

        self.plugin.start()
        start_mock.assert_called_once()
        popen_mock.assert_not_called()

        self.assertTrue(("-a", "gnome-shell") in self.plugin.args.items() or ("--app", "gnome-shell") in self.plugin.args.items())

        self.plugin.stop()
        stop_mock.assert_called_once()

    @patch("Plugins.Profilers.DataSource.CLISource.start", autospec=True)
    @patch("Plugins.Profilers.DataSource.CLISource.stop", autospec=True)
    @patch("subprocess.Popen")
    def test_mode_pid(self, popen_mock, stop_mock, start_mock):
        self.plugin = JoularCore(pid=4462)

        self.plugin.start()
        start_mock.assert_called_once()
        popen_mock.assert_not_called()

        self.assertTrue(("-p", 4462) in self.plugin.args.items() or ("--pid", 4462) in self.plugin.args.items())

        self.plugin.stop()
        stop_mock.assert_called_once()

    @patch("Plugins.Profilers.DataSource.CLISource.start", autospec=True)
    @patch("Plugins.Profilers.DataSource.CLISource.stop", autospec=True)
    @patch("subprocess.Popen")
    def test_mode_whole_system_default(self, popen_mock, stop_mock, start_mock):
        self.plugin = JoularCore()

        self.plugin.start()
        start_mock.assert_called_once()
        popen_mock.assert_not_called()

        self.assertFalse(("-a" in self.plugin.args) or ("--app" in self.plugin.args))
        self.assertFalse(("-p" in self.plugin.args) or ("--pid" in self.plugin.args))

        self.plugin.stop()
        stop_mock.assert_called_once()

    def test_both_app_and_pid_is_deferred_to_joularcore(self):
        # This should not raise at init time; JoularCore CLI will error when executed.
        try:
            self.plugin = JoularCore(additional_args={"-a": "gnome-shell", "-p": 4462})
        except ValueError as e:
            self.fail(f"Plugin raised ValueError but should defer -a/-p mutual exclusion to JoularCore CLI: {e}")


# -----------------------------
# INTEGRATION TESTS (real joularcore)
# Run with:
#   python3 -m unittest test_joularcore.TestJoularCoreIntegration
#
# Notes:
# - These tests assume `joularcore` is in PATH.
# - They may require appropriate permissions to read energy counters on your machine.
# -----------------------------
@unittest.skipUnless(_which_joularcore(), "joularcore is not on PATH.")
class TestJoularCoreIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="joularcore_tests_"))
        self.plugin = None

    def tearDown(self):
        # Best-effort stop plugin and cleanup files
        try:
            if self.plugin is not None:
                try:
                    self.plugin.stop(wait=False)
                except Exception:
                    pass
        finally:
            self.plugin = None

        for p in self.tmpdir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            self.tmpdir.rmdir()
        except Exception:
            pass

    def test_spawn_single_process_and_kill(self):
        """
        Validates (refactored, manual spawn):
        - Test spawns a target process manually
        - Plugin attaches to the PID (-p)
        - Plugin starts/stops joularcore
        - Test terminates the target process (plugin no longer owns lifecycle)
        - CSV output exists and is parseable with at least one row
        """
        outfile = self.tmpdir / "jc_single.csv"

        # Spawn target manually (long-running so we can terminate it early)
        # Use a cross-platform CPU burner if you prefer:
        #   target_cmd = 'python -c "while True: pass"'
        target_cmd = "sleep 30" if os.name != "nt" else 'python3 -c "import time; time.sleep(30)"'

        target_proc = subprocess.Popen(
            target_cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        target_pid = int(target_proc.pid)
        self.assertTrue(psutil.pid_exists(target_pid), f"Expected spawned target PID to exist: {target_pid}")

        # Attach plugin to the target PID (no internal spawning)
        self.plugin = JoularCore(out_file=outfile, pid=target_pid)

        try:
            self.plugin.start()

            # Ensure joularcore started and let it write some samples (best-effort, allow a short delay)
            time.sleep(2.5)
            print(f"JoularCore args: {self.plugin._format_cmd()}")
            # Stop joularcore monitoring
            self.plugin.stop(wait=False)

        finally:
            # Terminate target manually (plugin no longer owns lifecycle)
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(target_pid), signal.SIGTERM)
                else:
                    target_proc.terminate()
            except Exception:
                pass

            # Reap / escalate if needed to avoid zombies and ResourceWarnings
            try:
                target_proc.wait(timeout=2.0)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(target_pid), signal.SIGKILL)
                    else:
                        target_proc.kill()
                except Exception:
                    pass
                try:
                    target_proc.wait(timeout=2.0)
                except Exception:
                    pass

        # Validate CSV output
        _assert_csv_has_rows(self, outfile)

        # Validate plugin parse_log path works
        parsed = self.plugin.parse_log(outfile)
        self.assertIsInstance(parsed, dict)
        self.assertGreaterEqual(len(parsed.keys()), 1)


if __name__ == "__main__":
    unittest.main()

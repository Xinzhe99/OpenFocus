"""Cross-process isolation of the session settings fixture."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.process_isolation import sandbox_env


_PROBE = '''
import os
from pathlib import Path
import time
import pytest


def wait_for(path):
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for peer: " + str(path))
        time.sleep(.02)


def test_session(qapp):
    from utils.settings_store import get_settings
    root = Path(os.environ["PROBE_ROOT"])
    role = os.environ["PROBE_ROLE"]
    settings = get_settings()
    settings.setValue("isolation/sentinel", role)
    settings.sync()
    (root / (role + ".path")).write_text(settings.fileName(), encoding="utf-8")
    if role == "a":
        wait_for(root / "b.path")
        assert settings.value("isolation/sentinel") == "a"
        outcome = os.environ["PROBE_OUTCOME"]
        if outcome == "failure":
            pytest.fail("intentional failure after peer started")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
    else:
        wait_for(root / "a.done")
        settings.sync()
        assert settings.value("isolation/sentinel") == "b"
        assert Path(settings.fileName()).exists()
'''


def _wait_for(path, processes):
    deadline = time.monotonic() + 90
    while not path.exists():
        for process in processes:
            if process.poll() is not None:
                pytest.fail(f"settings probe exited early: {process.returncode}")
        if time.monotonic() >= deadline:
            pytest.fail(f"settings probe did not create {path.name}")
        time.sleep(.02)


@pytest.mark.parametrize("outcome", ["normal", "failure", "interrupt"])
def test_parallel_settings_sessions_survive_peer_exit(tmp_path, outcome):
    pytest.importorskip("PyQt6")
    probe = tmp_path / "probe_test.py"
    probe.write_text(_PROBE, encoding="utf-8")
    project_root = Path(__file__).resolve().parent.parent
    processes = []
    logs = []
    try:
        for role in ("a", "b"):
            env = sandbox_env(tmp_path / role)
            env.update(PROBE_ROOT=str(tmp_path), PROBE_ROLE=role, PROBE_OUTCOME=outcome,
                       QT_QPA_PLATFORM="offscreen")
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(project_root), env.get("PYTHONPATH"))))
            log = (tmp_path / f"{role}.log").open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q",
                 str(probe), "--basetemp", str(tmp_path / (role + "-pytest"))],
                cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            processes.append(process)
            _wait_for(tmp_path / (role + ".path"), [process])
        assert (tmp_path / "a.path").read_text() != (tmp_path / "b.path").read_text()
        code = processes[0].wait(timeout=90)
        assert code == (0 if outcome == "normal" else 1 if outcome == "failure" else 2)
        (tmp_path / "a.done").touch()
        assert processes[1].wait(timeout=90) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()

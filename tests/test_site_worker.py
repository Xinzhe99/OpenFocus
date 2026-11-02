"""Run the Cloudflare Worker download/routing tests with Node's test runner."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_site_worker_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the site worker tests")

    test_file = Path(__file__).with_name("site_worker.test.mjs")
    completed = subprocess.run(
        [node, "--test", str(test_file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

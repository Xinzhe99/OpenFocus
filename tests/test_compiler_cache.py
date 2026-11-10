"""Compiled kernels remain correct when their optional cache is unavailable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_fusion_survives_contended_and_corrupt_compiler_cache(tmp_path):
    environment = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
    code = textwrap.dedent('''
        import hashlib
        import json
        import numpy as np
        from fusion_methods.openfocus_bc import openfocus_b_impl

        rng = np.random.default_rng(177)
        frames = [rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
                  for _ in range(3)]
        result = openfocus_b_impl(frames, thread_count=1)
        digest = hashlib.sha256(result.tobytes()).hexdigest()
        print(json.dumps({'digest': digest}))
    ''')

    def render():
        result = subprocess.run(
            [sys.executable, '-c', code], env=environment,
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])['digest']

    expected = render()
    indexes = list((tmp_path / 'numba').rglob('_depth_kernel.axis_block*.nbi'))
    assert len(indexes) == 1
    from fusion_methods._depth_kernel import _cache_access
    with _cache_access(indexes[0].parent) as locked:
        assert locked
        assert render() == expected
    indexes[0].write_bytes(b'broken cache index')
    assert render() == expected

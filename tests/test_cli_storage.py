"""CLI temporary-storage warning and --temp-dir; the run is never blocked."""
import json
import os
import shutil

import numpy as np
import pytest

from core import frame_storage

GIB = 1024 ** 3


def _fake_space(free_by_directory, total=100 * GIB):
    """Replace _space with fixed free space per directory (never touches a disk)."""
    def space(directory):
        key = os.path.abspath(directory)
        if key not in free_by_directory:
            raise OSError(f'not mocked: {key}')
        free = free_by_directory[key]
        if isinstance(free, Exception):
            raise free
        usage = shutil._ntuple_diskusage(total, total - free, free)
        reserve = max(GIB, min(total // 20, 4 * GIB))
        return usage, reserve, max(0, free - reserve)
    return space


def _assessment(status, directory='/scratch'):
    return {'status': status, 'directory': directory, 'explicit': True, 'free_bytes': 9 * GIB,
            'reserve_bytes': 4 * GIB, 'available_bytes': 5 * GIB, 'minimum_bytes': GIB,
            'expected_bytes': 2 * GIB, 'optional_bytes': 4 * GIB, 'required_bytes': 6 * GIB}


def _assess_as(monkeypatch, statuses):
    """Make assess_scratch return the given statuses in order (per directory)."""
    calls = []

    def assess(*args, directory=None):
        calls.append(directory)
        status = statuses[directory] if isinstance(statuses, dict) else statuses
        return _assessment(status, directory or '/scratch')
    monkeypatch.setattr(frame_storage, 'assess_scratch', assess)
    return calls


def _cli(monkeypatch, tmp_path, *argv):
    monkeypatch.setenv('OPENFOCUS_LOG_DIR', str(tmp_path / 'logs'))
    from core.cli import run_cli
    return run_cli(list(argv))


def _cli_stack(tmp_path):
    from utils.image_utils import imwrite_auto
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index in range(2):
        image = np.full((24, 32, 3), 40 + index * 100, np.uint8)
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    return stack


@pytest.mark.parametrize('status', ['advisory', 'critical'])
def test_cli_warns_on_stderr_and_never_blocks(monkeypatch, tmp_path, capsys, status):
    stack = _cli_stack(tmp_path)
    _assess_as(monkeypatch, status)
    code = _cli(monkeypatch, tmp_path, '--input', str(stack), '--output', str(tmp_path / 'o.png'),
                '--method', 'dct', '--json', str(tmp_path / 's.json'))
    captured = capsys.readouterr()
    assert code == 0
    assert 'warning: low temporary storage in /scratch' in captured.err
    assert '--temp-dir' in captured.err
    assert json.loads((tmp_path / 's.json').read_text())['output'] == str(tmp_path / 'o.png')


def test_cli_keeps_the_existing_scratch_error_and_exit_code(monkeypatch, tmp_path, capsys):
    stack = _cli_stack(tmp_path)
    scratch = tmp_path / 'tiny'
    scratch.mkdir()
    monkeypatch.setattr(frame_storage, '_space', _fake_space({str(scratch): 4 * GIB}))
    code = _cli(monkeypatch, tmp_path, '--input', str(stack), '--output', str(tmp_path / 'o.png'),
                '--method', 'dct', '--temp-dir', str(scratch))
    captured = capsys.readouterr()
    assert code == 1
    assert 'Temporary storage' in captured.err or str(scratch) in captured.err
    frame_storage.configure_storage_directory(None)


def test_cli_temp_dir_must_be_writable(monkeypatch, tmp_path, capsys):
    stack = _cli_stack(tmp_path)
    code = _cli(monkeypatch, tmp_path, '--input', str(stack), '--output', str(tmp_path / 'o.png'),
                '--method', 'dct', '--temp-dir', str(tmp_path / 'missing'))
    assert code == 2
    assert '--temp-dir is not a writable directory' in capsys.readouterr().err
    assert frame_storage._storage_directory is None


def test_cli_assesses_every_page_of_a_multipage_tiff(monkeypatch, tmp_path, capsys):
    """A multipage TIFF is rendered as a whole stack, so the warning counts every page."""
    import tifffile
    pages = np.stack([np.full((24, 32, 3), 40 + index * 60, np.uint8) for index in range(3)])
    source = tmp_path / 'pages.tif'
    tifffile.imwrite(source, pages, photometric='rgb')
    estimates = []

    def assess(minimum, expected, optional=0, source_paths=(), *, directory=None):
        estimates.append(optional)
        return _assessment('ok')
    monkeypatch.setattr(frame_storage, 'assess_scratch', assess)
    code = _cli(monkeypatch, tmp_path, '--input', str(source), '--output', str(tmp_path / 'o.png'),
                '--method', 'dct')
    assert code == 0, capsys.readouterr().err
    # The cache working set of all three pages, plus its one-frame staging floor.
    assert estimates == [pages.nbytes + 64 * 1024 ** 2]

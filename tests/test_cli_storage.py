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


def test_cli_scratch_is_scoped_across_success_and_failure(monkeypatch, tmp_path):
    from core import cli

    stack = _cli_stack(tmp_path)
    first, second, saved, automatic = (tmp_path / name for name in ('first', 'second', 'saved', 'automatic'))
    for directory in (first, second, saved, automatic):
        directory.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(saved))
    paths, states = [], []
    original = cli._fuse_stack

    def fuse(images, args):
        # Lazy loading has not selected a volume yet; model the first stage
        # that admits a disk allocation before creating its mapping.
        frame_storage.select_scratch(64)
        mapped = frame_storage.store_frame(np.zeros((8, 8), np.uint8), prefer_ram=False)
        paths.append(mapped.filename)
        states.append(frame_storage.capture_scratch())
        # Logging retains exception tracebacks, including live local mappings.
        # Release this view before raising so cleanup can be observed.
        del mapped
        if len(paths) == 2:
            raise RuntimeError('deliberate failure')
        return original(images, args)

    monkeypatch.setattr(cli, '_fuse_stack', fuse)
    common = ['--input', str(stack), '--method', 'dct']
    assert _cli(monkeypatch, tmp_path, *common, '--output', str(tmp_path / 'a.png'),
                '--temp-dir', str(first)) == 0
    assert _cli(monkeypatch, tmp_path, *common, '--output', str(tmp_path / 'b.png'),
                '--temp-dir', str(second)) == 1
    assert [os.path.dirname(path) for path in paths] == [str(first), str(second)]
    assert frame_storage._storage_directory == str(saved)
    assert frame_storage.capture_scratch() is None
    assert states[1].closed
    assert not os.path.exists(paths[1])
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, sources: iter([str(automatic)]))
    assert _cli(monkeypatch, tmp_path, *common, '--output', str(tmp_path / 'c.png')) == 0
    assert os.path.commonpath((str(automatic), paths[2])) == str(automatic)
    assert frame_storage._storage_directory == str(saved)
    assert frame_storage.capture_scratch() is None


def test_batch_preflight_uses_each_folders_sources_and_options(monkeypatch, tmp_path, capsys):
    from core import cli, storage_preflight
    from utils.image_utils import imwrite_auto

    folders = [tmp_path / 'a', tmp_path / 'b']
    for folder in folders:
        folder.mkdir()
        for index in range(2):
            assert imwrite_auto(str(folder / f'{index}.png'), np.full((24, 32, 3), index * 50, np.uint8))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'a': {'exposure_match': False},
                                  'b': {'method': 'dct', 'exposure_match': True}}),
                      encoding='utf-8')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    scratch_paths = []
    fuse_original = cli._fuse_stack

    def fuse(images, args):
        mapped = frame_storage.store_frame(np.zeros((8, 8), np.uint8), prefer_ram=False)
        scratch_paths.append(mapped.filename)
        return fuse_original(images, args)

    monkeypatch.setattr(cli, '_fuse_stack', fuse)
    estimates = []
    original = storage_preflight.render_storage_estimate

    def estimate(frames, **options):
        estimates.append((tuple(frame.path for frame in frames), options))
        return original(frames, **options)

    monkeypatch.setattr(storage_preflight, 'render_storage_estimate', estimate)
    _assess_as(monkeypatch, 'advisory')
    code = _cli(monkeypatch, tmp_path, '-i', *(str(folder) for folder in folders),
                '-d', str(tmp_path / 'out'), '-m', 'dct', '--config', str(config),
                '--temp-dir', str(scratch))
    assert code == 0
    assert len(scratch_paths) == 2
    assert all(os.path.dirname(path) == str(scratch) for path in scratch_paths)
    assert len(estimates) == 2
    for (paths, _), folder in zip(estimates, folders):
        assert len(paths) == 2 and all(os.path.dirname(path) == str(folder) for path in paths)
    assert estimates[0][1]['exposure_match'] is False
    assert estimates[1][1]['exposure_match'] is True
    errors = capsys.readouterr().err
    assert all(f'for {folder}' in errors for folder in folders)


def test_grouped_batch_skips_whole_stack_storage_estimate(monkeypatch, tmp_path, capsys):
    stack = _cli_stack(tmp_path)
    calls = _assess_as(monkeypatch, 'advisory')
    code = _cli(monkeypatch, tmp_path, '-i', str(stack), '-d', str(tmp_path / 'out'),
                '-m', 'dct', '--cpu', '--group-size', '2')
    assert code == 0, capsys.readouterr().err
    assert calls == []
    assert (tmp_path / 'out' / 'stack.png').is_file()


def test_cli_preflight_reports_unknown_when_runtime_volume_depends_on_goal(monkeypatch, tmp_path):
    from core import cli, storage_preflight

    stack = _cli_stack(tmp_path)
    paths = [str(stack / 'f0.png'), str(stack / 'f1.png')]
    small, large = tmp_path / 'small', tmp_path / 'large'
    small.mkdir(), large.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_space', _fake_space({
        str(small): 4 * GIB + 350 * 1024 ** 2,
        str(large): 20 * GIB,
    }))
    estimate = storage_preflight.render_storage_estimate

    def large_estimate(frames, **options):
        minimum, _, _, sources = estimate(frames, **options)
        return minimum, 400 * 1024 ** 2, 0, sources

    monkeypatch.setattr(storage_preflight, 'render_storage_estimate', large_estimate)
    orders = []

    def candidates(explicit, sources):
        order = [str(small), str(large)] if sources[0] == paths[0] else [str(large), str(small)]
        orders.append(order)
        return iter(order)

    monkeypatch.setattr(frame_storage, '_scratch_candidates', candidates)
    args = cli.build_parser().parse_args(['-i', str(stack), '-o', str(tmp_path / 'out.png'),
                                         '-m', 'dct'])
    for ordered in (paths, list(reversed(paths))):
        result = cli._warn_low_storage(args, ordered)
        assert result['status'] == 'unknown'
        assert result['directory'] is None
    assert len(orders) == 2 and orders[0] == list(reversed(orders[1]))


def test_cli_warns_without_claiming_a_directory_for_multiple_volumes(monkeypatch, tmp_path, capsys):
    stack = _cli_stack(tmp_path)
    assessment = _assessment('advisory', None)
    monkeypatch.setattr(frame_storage, 'assess_scratch', lambda *args, **kwargs: assessment)
    code = _cli(monkeypatch, tmp_path, '-i', str(stack), '-o', str(tmp_path / 'out.png'),
                '-m', 'dct')
    assert code == 0
    warning = capsys.readouterr().err
    assert 'low temporary storage across available volumes' in warning
    assert 'at most ' in warning
    assert 'in None' not in warning


def test_cli_profiles_with_scoped_scratch(monkeypatch, tmp_path):
    from core import cli

    stack = _cli_stack(tmp_path)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    profile = tmp_path / 'profile.json'
    original = cli._fuse_stack
    placements = []

    def fuse(images, args):
        frame_storage.select_scratch(64)
        mapped = frame_storage.store_frame(np.zeros((8, 8), np.uint8), prefer_ram=False)
        placements.append(mapped.filename)
        return original(images, args)

    monkeypatch.setattr(cli, '_fuse_stack', fuse)
    code = _cli(monkeypatch, tmp_path, '-i', str(stack), '-o', str(tmp_path / 'out.png'),
                '-m', 'dct', '--temp-dir', str(scratch), '--profile-performance', str(profile))
    assert code == 0
    assert len(placements) == 1 and os.path.dirname(placements[0]) == str(scratch)
    assert json.loads(profile.read_text(encoding='utf-8'))['outcome'] == 'completed'
    assert frame_storage.capture_scratch() is None

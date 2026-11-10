"""Reproducible CPU benchmarks for StackMFF V3; CPU only, deterministic synthetic input.

Subcommands (each timed configuration runs in its own subprocess):
  sweep     one-tile inference at several thread counts, soft and hard
  sessions  replay neural session keys through the 2-session CPU cache
  tiling    overlap and tile-size comparison with output differences
  cache     resolution-cache activity of a real tiled render, by entry kind
  overhead  the same render with profiling off and on, alternating

--smoke runs a seconds-long version (used by the test suite). --full uses the
historical 82-frame geometry (1040 x 778 portrait, block 389, overlap 256); it
prints the plan, a time estimate and the RAM requirement, and runs only with
--confirm. Nothing reads or writes user settings, photographs or the user's
scratch directory. Output is JSON (stdout, or --json-output); compare two runs
that differ in one setting with any JSON diff.

Examples:
  python tools/stackmff_cpu_benchmark.py tiling --smoke
  python tools/stackmff_cpu_benchmark.py sweep --full              # plan only
  python tools/stackmff_cpu_benchmark.py sweep --full --confirm --json-output sweep.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

# Geometry of a recorded 82-frame CPU render ("historical"); see docs/PERFORMANCE.md.
HISTORICAL = dict(frames=82, height=1040, width=778, block=389, overlap=256)
SMOKE = dict(frames=4, height=200, width=150, block=128, overlap=64)
# Inferred from the historical log's session timestamps (not a measurement of
# this machine): about 10 s for one 82 x 448 x 448 inference on 2 threads.
SECONDS_PER_FRAME_PIXEL_2_THREADS = 10.0 / (82 * 448 * 448)
# Peak RSS of the historical 82-frame render (whole pipeline, incl. RAW decode).
HISTORICAL_PEAK_RSS_BYTES = int(9.7e9)
# Completed configurations are appended here as they finish, so a run that is
# killed part-way (for example by a memory limit) keeps what it measured.
_CHECKPOINT = None


# ---- shared helpers ---------------------------------------------------------

def _padded(value):
    return value + (-value) % 64


def _git(root):
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                                     '--untracked-files=no'], capture_output=True, text=True,
                                    timeout=5, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def synthetic_stack(frames, height, width, seed):
    """Deterministic focus stack: a sharp texture blurred by distance to a depth ramp."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    texture = rng.random((height, width, 3), dtype=np.float32) * 255
    levels = [texture] + [cv2.GaussianBlur(texture, (0, 0), sigma) for sigma in (1.0, 2.0, 3.0, 4.0)]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = (0.6 * yy / max(1, height - 1) + 0.4 * xx / max(1, width - 1)) * (frames - 1)
    stack = []
    for index in range(frames):
        blur = np.clip(np.abs(index - depth) * 1.2, 0, len(levels) - 1.001)
        low = blur.astype(np.int32)
        fraction = (blur - low)[..., None]
        image = np.zeros_like(texture)
        for level in range(len(levels) - 1):
            mask = (low == level)[..., None]
            image += mask * (levels[level] * (1 - fraction) + levels[level + 1] * fraction)
        stack.append(np.clip(np.rint(image), 0, 255).astype(np.uint8))
    return stack


def _sha256_frames(frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(memoryview(frame.tobytes()))
    return digest.hexdigest()


def estimate_tile_seconds(frames, block, threads):
    """Rough single-tile estimate from the historical rate; scaling assumed ideal."""
    full = _padded(block) ** 2
    reduced = _padded(int(_padded(block) * 0.75)) ** 2
    inference = SECONDS_PER_FRAME_PIXEL_2_THREADS * frames * (full + reduced) * 2 / max(1, threads)
    return inference * 1.3   # resize, softmax and reconstruction measured at ~25% in the log


def estimate_ram_bytes(frames, height, width, block):
    """Planner-equivalent working set of one tile plus the retained stack."""
    from fusion_methods.neural_runtime import estimate_inference_bytes
    padded = _padded(block) ** 2
    crops = frames * block * block * 3 * 8
    staging = frames * padded * 4 * (2 * 3 + 3)
    inference = estimate_inference_bytes(frames, block, block, 'CPU')
    stack = frames * height * width * 3
    output = height * width * (4 * 4 + 3)
    return crops + staging + inference + stack + output


def _run_worker(root, scratch, config, timeout):
    """One measured configuration in a fresh interpreter; returns its JSON result."""
    env = dict(os.environ, OPENFOCUS_DATA_DIR=str(scratch / 'data'),
               OPENFOCUS_LOG_DIR=str(scratch / 'logs'), TMPDIR=str(scratch / 'tmp'),
               QT_QPA_PLATFORM='offscreen')
    for name in ('data', 'logs', 'tmp', 'storage'):
        (scratch / name).mkdir(exist_ok=True)
    (scratch / 'data' / 'settings.ini').write_text('[usage]\nenabled=false\n', encoding='utf-8')
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', json.dumps(config),
               '--repo-root', str(root), '--scratch', str(scratch)]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(root),
                               env=env, timeout=timeout)
    lines = [line for line in completed.stdout.splitlines() if line.startswith('{')]
    if completed.returncode != 0 or not lines:
        result = {'config': config, 'error': (completed.stderr or completed.stdout)[-2000:],
                  'returncode': completed.returncode}
    else:
        result = json.loads(lines[-1])
    if _CHECKPOINT is not None:
        with open(_CHECKPOINT, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'config': config, 'result': result}, default=str) + '\n')
    return result


def _summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {'median': round(statistics.median(values), 4), 'min': round(min(values), 4),
            'max': round(max(values), 4), 'n': len(values)}


# ---- worker-side measurements ----------------------------------------------

def _worker_setup(scratch):
    from core.frame_storage import configure_storage_directory
    configure_storage_directory(str(scratch / 'storage'))


def _stage(report, suffix):
    return sum(item['total_ms'] for path, item in report['stages'].items()
               if path.endswith(suffix)) / 1000


def worker_sweep(config, scratch):
    """Cold first tile, then warm repetitions of the identical tile."""
    import numpy as np
    from fusion_methods.stackmffv3 import _stackmffv3_batch_impl
    from utils import performance_profile as pp
    from utils.memory import peak_rss_bytes
    frames = synthetic_stack(config['frames'], config['block'], config['block'], config['seed'])
    results = []
    for repetition in range(1 + config['repetitions']):
        profile = pp.PerformanceProfile()
        started = time.perf_counter()
        with pp.active(profile):
            output = _stackmffv3_batch_impl([frames], 'weights/stackmff_v3_final.onnx', False,
                                            config['mode'], thread_count=config['threads'],
                                            resource_preference='maximum')[0]
        total = time.perf_counter() - started
        report = profile.report('completed')
        results.append({
            'total_s': total,
            'session_construct_s': report['sessions']['construct_ms'] / 1000,
            'logits_full_s': _stage(report, 'logits.full'),
            'resize_and_reduced_s': _stage(report, 'resize.0.75') + _stage(report, 'logits.reduced'),
            'probability_processing_s': (_stage(report, 'softmax.full') + _stage(report, 'softmax.reduced')
                                         + _stage(report, 'upsample_combine') + _stage(report, 'gaussian')),
            'colour_reconstruction_s': _stage(report, 'blend_tile'),
            'intra_op_threads': report.get('threads', {}).get('session_intra_op'),
        })
    warm = results[1:]
    megapixel_frames = config['frames'] * config['block'] ** 2 / 1e6
    return {'config': config, 'input_sha256': _sha256_frames(frames),
            'output_sha256': hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
            'cold': results[0], 'warm': {key: _summary([r[key] for r in warm])
                                         for key in results[0] if key != 'intra_op_threads'},
            'intra_op_threads': results[0]['intra_op_threads'],
            'throughput_megapixel_frames_per_s': _summary(
                [megapixel_frames / r['total_s'] for r in warm]),
            'peak_rss_bytes': peak_rss_bytes()}


def worker_sessions(config, scratch):
    """Construct sessions in the recorded order through the real 2-entry cache."""
    from fusion_methods import neural_runtime
    from utils import performance_profile as pp
    model = os.path.abspath('weights/stackmff_v3_final.onnx')
    profile = pp.PerformanceProfile()
    started = time.perf_counter()
    with pp.active(profile):
        for shape in config['sequence']:
            neural_runtime._session(model, tuple(shape), 'CPU', config['threads'])
    elapsed = time.perf_counter() - started
    sessions = profile.report('completed')['sessions']
    lookups = sessions['created'] + sessions['hits']
    return {'config': {k: v for k, v in config.items() if k != 'sequence'},
            'lookups': lookups, 'created': sessions['created'], 'hits': sessions['hits'],
            'evictions': sessions['evictions'],
            'hit_ratio': round(sessions['hits'] / lookups, 4) if lookups else None,
            'construct_s': sessions['construct_ms'] / 1000, 'hash_s': sessions['hash_ms'] / 1000,
            'construct_s_per_session': (round(sessions['construct_ms'] / 1000 / sessions['created'], 4)
                                        if sessions['created'] else None),
            'replay_wall_s': round(elapsed, 4), 'keys': sessions['keys']}


def _render_worker(frames, config, profile_performance):
    from core.workers import RenderWorker
    # Only pass the flag when profiling, so --baseline-root can run a checkout
    # that predates it.
    extra = {'profile_performance': True} if profile_performance else {}
    return RenderWorker(frames, algorithm='stackmffv3', use_gpu=False,
                        thread_count=config['threads'], exposure_match=False,
                        fusion_parameters={'mode': config['mode']}, tile_enabled=True,
                        tile_block_size=config['block'], tile_overlap=config['overlap'],
                        tile_threshold=32, resource_preference='maximum', **extra)


def _run_render(worker):
    outcome = {}
    worker.finished_signal.connect(lambda *value: outcome.setdefault('result', value))
    worker.error_signal.connect(lambda message: outcome.setdefault('error', message))
    started = time.perf_counter()
    worker.run()
    return time.perf_counter() - started, outcome


def _stack_files(config, scratch):
    """Write the synthetic stack once as PNG so renders use real source descriptors."""
    from core.source_frames import scan_source
    from utils.image_utils import imwrite_auto
    directory = scratch / 'stack'
    if not directory.is_dir():
        directory.mkdir()
        for index, image in enumerate(synthetic_stack(config['frames'], config['height'],
                                                      config['width'], config['seed'])):
            assert imwrite_auto(str(directory / f'frame{index:03d}.png'), image)
    return [scan_source(str(path))[0] for path in sorted(directory.glob('frame*.png'))]


def worker_render(config, scratch):
    """One tiled render (tiling, cache and overhead benchmarks)."""
    import numpy as np
    from core.source_frames import resolve_frame
    frames = _stack_files(config, scratch)
    if config.get('warmup'):
        _run_render(_render_worker(frames, config, False))
    worker = _render_worker(frames, config, config['profile'])
    elapsed, outcome = _run_render(worker)
    if 'error' in outcome:
        return {'config': config, 'error': outcome['error']}
    fused = np.ascontiguousarray(resolve_frame(outcome['result'][1]))
    result = {'config': config, 'wall_s': round(elapsed, 4),
              'output_sha256': hashlib.sha256(fused.tobytes()).hexdigest()}
    if config.get('save_output'):
        path = scratch / f"output-{config['label']}.npy"
        np.save(path, fused)
        result['output_path'] = str(path)
    report = getattr(worker, 'performance_report', None)
    if report:
        result['report'] = report
    return result


WORKERS = {'sweep': worker_sweep, 'sessions': worker_sessions, 'render': worker_render}


# ---- parent-side benchmarks --------------------------------------------------

def _thread_counts(requested):
    from core.resource_policy import cpu_capacity, probe_resources
    capacity = cpu_capacity(probe_resources(), 'maximum')
    kept = [count for count in requested if count <= capacity]
    skipped = [count for count in requested if count > capacity]
    return kept, skipped, capacity


def _plan(args, description, estimate_s, ram_bytes, *, whole_render=False):
    plan = {'mode': 'full' if args.full else 'smoke', 'description': description,
            'estimated_seconds': round(estimate_s), 'estimated_ram_bytes': int(ram_bytes),
            'ram_basis': 'planner working-set formula for the largest tile plus the stack',
            'estimate_basis': 'historical log timestamps (~10 s per 82x448x448 inference '
                              'on 2 threads), ideal thread scaling assumed; not a measurement'}
    if whole_render and args.full:
        plan['historical_peak_rss_bytes'] = HISTORICAL_PEAK_RSS_BYTES
        plan['ram_note'] = ('the historical render of this geometry peaked at 9.7 GB RSS '
                            '(it also decoded RAW files); keep that much free')
    if args.full:
        plan['ram_measured_note'] = ('measured peak RSS has been 1.6-1.7x this estimate '
                                     '(docs/PERFORMANCE.md); a 640 tile exceeded 16 GiB')
    print(json.dumps({'plan': plan}, indent=2), file=sys.stderr)
    if args.full and not args.confirm:
        print('Full mode prints its plan only; rerun with --confirm to start it.', file=sys.stderr)
        return False
    return True


def _geometry(args):
    geometry = dict(HISTORICAL if args.full else SMOKE)
    for key in ('frames', 'height', 'width', 'block', 'overlap'):
        if getattr(args, key) is not None:
            geometry[key] = getattr(args, key)
    return geometry


def bench_sweep(args, root, scratch):
    geometry = _geometry(args)
    frames = geometry['frames']
    # The sweep's one tile is the padded 448 inference of the historical grid.
    block = args.block or (448 if args.full else geometry['block'])
    threads, skipped, capacity = _thread_counts(args.threads or ([1, 2, 4, 6, 8] if args.full else [1, 2]))
    modes = ['soft', 'hard']
    repetitions = args.repetitions or 3
    estimate = sum(estimate_tile_seconds(frames, block, t) * (repetitions + 1)
                   for t in threads for _ in modes)
    if not _plan(args, f'{len(threads) * len(modes)} configurations of one {frames}x{block}x{block} '
                 f'tile, {repetitions} warm repetitions each', estimate,
                 estimate_ram_bytes(frames, block, block, block)):
        return None
    results = [_run_worker(root, scratch, {'kind': 'sweep', 'frames': frames, 'block': block,
                                           'threads': count, 'mode': mode, 'seed': args.seed,
                                           'repetitions': repetitions}, args.timeout)
               for mode in modes for count in threads]
    return {'results': results, 'maximum_capacity': capacity,
            'skipped_threads': {str(count): f'exceeds cpu_capacity(maximum)={capacity}'
                                for count in skipped}}


def historical_sequence(frames, height, width, block, overlap):
    """Session shapes in tile order, padded exactly as StackMFF V3 pads them."""
    from core.multi_focus_fusion import _ai_effective_tiling
    _, _, coords = _ai_effective_tiling(frames, block, height, width, overlap, 1)
    sequence = []
    for x0, y0, x1, y1 in coords:
        full = (_padded(y1 - y0), _padded(x1 - x0))
        reduced = tuple(_padded(int(side * 0.75)) for side in full)
        sequence += [(frames, *full), (frames, *reduced)]
    return sequence, len(coords)


def bench_sessions(args, root, scratch):
    geometry = _geometry(args)
    frames = geometry['frames']
    recorded, tiles = historical_sequence(frames, geometry['height'], geometry['width'],
                                          geometry['block'], geometry['overlap'])
    shapes = [(frames, 64 * a, 64 * b) for a, b in ((2, 2), (2, 3), (3, 2))]
    synthetic = shapes * 4
    if not _plan(args, f'replay {len(recorded)} recorded and {len(synthetic)} synthetic lookups',
                 len(recorded) + len(synthetic), 512 * 1024 ** 2):
        return None
    threads = args.threads[0] if args.threads else 2
    return {'recorded': _run_worker(root, scratch, {'kind': 'sessions', 'threads': threads,
                                                    'sequence': recorded, 'tiles': tiles,
                                                    'geometry': geometry}, args.timeout),
            'synthetic_eviction': _run_worker(root, scratch, {'kind': 'sessions', 'threads': threads,
                                                              'sequence': synthetic}, args.timeout)}


def _boundary_mask(height, width, coords, band=8):
    import numpy as np
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in coords:
        for x in (x0, x1):
            if 0 < x < width:
                mask[:, max(0, x - band):x + band] = True
        for y in (y0, y1):
            if 0 < y < height:
                mask[max(0, y - band):y + band, :] = True
    return mask


def _compare(reference, candidate, ref_coords, coords, diff_dir, label):
    import numpy as np
    a = np.load(reference['output_path']).astype(np.int32)
    b = np.load(candidate['output_path']).astype(np.int32)
    diff = np.abs(a - b)
    plane = diff.max(axis=2) if diff.ndim == 3 else diff
    mask = _boundary_mask(a.shape[0], a.shape[1], list(ref_coords) + list(coords))
    result = {'mean_abs_diff': round(float(diff.mean()), 5), 'max_abs_diff': int(diff.max()),
              'boundary_mean_abs_diff': round(float(plane[mask].mean()), 5) if mask.any() else None,
              'interior_mean_abs_diff': round(float(plane[~mask].mean()), 5) if (~mask).any() else None,
              'identical': bool(not diff.any())}
    if diff_dir is not None:
        from utils.image_utils import imwrite_auto
        imwrite_auto(str(Path(diff_dir) / f'diff-{label}.png'), np.clip(diff, 0, 255).astype(np.uint8))
    return result


def _tile_sizes(args, defaults, base_block):
    """--tile-sizes replaces the default comparison sizes; the base block always runs."""
    chosen = defaults if args.tile_sizes is None else args.tile_sizes
    return sorted({base_block, *chosen})


def bench_tiling(args, root, scratch):
    geometry = _geometry(args)
    if args.full:
        base_block, sizes = 448, _tile_sizes(args, (256, 448, 640), 448)
        overlaps = [224, 128, 64]
    else:
        base_block, sizes = 128, _tile_sizes(args, (64, 128), 128)
        overlaps = [64, 32, 16]
    threads = args.threads[0] if args.threads else 2
    configs = [dict(label=f'historical-b{geometry["block"]}-o{geometry["overlap"]}',
                    block=geometry['block'], overlap=geometry['overlap'])]
    configs += [dict(label=f'b{base_block}-o{overlap}', block=base_block, overlap=overlap)
                for overlap in overlaps]
    configs += [dict(label=f'b{size}-o{overlaps[0]}', block=size, overlap=overlaps[0])
                for size in sizes if size != base_block]
    from core.multi_focus_fusion import _ai_effective_tiling
    estimate = 0
    for config in configs:
        block, _, coords = _ai_effective_tiling(geometry['frames'], config['block'], geometry['height'],
                                                geometry['width'], config['overlap'], 1)
        estimate += len(coords) * estimate_tile_seconds(geometry['frames'], block, threads)
    if not _plan(args, f'{len(configs)} tiled renders of {geometry["frames"]} x '
                 f'{geometry["height"]}x{geometry["width"]}', estimate,
                 estimate_ram_bytes(geometry['frames'], geometry['height'], geometry['width'],
                                    max(c['block'] for c in configs)),
                 whole_render=True):
        return None
    results = {}
    for config in configs:
        run = {**geometry, **config, 'kind': 'render', 'profile': True, 'save_output': True,
               'threads': threads, 'mode': args.mode, 'seed': args.seed}
        results[config['label']] = _run_worker(root, scratch, run, args.timeout)
    reference_label = f'b{base_block}-o{overlaps[0]}'
    reference = results[reference_label]
    reference_coords = (reference['report']['tiling'].get('coords') or []
                        if 'error' not in reference else [])
    if args.diff_images:
        Path(args.diff_images).mkdir(parents=True, exist_ok=True)
    table = []
    for label, result in results.items():
        if 'error' in result:
            table.append({'label': label, 'error': result['error']})
            continue
        report = result['report']
        tiling = report['tiling']
        row = {'label': label, 'tiles': tiling['count'], 'coords': tiling.get('coords'),
               'block_requested': tiling['block_requested'],
               'block_after_memory_plan': tiling['block_after_memory_plan'],
               'block_effective': tiling['block_effective'],
               'overlap_requested': tiling['overlap_requested'],
               'overlap_effective': tiling['overlap_effective'],
               'clipped_area_ratio': tiling['clipped_area_ratio'],
               'padded_inference_area_ratio': tiling['padded_inference_area_ratio'],
               'redundancy': tiling['redundancy'],
               'inference_calls': sum(key['runs'] for key in report['sessions']['keys']),
               'sessions_created': report['sessions']['created'],
               'fusion_s': round(report['stages']['render/fusion']['total_ms'] / 1000, 4),
               'wall_s': result['wall_s'], 'output_sha256': result['output_sha256']}
        if 'error' not in reference:
            row['vs_reference'] = _compare(reference, result, reference_coords,
                                           tiling.get('coords') or [], args.diff_images, label)
        table.append(row)
    return {'geometry': geometry, 'reference': reference_label, 'rows': table}


def bench_cache(args, root, scratch):
    geometry = _geometry(args)
    threads = args.threads[0] if args.threads else 2
    from core.multi_focus_fusion import _ai_effective_tiling
    block, _, coords = _ai_effective_tiling(geometry['frames'], geometry['block'], geometry['height'],
                                            geometry['width'], geometry['overlap'], 1)
    if not _plan(args, f'one tiled render, {len(coords)} tiles',
                 len(coords) * estimate_tile_seconds(geometry['frames'], block, threads),
                 estimate_ram_bytes(geometry['frames'], geometry['height'], geometry['width'], block),
                 whole_render=True):
        return None
    result = _run_worker(root, scratch, {**geometry, 'kind': 'render', 'profile': True,
                                         'threads': threads, 'mode': args.mode,
                                         'seed': args.seed, 'label': 'cache'}, args.timeout)
    if 'error' in result:
        return result
    report = result['report']
    cache = report['cache']
    crop = cache['by_kind'].get('crop', {})
    return {'geometry': geometry, 'cache': cache,
            'resolve_region_s': round(sum(item['total_ms'] for path, item in report['stages'].items()
                                          if path.endswith('resolve_region')) / 1000, 4),
            'fusion_s': round(report['stages']['render/fusion']['total_ms'] / 1000, 4),
            'estimated_savings_if_crops_excluded': {
                'bytes_not_written': crop.get('bytes_written', 0),
                'put_seconds_avoided': round(crop.get('put_ms', 0) / 1000, 4),
                'entries_not_created': crop.get('puts', 0),
                'basis': 'crop entries are written once and never hit; avoided put time is the '
                         'measured cache put time for crop entries only'},
            'report': report}


def bench_overhead(args, root, scratch):
    geometry = _geometry(args)
    repetitions = max(5, args.repetitions or 5)
    threads = args.threads[0] if args.threads else 2
    from core.multi_focus_fusion import _ai_effective_tiling
    block, _, coords = _ai_effective_tiling(geometry['frames'], geometry['block'], geometry['height'],
                                            geometry['width'], geometry['overlap'], 1)
    arms = ['off', 'on'] + (['baseline'] if args.baseline_root else [])
    # Every timed render follows a warm-up render in the same fresh process.
    timed = len(arms) * repetitions
    estimate = 2 * timed * len(coords) * estimate_tile_seconds(geometry['frames'], block, threads)
    if not _plan(args, f'{timed} timed renders (profiling {", ".join(arms)}; rotating order), '
                       f'each after a warm-up render: {2 * timed} renders in all', estimate,
                 estimate_ram_bytes(geometry['frames'], geometry['height'], geometry['width'], block),
                 whole_render=True):
        return None
    times = {arm: [] for arm in arms}
    hashes = {}
    for index in range(repetitions):
        order = arms[index % len(arms):] + arms[:index % len(arms)]   # rotate the order
        for arm in order:
            arm_root = args.baseline_root.resolve() if arm == 'baseline' else root
            result = _run_worker(arm_root, scratch, {**geometry, 'kind': 'render',
                                                     'profile': arm == 'on', 'warmup': True,
                                                     'threads': threads, 'mode': args.mode,
                                                     'seed': args.seed, 'label': 'overhead'},
                                 args.timeout)
            if 'error' in result:
                return result
            times[arm].append(result['wall_s'])
            hashes.setdefault(arm, set()).add(result['output_sha256'])
    medians = {arm: statistics.median(values) for arm, values in times.items()}
    output = {'geometry': geometry, 'threads': threads,
              **{f'{arm}_s': _summary(values) for arm, values in times.items()},
              'on_vs_off_percent': round(100 * (medians['on'] - medians['off']) / medians['off'], 2),
              'outputs_identical': len(set().union(*hashes.values())) == 1}
    if args.baseline_root:
        baseline_commit, baseline_dirty = _git(args.baseline_root)
        output.update(baseline_root=str(args.baseline_root), baseline_commit=baseline_commit,
                      off_vs_baseline_percent=round(
                          100 * (medians['off'] - medians['baseline']) / medians['baseline'], 2))
    return output


BENCHMARKS = {'sweep': bench_sweep, 'sessions': bench_sessions, 'tiling': bench_tiling,
              'cache': bench_cache, 'overhead': bench_overhead}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('benchmark', choices=sorted(BENCHMARKS) + ['_worker'])
    parser.add_argument('config', nargs='?', help=argparse.SUPPRESS)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--smoke', action='store_true', help='seconds-long run (default)')
    size.add_argument('--full', action='store_true', help='historical 82-frame geometry')
    parser.add_argument('--confirm', action='store_true', help='actually start a --full run')
    parser.add_argument('--threads', type=int, nargs='+', help='thread counts (sweep) or one count')
    parser.add_argument('--frames', type=int, help='override the frame count')
    parser.add_argument('--height', type=int, help='override the output height')
    parser.add_argument('--width', type=int, help='override the output width')
    parser.add_argument('--block', type=int, help='override the requested tile size')
    parser.add_argument('--overlap', type=int, help='override the requested overlap')
    parser.add_argument('--baseline-root', type=Path,
                        help='overhead: also time this checkout (code before profiling)')
    parser.add_argument('--tile-sizes', type=int, nargs='*',
                        help='tiling: compare these tile sizes instead of the defaults '
                             '(the base size always runs)')
    parser.add_argument('--mode', choices=('soft', 'hard'), default='soft')
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--diff-images', help='write difference images to this directory (tiling)')
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scratch', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    if args.benchmark == '_worker':
        os.chdir(root)
        _worker_setup(args.scratch)
        config = json.loads(args.config)
        print(json.dumps(WORKERS[config['kind']](config, args.scratch), default=str))
        return 0
    commit, dirty = _git(root)
    global _CHECKPOINT
    if args.json_output:
        _CHECKPOINT = args.json_output.with_name(args.json_output.name + '.partial.jsonl')
        _CHECKPOINT.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='openfocus-stackmff-bench-') as directory:
        scratch = Path(directory)
        output = BENCHMARKS[args.benchmark](args, root, scratch)
        if output is None:
            return 0
        import cv2
        import numpy as np
        import onnxruntime
        from utils.platform_utils import cpu_limits
        cores, logical, limit, known = cpu_limits()
        payload = {'benchmark': args.benchmark, 'mode': 'full' if args.full else 'smoke',
                   'commit': commit, 'dirty': dirty, 'seed': args.seed,
                   'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                                'opencv': cv2.__version__, 'onnxruntime': onnxruntime.__version__},
                   'cpu': {'physical_cores': cores, 'logical_cpus': logical, 'cpu_limit': limit,
                           'physical_cores_known': known},
                   'arguments': {key: (str(value) if isinstance(value, Path) else value)
                                 for key, value in vars(args).items() if key not in ('config', 'scratch')},
                   'output': output}
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        args.json_output.write_text(text + '\n', encoding='utf-8')
        if _CHECKPOINT is not None:
            _CHECKPOINT.unlink(missing_ok=True)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())

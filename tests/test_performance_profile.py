"""Opt-in performance profile: semantics, bounds and failure isolation."""
import json
import threading

import pytest

from utils import performance_profile as pp


class FakeClock:
    """Each read advances by the next scripted step (nanoseconds)."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += int(ms * 1e6)


def test_disabled_profile_is_a_shared_no_op():
    assert pp.current() is pp.NULL_PROFILE
    assert not pp.current().enabled
    first = pp.NULL_PROFILE.stage('a')
    assert first is pp.NULL_PROFILE.stage('b', resources=True)
    assert first is pp.NULL_PROFILE.tile(0, (0, 0, 1, 1), 2)
    with first:
        pass
    # Recording calls are accepted and ignored.
    assert pp.NULL_PROFILE.session_lookup(('m', (1,), 'CPU', 1), False) is None
    assert pp.NULL_PROFILE.cache_event('crop', 'put', 10) is None


def test_activation_is_scoped_and_restored():
    profile = pp.PerformanceProfile()
    with pp.active(profile):
        assert pp.current() is profile
        inner = pp.PerformanceProfile()
        with pp.active(inner):
            assert pp.current() is inner
        assert pp.current() is profile
    assert pp.current() is pp.NULL_PROFILE


def test_nested_stages_report_inclusive_and_exclusive_time():
    clock = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
    with profile.stage('render'):
        clock.advance(5)
        for _ in range(2):
            with profile.stage('tile'):
                clock.advance(10)
                with profile.stage('infer'):
                    clock.advance(30)
        clock.advance(5)
    stages = profile.report('completed')['stages']
    assert stages['render']['total_ms'] == 90
    assert stages['render']['self_ms'] == 10
    assert stages['render']['parent'] is None
    assert stages['render/tile'] == {
        'count': 2, 'total_ms': 80, 'self_ms': 20, 'mean_ms': 40, 'min_ms': 40,
        'max_ms': 40, 'order': 1, 'parent': 'render', 'pct_of_parent': 88.89}
    assert stages['render/tile/infer']['total_ms'] == 60
    assert stages['render/tile/infer']['pct_of_parent'] == 75.0
    # Children never exceed their parent; self times add up to the root total.
    assert sum(item['self_ms'] for item in stages.values()) == stages['render']['total_ms']


def test_min_max_mean_and_optional_cpu_time():
    clock = FakeClock()
    cpu = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=cpu)
    for ms in (1, 4, 7):
        with profile.stage('step', resources=True):
            clock.advance(ms)
            cpu.advance(ms * 3)
    with profile.stage('wall_only'):
        clock.advance(1)
    stages = profile.report('completed')['stages']
    assert (stages['step']['min_ms'], stages['step']['max_ms'], stages['step']['mean_ms']) == (1, 7, 4)
    assert stages['step']['process_cpu_ms'] == 36
    assert 'process_cpu_ms' not in stages['wall_only']


def test_major_stages_report_how_far_they_raised_peak_memory():
    peak = [100]
    profile = pp.PerformanceProfile(peak_rss=lambda: peak[0])
    with profile.stage('render', resources=True):
        with profile.stage('registration', resources=True):
            peak[0] = 400
        with profile.stage('fusion', resources=True):
            pass                      # the high-water mark does not move
        with profile.stage('detail'):
            peak[0] = 450             # not a major stage: no memory fields
    report = profile.report('completed')
    stages = report['stages']
    assert stages['render/registration']['peak_rss_growth_bytes'] == 300
    assert stages['render/registration']['peak_rss_bytes'] == 400
    assert stages['render/fusion']['peak_rss_growth_bytes'] == 0
    assert stages['render']['peak_rss_growth_bytes'] == 350
    assert 'peak_rss_bytes' not in stages['render/detail']
    assert report['render']['peak_rss_bytes'] == 450
    assert 'peak memory +300 B' in pp.summary(report)


def test_unreadable_peak_memory_is_omitted_not_reported_as_zero():
    profile = pp.PerformanceProfile(peak_rss=lambda: 0)
    with profile.stage('render', resources=True):
        pass
    report = profile.report('completed')
    assert 'peak_rss_bytes' not in report['stages']['render']
    assert report['render']['peak_rss_bytes'] is None


def test_stage_order_follows_execution_not_names():
    clock = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
    with profile.stage('render'):
        for name in ('registration', 'fusion', 'alignment_cache_save'):
            with profile.stage(name):
                clock.advance(1)
        with profile.stage('fusion'):       # a repeat keeps its first position
            clock.advance(1)
    report = profile.report('completed')
    ordered = sorted(report['stages'], key=lambda path: report['stages'][path]['order'])
    assert ordered == ['render', 'render/registration', 'render/fusion',
                       'render/alignment_cache_save']
    text = pp.summary(report)
    positions = [text.index(f'{name}: x') for name in
                 ('registration', 'fusion', 'alignment_cache_save')]
    assert positions == sorted(positions)


def test_stages_from_other_threads_are_reported_as_concurrent():
    profile = pp.PerformanceProfile()
    with pp.active(profile), profile.stage('registration'):
        def decode():
            with pp.current().stage('source_decode'):
                pass
        threads = [threading.Thread(target=decode) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    report = profile.report('completed')
    decode = report['concurrent']['source_decode']
    assert decode['count'] == 4
    assert decode['max_ms'] <= decode['total_ms']
    assert set(report['stages']) == {'registration'}


def test_exception_propagates_unchanged_and_partial_report_is_labelled():
    profile = pp.PerformanceProfile()
    error = RuntimeError('original')
    with pytest.raises(RuntimeError) as raised:
        with profile.stage('fusion'):
            with profile.stage('tile'):
                raise error
    assert raised.value is error
    report = profile.report('failed')
    assert report['outcome'] == 'failed'
    assert report['stages']['fusion/tile']['count'] == 1


def test_bookkeeping_failure_never_escapes(monkeypatch):
    profile = pp.PerformanceProfile()
    monkeypatch.setattr(profile, '_record', lambda *args: 1 / 0)
    with profile.stage('render'):
        pass
    assert profile.errors == 1
    assert profile.report('completed')['profiler_errors'] == 1


def test_memory_is_bounded_for_many_tiles():
    clock = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
    coords = [(x, 0, x + 8, 8) for x in range(0, 80000, 8)]
    profile.tiling(height=8, width=80000, coords=coords, overlap_requested=0,
                   overlap_effective=0, block_effective=8)
    for index, coord in enumerate(coords):
        with profile.tile(index, coord, 3):
            with profile.stage('runner'):
                clock.advance(1 + index % 97)
    report = profile.report('completed')
    assert len(coords) == 10000
    assert len(report['slowest_tiles']) == pp.SLOWEST_TILES
    assert all(tile['total_ms'] == 97 for tile in report['slowest_tiles'])
    assert report['tiling']['count'] == 10000
    assert len(report['tiling']['coords']) == pp.MAX_TILE_COORDS
    assert report['tiling']['coords_truncated'] is True
    assert report['stages']['tile']['count'] == 10000
    assert len(pp.dumps(report)) < 64 * 1024
    assert 'coords' not in pp.compact(report)['tiling']
    assert 'coords' in report['tiling']                  # the full report is not altered


def test_slowest_tile_keeps_stage_breakdown_and_session_creations():
    clock = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
    with profile.tile(0, (0, 0, 4, 2), 5):
        with profile.stage('runner'):
            clock.advance(2)
            with profile.stage('session.acquire'):
                profile.session_lookup(('m.onnx', (5, 64, 64), 'CPU', 2), False)
                clock.advance(3)
    tile = profile.report('completed')['slowest_tiles'][0]
    assert tile['coords'] == [0, 0, 4, 2]
    assert (tile['width'], tile['height'], tile['frames']) == (4, 2, 5)
    assert tile['sessions_created'] == 1
    assert tile['stages_self_ms'] == {'runner': 2, 'session.acquire': 3}


def test_report_serialization_is_stable_and_versioned():
    def build():
        clock = FakeClock()
        profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
        profile.add('threads', 'session_intra_op', 4)
        profile.add('threads', 'session_intra_op', 2)
        profile.set('render', frames=3)
        with profile.stage('b'):
            clock.advance(1)
        with profile.stage('a'):
            clock.advance(2)
        report = profile.report('completed')
        report['render'].pop('peak_rss_bytes')
        return report
    first, second = build(), build()
    assert pp.dumps(first) == pp.dumps(second)
    decoded = json.loads(pp.dumps(first))
    assert decoded['schema_version'] == pp.SCHEMA_VERSION
    assert decoded['threads']['session_intra_op'] == [2, 4]
    assert list(decoded['stages']) == ['a', 'b']


def test_session_miss_reasons_and_counts():
    profile = pp.PerformanceProfile()
    a = ('m.onnx', (82, 448, 448), 'CPU', 2)
    b = ('m.onnx', (82, 384, 384), 'CPU', 2)
    c = ('m.onnx', (82, 448, 384), 'CPU', 2)
    a4 = ('m.onnx', (82, 448, 448), 'CPU', 4)
    profile.session_lookup(a, False)
    profile.session_lookup(b, False)
    profile.session_lookup(a, True)
    profile.session_run(a)
    profile.session_evicted(a)
    profile.session_lookup(c, False)
    profile.session_lookup(a, False)
    profile.sessions_cleared([b])
    profile.session_lookup(b, False)
    profile.session_lookup(a4, False)
    sessions = profile.report('completed')['sessions']
    assert (sessions['created'], sessions['hits'], sessions['evictions'],
            sessions['clears']) == (6, 1, 1, 1)
    reasons = {(tuple(k['shape']), k['threads']): k['miss_reasons'] for k in sessions['keys']}
    assert reasons[((82, 448, 448), 2)] == {'evicted': 1, 'first_use': 1}
    assert reasons[((82, 384, 384), 2)] == {'cleared': 1, 'new_shape': 1}
    assert reasons[((82, 448, 384), 2)] == {'new_shape': 1}
    assert reasons[((82, 448, 448), 4)] == {'new_key_same_shape': 1}
    runs = {(tuple(k['shape']), k['threads']): k['runs'] for k in sessions['keys']}
    assert runs[((82, 448, 448), 2)] == 1


def test_cache_counters_are_separated_by_entry_kind():
    profile = pp.PerformanceProfile()
    profile.cache_event('crop', 'miss')
    profile.cache_event('crop', 'put', 100, 2_000_000)
    profile.cache_event('source', 'hit')
    profile.cache_state(retained_bytes=100)
    profile.cache_state(retained_bytes=50, degraded=True)
    cache = profile.report('completed')['cache']
    assert cache['by_kind']['crop'] == {'lookups': 1, 'hits': 0, 'misses': 1, 'puts': 1,
                                        'bytes_written': 100, 'put_ms': 2.0}
    assert cache['by_kind']['source']['hits'] == 1
    assert cache['peak_retained_bytes'] == 100
    assert cache['degraded_events'] == 1


def test_tiling_metadata_matches_the_real_grid():
    from core.multi_focus_fusion import _ai_effective_tiling
    # The historical 82-frame render: 1040 x 778 portrait, block 389, overlap 256.
    block, _, coords = _ai_effective_tiling(82, 389, 1040, 778, 256, 1)
    profile = pp.PerformanceProfile()
    profile.tiling(height=1040, width=778, coords=coords, overlap_requested=256,
                   overlap_effective=min(256, block // 2), block_effective=block,
                   pad_alignment=64)
    tiling = profile.report('completed')['tiling']
    assert tiling['count'] == len(coords) == 20
    assert tiling['coords'] == [list(c) for c in coords]
    assert tiling['overlap_effective'] == 194
    clipped = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in coords)
    assert tiling['clipped_area_ratio'] == round(clipped / (1040 * 778), 4)
    padded = sum((-(-(x1 - x0) // 64) * 64) * (-(-(y1 - y0) // 64) * 64)
                 for x0, y0, x1, y1 in coords)
    assert tiling['padded_inference_area_ratio'] == round(padded / (1040 * 778), 4)
    assert tiling['redundancy'] == round(tiling['clipped_area_ratio'] - 1, 4)


def test_tiling_without_a_padding_backend_reports_no_padded_area():
    profile = pp.PerformanceProfile()
    profile.tiling(height=100, width=100, coords=[(0, 0, 60, 100), (40, 0, 100, 100)],
                   overlap_requested=20, overlap_effective=20, block_effective=60)
    tiling = profile.report('completed')['tiling']
    assert tiling['clipped_area_ratio'] == 1.2
    assert tiling['padded_inference_area_ratio'] is None
    assert tiling['pad_alignment'] is None


def test_summary_reads_a_report():
    profile = pp.PerformanceProfile()
    with profile.stage('render'):
        pass
    text = pp.summary(profile.report('cancelled'))
    assert 'Outcome: cancelled' in text
    assert 'render' in text
    assert pp.summary(None) == ''


def test_summary_ranks_where_the_time_went_by_self_time():
    clock = FakeClock()
    profile = pp.PerformanceProfile(clock=clock, cpu_clock=clock)
    with profile.stage('render'):
        clock.advance(1000)
        with profile.stage('registration'):
            clock.advance(3000)
        with profile.stage('fusion'):
            clock.advance(2000)
            with profile.stage('session.run'):
                clock.advance(94_000)
    text = pp.summary(profile.report('completed'))
    ranking = text.split('Where the time went')[1].split('Stages in execution order')[0]
    rows = [line.split()[-2] for line in ranking.splitlines()[1:] if line.strip()]
    assert rows == ['render/fusion/session.run', 'render/registration', 'render/fusion', 'render']
    assert '94.0%' in ranking and '1 min 34 s' in ranking
    assert 'wall time: 1 min 40 s' in text


def test_summary_tolerates_reports_from_an_older_or_partial_profile():
    # A child whose parent never finished, and stages without the newer fields.
    report = {'outcome': 'failed', 'wall_ms': 10.0, 'stages': {
        'render/fusion': {'count': 1, 'total_ms': 4.0, 'self_ms': 4.0, 'parent': 'render'}},
        'concurrent': {'source_decode': {'count': 2, 'total_ms': 3.0}}}
    text = pp.summary(report)
    assert 'fusion: x1 4.0 ms / 4.0 ms' in text
    assert 'source_decode: x2 3.0 ms' in text

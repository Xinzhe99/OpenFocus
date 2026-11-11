"""Opt-in render performance profile: bounded aggregates, never a render failure.

Standard library only at import. One profile is active per process, like the frame
resolution cache: renders are serialized per process (GUI render child, CLI,
one batch task) and registration's pool threads must see the same profile.

Disabled is the default. ``current()`` then returns a shared null profile whose
``stage()`` returns one shared null context manager: no clock reads and no
allocation. Hot callers may test ``profile.enabled`` instead.

Timing semantics
----------------
``stage(name)`` scopes nest on the thread that activated the profile. Each
stage is aggregated by its path (``fusion/tile/logits.full``): ``total_ms`` is
inclusive wall time, ``self_ms`` excludes time spent in child stages, and
``pct_of_parent`` relates ``total_ms`` to the parent path's ``total_ms``.
Sibling durations never overlap, so they can be added up within one parent.
``order`` is the position at which a path was first entered, so a reader can
show the tree in execution order. Stages entered on other threads are recorded
separately under ``concurrent``: their totals may exceed wall time and are
never folded into the stage tree.

``stage(name, resources=True)`` marks a major stage. It also records process
CPU time (all threads) and how far the stage raised the process's peak
resident memory. Both are process-wide readings, not per-thread ones.
"""
import heapq
import json
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

SCHEMA_VERSION = 1
SLOWEST_TILES = 10
MAX_TILE_COORDS = 256
_NULL_CONTEXT = nullcontext()


class _NullProfile:
    """Shared no-op profile used whenever profiling is off."""

    enabled = False

    def stage(self, name, *, resources=False):
        return _NULL_CONTEXT

    def tile(self, index, coords, frames):
        return _NULL_CONTEXT

    def __getattr__(self, name):
        # Every recording method is a no-op on the null profile.
        return _ignore


def _ignore(*args, **kwargs):
    return None


NULL_PROFILE = _NullProfile()
_active = NULL_PROFILE
_active_lock = threading.RLock()


def current():
    """Return the active profile, or the shared null profile."""
    return _active


@contextmanager
def active(profile):
    """Make ``profile`` the process's active profile for this scope."""
    global _active
    with _active_lock:
        previous = _active
        _active = profile
    if profile.enabled:
        profile._owner = threading.get_ident()
    try:
        yield profile
    finally:
        with _active_lock:
            _active = previous


class _Stage:
    __slots__ = ('profile', 'name', 'resources', 'path', 'started', 'cpu_started',
                 'rss_started', 'child_ns')

    def __init__(self, profile, name, resources):
        self.profile = profile
        self.name = name
        self.resources = resources

    def __enter__(self):
        profile = self.profile
        stack = profile._stack
        self.path = path = f'{stack[-1].path}/{self.name}' if stack else self.name
        if path not in profile._order:
            profile._order[path] = len(profile._order)
        self.child_ns = 0
        self.cpu_started = self.rss_started = None
        if self.resources:
            self.rss_started = profile._peak_rss()
            self.cpu_started = profile._cpu_clock()
        stack.append(self)
        self.started = profile._clock()
        return self

    def __exit__(self, kind, value, traceback):
        profile = self.profile
        try:
            elapsed = profile._clock() - self.started
            stack = profile._stack
            if stack and stack[-1] is self:
                stack.pop()
            if stack:
                stack[-1].child_ns += elapsed
            cpu = rss = None
            if self.resources:
                cpu = profile._cpu_clock() - self.cpu_started
                rss = (self.rss_started, profile._peak_rss())
            profile._record(self.path, self.name, elapsed, self.child_ns, cpu, rss)
        except Exception:
            profile.errors += 1
        return False


class _ConcurrentStage:
    __slots__ = ('profile', 'name', 'started')

    def __init__(self, profile, name):
        self.profile = profile
        self.name = name

    def __enter__(self):
        self.started = self.profile._clock()
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.profile._record_concurrent(self.name, self.profile._clock() - self.started)
        except Exception:
            self.profile.errors += 1
        return False


class _Tile:
    __slots__ = ('profile', 'record', 'stage')

    def __init__(self, profile, record):
        self.profile = profile
        self.record = record
        self.stage = _Stage(profile, 'tile', False)

    def __enter__(self):
        self.profile._tile = self.record
        self.record['sessions_created'] = self.profile._sessions['created']
        self.stage.__enter__()
        return self

    def __exit__(self, kind, value, traceback):
        result = self.stage.__exit__(kind, value, traceback)
        profile = self.profile
        try:
            profile._tile = None
            record = self.record
            record['total_ns'] = profile._clock() - self.stage.started
            record['sessions_created'] = (profile._sessions['created']
                                          - record['sessions_created'])
            if kind is None:
                entry = (record['total_ns'], -record['index'], record)
                if len(profile._slowest) < SLOWEST_TILES:
                    heapq.heappush(profile._slowest, entry)
                elif entry[:2] > profile._slowest[0][:2]:
                    heapq.heapreplace(profile._slowest, entry)
        except Exception:
            profile.errors += 1
        return result


def _ms(ns):
    return round(ns / 1e6, 3)


def _process_peak_rss():
    """Process peak resident bytes, or 0 when the platform cannot say."""
    try:
        from utils.memory import peak_rss_bytes
        return int(peak_rss_bytes())
    except Exception:
        return 0


def _git_revision(root):
    """Commit and dirty flag for a source checkout; None when not cheap/known."""
    if getattr(sys, 'frozen', False) or not os.path.exists(os.path.join(root, '.git')):
        return None, None
    # A windowed Windows process must not flash a console for each git call.
    options = dict(capture_output=True, text=True, timeout=2, check=True,
                   stdin=subprocess.DEVNULL,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        commit = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                                **options).stdout.strip()
        status = subprocess.run(['git', '-C', root, 'status', '--porcelain',
                                 '--untracked-files=no'], **options).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def finished_now():
    """Local finish time of a report, so saved reports can be told apart."""
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def runtime_metadata():
    """Versions and platform facts; reads only modules that are already imported."""
    from constants import app_display_version
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = _git_revision(root)
    versions = {'python': platform.python_version()}
    for name, module in (('numpy', 'numpy'), ('opencv', 'cv2'), ('onnxruntime', 'onnxruntime')):
        loaded = sys.modules.get(module)
        versions[name] = getattr(loaded, '__version__', None) if loaded is not None else None
    return {'openfocus_version': app_display_version(), 'commit': commit, 'dirty': dirty,
            'versions': versions, 'os': platform.platform(terse=True),
            'machine': platform.machine()}


class PerformanceProfile:
    """Aggregate stage timings and counters for one render."""

    enabled = True

    def __init__(self, *, clock=time.perf_counter_ns, cpu_clock=time.process_time_ns,
                 peak_rss=None):
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._peak_rss = peak_rss or _process_peak_rss
        self._order = {}
        self._owner = threading.get_ident()
        self._lock = threading.Lock()
        self._stack = []
        self._stages = {}
        self._concurrent = {}
        self._tile = None
        self._slowest = []
        self._sections = {}
        self._sets = {}
        self._unmeasured = set()
        self._sessions = {'created': 0, 'hits': 0, 'evictions': 0, 'clears': 0,
                          'hash_ns': 0, 'construct_ns': 0}
        self._session_keys = {}
        self._evicted = set()
        self._cleared = set()
        self._cache = {}
        self._cache_degraded = 0
        self._cache_peak = 0
        self.errors = 0
        self.started = clock()

    # ---- stages -----------------------------------------------------------
    def stage(self, name, *, resources=False):
        if threading.get_ident() != self._owner:
            return _ConcurrentStage(self, name)
        return _Stage(self, name, resources)

    def _record(self, path, name, elapsed, child, cpu, rss):
        entry = self._stages.get(path)
        if entry is None:
            parent = path.rpartition('/')[0] or None
            entry = self._stages[path] = {'count': 0, 'total': 0, 'child': 0,
                                          'min': elapsed, 'max': elapsed,
                                          'parent': parent, 'cpu': None,
                                          'rss_peak': None, 'rss_growth': 0}
        entry['count'] += 1
        entry['total'] += elapsed
        entry['child'] += child
        entry['min'] = min(entry['min'], elapsed)
        entry['max'] = max(entry['max'], elapsed)
        if cpu is not None:
            entry['cpu'] = (entry['cpu'] or 0) + cpu
        if rss is not None and rss[1]:
            entry['rss_peak'] = max(entry['rss_peak'] or 0, rss[1])
            entry['rss_growth'] += max(0, rss[1] - rss[0])
        tile = self._tile
        if tile is not None and name != 'tile':
            stages = tile['stages']
            stages[name] = stages.get(name, 0) + elapsed - child

    def _record_concurrent(self, name, elapsed):
        with self._lock:
            entry = self._concurrent.setdefault(name, [0, 0, 0])
            entry[0] += 1
            entry[1] += elapsed
            entry[2] = max(entry[2], elapsed)

    def tile(self, index, coords, frames):
        """Time one tile; the slowest few keep their per-stage breakdown."""
        x0, y0, x1, y1 = coords
        return _Tile(self, {'index': int(index), 'coords': [int(x0), int(y0), int(x1), int(y1)],
                            'height': int(y1 - y0), 'width': int(x1 - x0),
                            'frames': int(frames), 'stages': {}})

    # ---- metadata ---------------------------------------------------------
    def set(self, section, **values):
        with self._lock:
            self._sections.setdefault(section, {}).update(values)

    def add(self, section, key, value):
        """Record a value that may differ between calls (reported as a sorted set)."""
        with self._lock:
            self._sets.setdefault((section, key), set()).add(value)

    def unmeasured(self, note):
        with self._lock:
            self._unmeasured.add(str(note))

    def tiling(self, *, height, width, coords, overlap_requested, overlap_effective,
               pad_alignment=None, **blocks):
        """Tile geometry with provenance; areas are relative to the output.

        ``pad_alignment`` is the multiple a backend pads each tile to before
        inference (neural methods); without it no padded area is reported.
        """
        clipped = padded = 0
        align = int(pad_alignment or 0)
        for x0, y0, x1, y1 in coords:
            h, w = y1 - y0, x1 - x0
            clipped += h * w
            if align:
                padded += (h + (-h) % align) * (w + (-w) % align)
        area = max(1, int(height) * int(width))
        self.set('tiling', count=len(coords),
                 coords=[list(map(int, c)) for c in coords[:MAX_TILE_COORDS]],
                 coords_truncated=len(coords) > MAX_TILE_COORDS,
                 overlap_requested=int(overlap_requested),
                 overlap_effective=int(overlap_effective),
                 pad_alignment=align or None,
                 clipped_area_ratio=round(clipped / area, 4),
                 padded_inference_area_ratio=round(padded / area, 4) if align else None,
                 # Tiles cover the whole output, so the redundant share is the
                 # clipped ratio minus one.
                 redundancy=round(clipped / area - 1, 4),
                 **{key: (int(value) if value is not None else None)
                    for key, value in blocks.items()})

    # ---- neural sessions --------------------------------------------------
    def session_lookup(self, key, hit):
        """Account one session-cache lookup; key is a path-free description."""
        with self._lock:
            entry = self._session_keys.get(key)
            if hit:
                self._sessions['hits'] += 1
                if entry is not None:
                    entry['hits'] += 1
                return
            if key in self._evicted:
                reason = 'evicted'
            elif key in self._cleared:
                reason = 'cleared'
            elif any(other[1] == key[1] for other in self._session_keys):
                reason = 'new_key_same_shape'
            elif self._session_keys:
                reason = 'new_shape'
            else:
                reason = 'first_use'
            self._evicted.discard(key)
            self._cleared.discard(key)
            if entry is None:
                entry = self._session_keys[key] = {'created': 0, 'hits': 0, 'runs': 0,
                                                   'reasons': {}}
            entry['created'] += 1
            entry['reasons'][reason] = entry['reasons'].get(reason, 0) + 1
            self._sessions['created'] += 1

    def session_built(self, hash_ns, construct_ns):
        with self._lock:
            self._sessions['hash_ns'] += hash_ns
            self._sessions['construct_ns'] += construct_ns

    def session_evicted(self, key):
        with self._lock:
            self._sessions['evictions'] += 1
            self._evicted.add(key)

    def sessions_cleared(self, keys):
        with self._lock:
            self._sessions['clears'] += 1
            self._cleared.update(keys)

    def session_run(self, key):
        with self._lock:
            entry = self._session_keys.get(key)
            if entry is not None:
                entry['runs'] += 1

    # ---- resolution cache -------------------------------------------------
    def cache_event(self, kind, event, nbytes=0, elapsed_ns=0):
        with self._lock:
            entry = self._cache.setdefault(kind, {'lookups': 0, 'hits': 0, 'misses': 0,
                                                  'puts': 0, 'bytes_written': 0, 'put_ns': 0})
            if event == 'hit':
                entry['lookups'] += 1
                entry['hits'] += 1
            elif event == 'miss':
                entry['lookups'] += 1
                entry['misses'] += 1
            elif event == 'put':
                entry['puts'] += 1
                entry['bytes_written'] += int(nbytes)
                entry['put_ns'] += int(elapsed_ns)

    def cache_state(self, *, retained_bytes=None, degraded=False):
        with self._lock:
            if retained_bytes is not None:
                self._cache_peak = max(self._cache_peak, int(retained_bytes))
            if degraded:
                self._cache_degraded += 1

    # ---- report -----------------------------------------------------------
    def report(self, outcome):
        """Deterministic JSON-ready report of everything recorded so far."""
        wall = self._clock() - self.started
        with self._lock:
            stages = {}
            for path, entry in self._stages.items():
                parent = self._stages.get(entry['parent']) if entry['parent'] else None
                item = {'count': entry['count'], 'total_ms': _ms(entry['total']),
                        'self_ms': _ms(entry['total'] - entry['child']),
                        'mean_ms': _ms(entry['total'] / entry['count']),
                        'min_ms': _ms(entry['min']), 'max_ms': _ms(entry['max']),
                        'order': self._order.get(path, len(self._order)),
                        'parent': entry['parent'],
                        'pct_of_parent': (round(100 * entry['total'] / parent['total'], 2)
                                          if parent and parent['total'] else None)}
                if entry['cpu'] is not None:
                    item['process_cpu_ms'] = _ms(entry['cpu'])
                if entry['rss_peak'] is not None:
                    item['peak_rss_bytes'] = entry['rss_peak']
                    item['peak_rss_growth_bytes'] = entry['rss_growth']
                stages[path] = item
            sections = json.loads(json.dumps(self._sections, default=str))
            for (section, key), values in self._sets.items():
                sections.setdefault(section, {})[key] = sorted(values, key=str)
            sessions = {(key[:-3] + '_ms' if key.endswith('_ns') else key):
                        (_ms(value) if key.endswith('_ns') else value)
                        for key, value in self._sessions.items()}
            sessions['keys'] = [
                {'model': key[0], 'shape': list(key[1]), 'device': key[2], 'threads': key[3],
                 **{name: value for name, value in entry.items() if name != 'reasons'},
                 'miss_reasons': dict(sorted(entry['reasons'].items()))}
                for key, entry in sorted(self._session_keys.items(), key=lambda item: str(item[0]))]
            cache = {kind: {**{k: v for k, v in entry.items() if k != 'put_ns'},
                            'put_ms': _ms(entry['put_ns'])}
                     for kind, entry in sorted(self._cache.items())}
            slowest = [{**{k: v for k, v in record.items() if k not in ('stages', 'total_ns')},
                        'total_ms': _ms(record['total_ns']),
                        'stages_self_ms': {name: _ms(ns) for name, ns in sorted(record['stages'].items())}}
                       for _, _, record in sorted(self._slowest, reverse=True)]
            report = {
                'schema_version': SCHEMA_VERSION,
                'outcome': str(outcome),
                'wall_ms': _ms(wall),
                'stages': dict(sorted(stages.items())),
                'concurrent': {name: {'count': count, 'total_ms': _ms(total),
                                      'mean_ms': _ms(total / count), 'max_ms': _ms(longest)}
                               for name, (count, total, longest)
                               in sorted(self._concurrent.items())},
                'sessions': sessions,
                'cache': {'by_kind': cache, 'degraded_events': self._cache_degraded,
                          'peak_retained_bytes': self._cache_peak},
                'slowest_tiles': slowest,
                'unmeasured': sorted(self._unmeasured),
                'profiler_errors': self.errors,
                **sections,
            }
        report.setdefault('render', {})['peak_rss_bytes'] = self._peak_rss() or None
        return report


def compact(report):
    """The report without tile coordinates: the form recorded as one log event."""
    if 'tiling' not in report:
        return dict(report)
    return {**report, 'tiling': {key: value for key, value in report['tiling'].items()
                                 if key != 'coords'}}


def dumps(report):
    """Stable serialization: sorted keys, fixed schema."""
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False)


def _duration(ms):
    """Compact duration for reading: milliseconds up to minutes."""
    ms = float(ms or 0)
    if ms < 1000:
        return f'{ms:.1f} ms'
    seconds = ms / 1000
    if seconds < 60:
        return f'{seconds:.2f} s'
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f'{minutes} min {seconds:02d} s'


def _size(value):
    value = float(value or 0)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024 or unit == 'GiB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.2f} {unit}'
        value /= 1024


def _fields(section, skip=()):
    return ', '.join(f'{key}={value}' for key, value in sorted(section.items())
                     if key not in skip and value is not None)


def _stage_tree(stages):
    """Stage paths, parents before children and siblings in execution order."""
    children = {}
    for path, item in stages.items():
        children.setdefault(item.get('parent'), []).append(path)
    ordered = []

    def visit(parent):
        for path in sorted(children.get(parent, ()),
                           key=lambda name: (stages[name].get('order', 0), name)):
            ordered.append(path)
            visit(path)

    visit(None)
    # A partial report can hold a child whose parent never finished.
    ordered.extend(sorted(set(stages) - set(ordered)))
    return ordered


def summary(report, *, top=8):
    """Plain-text reading of a report for the GUI viewer and CLI stderr."""
    if not report:
        return ''
    wall = float(report.get('wall_ms') or 0)
    render = report.get('render', {})
    head = f"Outcome: {report.get('outcome')}   wall time: {_duration(wall)}"
    if report.get('finished_at'):
        head = f"Finished: {report['finished_at']}   " + head
    if render.get('peak_rss_bytes'):
        head += f"   peak memory: {_size(render['peak_rss_bytes'])}"
    lines = [head]
    if render:
        lines.append('Render: ' + _fields(render, skip=('peak_rss_bytes',)))
    threads = report.get('threads', {})
    if threads:
        lines.append('Threads (0 = automatic): ' + _fields(threads))
    tiling = report.get('tiling', {})
    if tiling:
        lines.append('Tiling: ' + _fields(tiling, skip=('coords',)))
    sessions = report.get('sessions', {})
    if sessions.get('created') or sessions.get('hits'):
        lines.append('Sessions: ' + _fields(sessions, skip=('keys',)))
        for key in sessions.get('keys', []):
            lines.append(f"  {key['shape']} threads={key['threads']} created={key['created']} "
                         f"hits={key['hits']} runs={key['runs']} reasons={key['miss_reasons']}")
    stages = report.get('stages', {})
    ranked = sorted(stages.items(), key=lambda item: -item[1].get('self_ms', 0))[:top]
    if ranked and wall:
        lines += ['', 'Where the time went (self time, share of wall time):']
        for path, item in ranked:
            if item.get('self_ms', 0) <= 0:
                break
            lines.append(f"  {100 * item['self_ms'] / wall:5.1f}%  {_duration(item['self_ms']):>12}"
                         f"  {path}  x{item['count']}")
    lines += ['', 'Stages in execution order (total / self):']
    for path in _stage_tree(stages):
        item = stages[path]
        extra = ''
        if item.get('pct_of_parent') is not None:
            extra += f"  {item['pct_of_parent']:.1f}% of parent"
        if 'process_cpu_ms' in item:
            extra += f"  cpu {_duration(item['process_cpu_ms'])}"
        if item.get('peak_rss_growth_bytes'):
            extra += f"  peak memory +{_size(item['peak_rss_growth_bytes'])}"
        lines.append(f"{'  ' * path.count('/')}{path.rpartition('/')[2]}: x{item['count']} "
                     f"{_duration(item['total_ms'])} / {_duration(item['self_ms'])}{extra}")
    if report.get('concurrent'):
        lines += ['', 'On other threads (summed across threads, may exceed wall time):']
        for name, item in report['concurrent'].items():
            longest = f", longest {_duration(item['max_ms'])}" if 'max_ms' in item else ''
            lines.append(f"  {name}: x{item['count']} {_duration(item['total_ms'])}{longest}")
    cache = report.get('cache', {})
    if cache.get('by_kind'):
        lines += ['', 'Resolution cache by entry kind:']
        for kind, item in cache['by_kind'].items():
            lines.append(f"  {kind}: " + _fields(item))
        lines.append(f"  degraded_events={cache.get('degraded_events')} "
                     f"peak_retained={_size(cache.get('peak_retained_bytes'))}")
    if report.get('slowest_tiles'):
        lines += ['', 'Slowest tiles:']
        for tile in report['slowest_tiles']:
            parts = sorted(tile['stages_self_ms'].items(), key=lambda item: -item[1])[:5]
            lines.append(f"  #{tile['index']} {tile['width']}x{tile['height']} "
                         f"{_duration(tile['total_ms'])} sessions_created={tile['sessions_created']} "
                         + ', '.join(f'{name}={_duration(ms)}' for name, ms in parts))
    if report.get('unmeasured'):
        lines += ['', 'Not measured: ' + '; '.join(report['unmeasured'])]
    if report.get('profiler_errors'):
        lines.append(f"Profiler bookkeeping errors: {report['profiler_errors']}")
    return '\n'.join(lines)

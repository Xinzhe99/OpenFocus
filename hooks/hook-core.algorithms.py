"""Collect every lazy algorithm entry point without importing its backend."""
from core.algorithms import fusion_algorithms, registration_algorithms


_entry_points = []
for _algorithm in fusion_algorithms():
    _entry_points.extend(
        getattr(_algorithm, field)
        for field in (
            "runner", "environment", "resource_estimator", "roi", "tile_decider",
            "tile_runner", "depth_planner",
        )
    )
    if _algorithm.depth is not None:
        _entry_points.append(_algorithm.depth.runner)
_entry_points.extend(algorithm.runner for algorithm in registration_algorithms())

hiddenimports = sorted({entry.partition(":")[0] for entry in _entry_points if entry})

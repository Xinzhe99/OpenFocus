"""Structured fusion diagnostics shared by desktop and headless processing."""
from core.algorithms import get_fusion
from utils.logging_setup import log_action


def log_effective_fusion(fusion, images, **context):
    """Record effective settings and ordered source headers without paths or pixels."""
    keys = ('algorithm', 'fusion_parameters', 'use_gpu', 'resource_preference',
            'tile_enabled', 'tile_block_size', 'tile_overlap', 'tile_threshold',
            'ai_batch_size', 'base_frame', 'sort_mode', 'exposure_match')
    settings = {key: getattr(fusion, key, None) for key in keys}
    sources = [{'index': index, 'shape': tuple(frame.shape) if getattr(frame, 'shape', None) is not None else None,
                'dtype': str(frame.dtype) if getattr(frame, 'dtype', None) is not None else None}
               for index, frame in enumerate(images)]
    algorithm = settings['algorithm']
    order = list(range(len(sources)))
    if algorithm and get_fusion(algorithm).sort_frames and settings['sort_mode'] == 'desc':
        order.reverse()
    reference = None
    if order and settings['exposure_match']:
        reference = order[0] if algorithm in ('openfocus_a', 'openfocus_b', 'openfocus_c') else order[-1]
    log_action('fusion.effective', settings=settings, sources=sources, fusion_order=order,
               exposure_reference_index=reference, **context)

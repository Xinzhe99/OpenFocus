"""Score-workspace limits shared by the StackMFF worker and desktop planner."""

# Apply this limit independently to spatial and cross-layer attention. Allow
# four fp32 score-sized buffers for scores, probabilities, masks and temporaries.
# QKV, features and outputs scale linearly and are budgeted separately.
SCORE_ELEMENTS = 4 * 1024**2
SCORE_WORKSPACE_BYTES_PER_ELEMENT = 4 * 4
FOCUS_QUERY_LIMIT = 128


def released_attention_workspace_bytes(layers, padded_height, padded_width):
    """Batch-one released checkpoints: 3 heads, window 8, inter-layer pool 2."""
    pixels = padded_height * padded_width
    spatial_scores = (pixels // 16) * 3 * layers * 64
    focus_scores = (pixels // 64) * 3 * layers * layers
    return min(SCORE_ELEMENTS, max(spatial_scores, focus_scores)) * SCORE_WORKSPACE_BYTES_PER_ELEMENT

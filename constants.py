APP_VERSION = "1.70.0"

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 950

TILE_BLOCK_SIZE = 1024
TILE_OVERLAP = 256
TILE_THRESHOLD = 2048
TILE_BLOCK_MIN = 64
TILE_BLOCK_MAX = 16384
TILE_OVERLAP_MAX = 4096
TILE_THRESHOLD_MIN = 256
TILE_THRESHOLD_MAX = 131072

REG_DOWNSCALE_WIDTH = 1024
HOMOGRAPHY_DOWNSCALE_WIDTH = 1600
ECC_DOWNSCALE_WIDTH = 1000

STATUS_UPDATE_INTERVAL_MS = 2000

BYTES_TO_GB = 1024 ** 3

FEATURE_MAX_KEYPOINTS = 200
FEATURE_MATCH_THRESHOLD = 0.01
FEATURE_RANSAC_REJECTS = 30
FEATURE_RANSAC_MAX_ITER = 3

MAGNIFIER_SIZE = 320
ANIMATION_DURATION_MS = 400

DEFAULT_THREAD_COUNT = 4
MIN_THREAD_COUNT = 1
MAX_THREAD_COUNT = 16

AI_BATCH_SIZE = 2
AI_BATCH_SIZE_MIN = 1
AI_BATCH_SIZE_MAX = 16

KERNEL_SIZE_MIN = 3
KERNEL_SIZE_MAX = 51
KERNEL_SIZE_DEFAULT_DCT = 7
KERNEL_SIZE_DEFAULT_GFF = 31
KERNEL_SIZE_DEFAULT_GFG = 7

# One slider means a different parameter per method family, so each family
# remembers its own value. mode -> (maximum, default); the GUI, the settings
# codec and project validation all read this table instead of keeping copies.
KERNEL_MODES = {
    "guided": (51, 31),
    "dct": (51, 7),
    "gfg": (51, 7),
    # OpenFocus A/B/C share the variance window (kernel = 2r+1); the
    # discriminative probe put r=5 -> 11 with a 1-21 range.
    "ofab": (21, 11),
}
METHOD_KERNEL_MODES = {
    "guided_filter": "guided",
    "dct": "dct",
    "gfgfgf": "gfg",
    "openfocus_a": "ofab",
    "openfocus_b": "ofab",
    "openfocus_c": "ofab",
}
SMOOTHING_MAX = 8.0

# Single source of truth for the parameters a fusion method actually consumes.
# Every algorithm-selecting surface (main window, batch dialog, compare dialog)
# reads this table, so a method can never expose a control its back-end ignores
# or hide one it needs. key -> (label_key, minimum, maximum, step, decimals,
# default). StackMFF V3 has a categorical output mode instead of a number and is
# listed separately.
METHOD_FUSION_PARAMS = {
    "guided_filter": (("kernel_size", "label_kernel_guided", 1, 51, 2, 0, 31),),
    "dct": (("kernel_size", "label_kernel_dct", 1, 51, 2, 0, 7),),
    "dtcwt": (("N", "compare_levels", 1, 6, 1, 0, 4),),
    "gfgfgf": (("kernel_size", "label_kernel_gfg", 1, 51, 2, 0, 7),),
    "stackmffv3": (),
    "openfocus_a": (
        ("kernel_size", "label_kernel_of", 1, 21, 2, 0, 11),
        ("blend_smoothing", "label_blend_smoothing", 0, 8, .2, 1, 2.0),
    ),
    "openfocus_b": (
        ("kernel_size", "label_kernel_of", 1, 21, 2, 0, 11),
        ("depth_smoothing", "label_depth_smoothing", 0, 8, .2, 1, 1.2),
    ),
    # OpenFocus C has no radius: its only parameter is the blending smoothing.
    "openfocus_c": (
        ("blend_smoothing", "label_blend_smoothing", 0, 8, .2, 1, 3.0),
    ),
}

# Methods whose output mode is a soft/hard choice rather than a number.
METHOD_OUTPUT_MODES = ("stackmffv3",)
AI_MODES = ("soft", "hard")

# Parameters the depth estimator actually consumes, per measure. Smoothing is
# a display setting there, so it is deliberately absent; DTCWT does use its
# decomposition depth, and AI uses its output mode.
METHOD_DEPTH_PARAMS = {
    "dct": ("kernel_size",),
    "gfgfgf": ("kernel_size",),
    "openfocus_b": ("kernel_size",),
    "openfocus_c": ("kernel_size",),
    "dtcwt": ("N",),
    "stackmffv3": ("mode",),
}

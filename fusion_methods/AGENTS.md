# FUSION METHODS

**Pattern:** Algorithm Implementations (5 algorithms)

## OVERVIEW
5 multi-focus fusion algorithms. All follow `fusion(input_images, options) -> fused` signature.

## ALGORITHMS
| File | Method | GPU | Notes |
|------|--------|-----|-------|
| `gff.py` | Guided Filter Fusion | ❌ CPU | Edge-preserving, fast |
| `dct.py` | DCT Multi-Focus | ❌ CPU | Frequency-domain, no dynamic resize |
| `dtcwt.py` | Dual-Tree Complex Wavelet | ❌ CPU | Texture preservation |
| `gfg_fgf.py` | GFG-FGF Gradient | ❌ CPU | Generalized 4-neighborhood Gaussian |
| `stackmffv4.py` | Neural Network (PyTorch) | ✅ CPU/GPU | Pretrained model (`weights/stackmffv4.pth`) |

## WHERE TO LOOK
| Task | File |
|------|------|
| Add new fusion method | Copy `gff.py` → implement `fuse()` → export in `__init__.py` |
| GPU handling | `stackmffv4.py:check_cuda_available()` |
| Memory optimization | `gff.py:222` (use `as_completed`, don't hold all results) |

## ANTI-PATTERNS (THIS DIR)
- **NEVER** use GPU with DCT/DTCWT/Guided Filter — CPU only
- **NEVER** resize images after DCT fusion starts
- **NEVER** hold all tile results in memory simultaneously

## CONVENTIONS
- `def fuse(images: List[np.ndarray], options: FusionOptions) -> np.ndarray`
- Handle `options.tile_enabled` for large images
- Call `normalize_kernel_size()` for kernel validation

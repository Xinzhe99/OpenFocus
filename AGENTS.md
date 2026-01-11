# OPENFOCUS KNOWLEDGE BASE

**Generated:** 2026-01-11 22:25
**Type:** PyQt6 Desktop Application (Python 3.10+)

## OVERVIEW
PyQt6-based multi-focus image fusion workstation. Registers and blends focus-stacked images using 5 algorithms (Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF V4 neural model).

## STRUCTURE
```
OpenFocus/
├── main.py                    # Entry point (OpenFocus QMainWindow)
├── config.py                  # Dataclasses: FusionMethod, ROIMode, Options
├── constants.py               # SCREAMING_SNAKE constants
├── styles.py                  # QSS dark theme
├── locales.py                 # i18n (en/zh, timezone auto-detect)
├── controllers/               # 7 managers (MVC pattern)
│   ├── source_manager.py      # Image loading, drag-drop
│   ├── render_manager.py      # Fusion orchestration
│   ├── output_manager.py      # Result handling
│   ├── export_manager.py      # GIF/PNG/BMP/TIFF export
│   ├── label_manager.py       # Annotation labels
│   ├── transform_manager.py   # Image transformations
│   └── batch_manager.py       # Batch processing
├── dialogs/                   # Refactored dialogs (modularized from monolithic dialogs.py)
│   ├── __init__.py            # Unified exports
│   ├── about.py               # EnvironmentInfoDialog, ContactInfoDialog
│   ├── help.py                # HelpDialog, RenderMethodHelpDialog, RegistrationHelpDialog, TileHelpDialog
│   ├── settings.py            # DurationDialog, DownsampleDialog, TileSettingsDialog, RegistrationSettingsDialog, ThreadSettingsDialog
│   ├── batch.py               # BatchProcessingDialog, FolderImportDialog
│   └── roi.py                 # ROIRenderOptionsDialog
├── fusion_methods/            # Algorithm implementations
│   ├── gff.py                 # Guided Filter Fusion
│   ├── dct.py                 # DCT Multi-Focus Fusion
│   ├── dtcwt.py               # Dual-Tree Complex Wavelet
│   ├── gfg_fgf.py             # Gradient-based Fusion
│   └── stackmffv4.py          # PyTorch neural model
├── ui/                        # UI components
│   ├── image_panels.py        # Source/Result display
│   ├── menus.py               # Menu bar setup
│   └── right_panel.py         # Control panel
├── widgets/
│   └── magnifier_label.py     # Zoom magnifier
└── assets/, weights/, docs/
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Add fusion algorithm | `fusion_methods/` (follow `gff.py` pattern) |
| UI changes | `ui/` or `styles.py` (QSS) |
| Image processing | `workers.py`, `multi_focus_fusion.py`, `Registration.py` |
| Configuration | `config.py` (dataclasses), `constants.py` (values) |
| i18n | `locales.py` (translations dict) |
| Batch processing | `controllers/batch_manager.py` |
| Dialogs | `dialogs/` (modularized) |

## ANTI-PATTERNS (THIS PROJECT)

### NEVER DO
1. **Emit `roiDeleted` during ROI creation** (`main.py:341`) — toggles button off unexpectedly
2. **Use GPU acceleration with DCT/DTCWT/Guided Filter** — CPU only
3. **Multi-thread with CuPy** — causes PCIe contention (`Registration.py:712`)
4. **Resize images after DCT fusion starts** — DCT doesn't support dynamic resize
5. **Hold all tile results in memory** — use `as_completed` pattern
6. **Add new dialogs to monolithic files** — Use `dialogs/` module structure

### ALWAYS
1. **Validate crop regions** before transformations (`Registration.py:403`)
2. **Ensure kernel_size is odd** (`workers.py:429`)
3. **Use tile mode for images > 2048px** — auto-switches (`constants.py: TILE_THRESHOLD`)
4. **Validate thread_count** — `max(1, int(thread_count))` (`workers.py:76`)
5. **New dialogs** go in `dialogs/` subdirectory (see existing patterns)

### PERFORMANCE NOTES
- DCT/DTCWT/Guided Filter: CPU only, no GPU
- StackMFF V4: Force serial to avoid GPU OOM (`workers.py:606`)
- Large images: Tiled fusion auto-enabled (`multi_focus_fusion.py:253`)

## CONVENTIONS

| Pattern | Example |
|---------|---------|
| Classes | `PascalCase` (`OpenFocus`, `FusionMethod`) |
| Constants | `SCREAMING_SNAKE_CASE` (`WINDOW_WIDTH`) |
| Functions/vars | `snake_case` (`show_message_box`) |
| Config | Dataclasses with type hints (`config.py`) |
| i18n | `trans.t("key")` function call (`locales.py`) |

## CODE CONCENTRATION

| File | Lines | Purpose |
|------|-------|---------|
| `Registration.py` | 1093 | Registration algorithms |
| `multi_focus_fusion.py` | 778 | Fusion orchestrator |
| `workers.py` | 737 | Background thread workers |
| `network.py` | 630 | Network utilities |
| `locales.py` | 563 | i18n translations |
| `styles.py` | 422 | QSS stylesheets |
| `dialogs/` | ~75/total | ✅ REFACTORED — Split into modular files |
| `dialogs/batch.py` | 891 | BatchProcessingDialog |
| `dialogs/settings.py` | 643 | Settings dialogs |

## COMMANDS
```bash
# Development
python main.py

# Environment (conda)
conda create -n openfocus python=3.10
conda activate openfocus
pip install opencv-python pyqt6 numpy imageio dtcwt scipy torch torchvision

# Build (Windows)
# See docs/BUILD_COMMANDS.md
```

## NOTES
- **No test infrastructure** — zero test files, pytest not configured
- **No pyproject.toml** — no modern Python packaging
- **i18n auto-detects** timezone UTC+8 → Chinese, else English
- **Empty `dialogs/` dir** — `dialogs.py` should be refactored there

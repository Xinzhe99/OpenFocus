# CONTROLLERS

**Pattern:** MVC Manager Classes

## OVERVIEW
7 manager classes handling business logic. Instantiated in `main.py` `OpenFocus.__init__()`.

## STRUCTURE
```
controllers/
├── source_manager.py    # Image I/O, drag-drop, stack management
├── render_manager.py    # Fusion pipeline orchestration
├── output_manager.py    # Result display, history
├── export_manager.py    # GIF/PNG/BMP/TIFF + metadata
├── label_manager.py     # ROI labels, annotations
├── transform_manager.py # Crop, resize, rotation
└── batch_manager.py     # Multi-folder batch jobs
```

## WHERE TO LOOK
| Manager | Responsibility |
|---------|----------------|
| `source_manager` | `load_images()`, `clear_images()`, drag-drop handlers |
| `render_manager` | `start_render()` → workers → result callback |
| `export_manager` | `export_gif()`, `export_images()`, metadata handling |
| `batch_manager` | `run_batch()`, progress signals |

## CONVENTIONS
- All managers receive `main_window` reference in `__init__`
- Use `main_window.render_manager.start_render(...)` for fusion
- Emit signals for async progress updates
- Access images via `main_window.stack_images`, `main_window.raw_images`

## ANTI-PATTERNS (THIS DIR)
- **Don't access UI directly** — use controllers for logic, ui/ for display
- **Don't emit `roiDeleted` during ROI creation** (`main.py:341`)

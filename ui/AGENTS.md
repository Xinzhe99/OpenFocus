# UI

**Pattern:** PyQt6 Component Separation

## OVERVIEW
UI components separate from business logic (controllers) and algorithms (fusion_methods).

## STRUCTURE
```
ui/
├── image_panels.py     # Source/Result display panels
├── menus.py            # Menu bar setup (File, Edit, View, etc.)
└── right_panel.py      # Control panel (fusion settings, ROI tools)
```

## WHERE TO LOOK
| Component | Purpose |
|-----------|---------|
| `image_panels.py` | `create_source_panel()`, `create_result_panel()`, magnifier integration |
| `menus.py` | `setup_menus()` → menu bar with actions |
| `right_panel.py` | `create_right_panel()`, `bind_right_panel()` → settings controls |

## STYLING
- All styles defined in `styles.py` (QSS strings)
- Dark theme: `GLOBAL_DARK_STYLE`
- Primary blue: `#0033A0`

## ANTI-PATTERNS (THIS DIR)
- **NEVER** put business logic in UI files — delegate to controllers
- **NEVER** call `update()` after `setPixmap()` — double repaint (`main.py:819`)

## CONVENTIONS
- Use `trans.t("key")` for all user-facing strings (i18n)
- Follow existing `QSS_*` patterns from `styles.py`
- Connect signals to controller methods, not UI logic

# OpenFocus User Manual

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Main Interface Overview](#3-main-interface-overview)
4. [Importing Images and Video](#4-importing-images-and-video)
5. [Viewing and Navigating Images](#5-viewing-and-navigating-images)
6. [Wipe Compare View](#6-wipe-compare-view)
7. [Image Registration](#7-image-registration)
8. [Image Fusion Methods](#8-image-fusion-methods)
9. [Image Transformations](#9-image-transformations)
10. [Retouching, Labels and Annotations](#10-retouching-labels-and-annotations)
11. [Settings and Configuration](#11-settings-and-configuration)
12. [Batch Processing](#12-batch-processing)
13. [Exporting Results](#13-exporting-results)
14. [Command Line Interface](#14-command-line-interface)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Introduction

OpenFocus is a professional multi-focus image fusion desktop application designed for researchers, engineers, and image processing professionals. The software processes multi-focus image sequences using advanced registration and fusion algorithms to generate all-in-focus images with full depth of field.

### Key Features

- **Multi-Focus Image Fusion**: Combine multiple images with different focus points into one fully focused image
- **Multiple Fusion Algorithms**: Choose from Guided Filter, DCT, DTCWT, GFG-FGF, and AI Fusion (deep learning)
- **Image Registration**: Align misaligned image sequences using ECC or Homography methods
- **Wipe Compare View**: Inspect alignment and compare results with an A/B divider view
- **Project Files**: restore input references and reproducible settings; export result pixels separately
- **Multi-page TIFF**: single-file Z-stacks load page-by-page and export back into one file
- **Depth Map**: per-pixel focus-position maps with custom pseudo-color schemes
- **Scale Bar**: publication-grade scale bars with automatic µm/px calibration
- **Crash Auto-Recovery**: the last working session is offered for restore after a crash
- **Multi-language**: twelve UI languages — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — with automatic system-language detection
- **Headless CLI**: Script OpenFocus from the command line for automation and CI
- **Flexible Export**: Save results as individual images, folders, or GIF animations
- **Image Transformations**: Rotate, flip, and resize image stacks
- **Label Addition**: Add text labels to registered and input stacks
- **Persistent Settings**: Your preferences, language, and recently opened stacks survive restarts

### Supported Input Formats

- Image files: JPG/JPEG, PNG, BMP, TIFF, WebP, and other common formats
- Video files: MP4 format (automatically decoded into image sequences)
- Folders containing numbered image sequences
- Camera RAW files (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 and other common
  RAW formats). Decoding uses `rawpy` (LibRaw), which `pip install openfocus`
  installs and the Windows/macOS builds bundle

### Supported Output Formats

- Individual images: JPG, PNG, BMP, TIFF, WebP
- Image sequences: Saved as folders
- Animations: GIF format

---

## 2. Getting Started

### Starting the Application

After installation, launch OpenFocus by:

1. **Windows (Executable)**: Double-click the OpenFocus shortcut or executable file
2. **Python Source**: Run `python main.py` in your activated conda environment

Upon launch, you will see the main application window with a dark theme interface.

### Quick Start Workflow

1. Import your image sequence (File → Open Folder or drag-and-drop)
2. Configure fusion settings in the right panel
3. Optionally enable registration for alignment
4. Adjust kernel size if needed
5. *(optional)* Check **Quick preview** to test parameters on a downscaled draft first
6. Click "Start Render" to begin processing
7. Export your results

---

## 3. Main Interface Overview

The OpenFocus interface consists of three main areas:

### Left Panel: Image Display Area

The left portion of the window displays your images in a split view:

- **Source Image Panel (Top)**: Shows the input image sequence
- **Result Image Panel (Bottom)**: Shows the fusion or registration results
- **Navigation Slider**: Use the slider below each panel to scroll through image frames
- **Image Info Labels**: Display current frame position and image dimensions

You can resize the panels by dragging the splitter handle between them.

### Right Panel: Control Panel

The right panel contains all configuration options organized vertically:

1. **Fusion Settings**: Select fusion algorithm and kernel size
2. **Registration Settings**: Enable/disable alignment methods
3. **Action Buttons**: Reset defaults and Start Render
4. **Source Images List**: Shows all loaded images with filenames
5. **Output List**: Shows generated results

### Menu Bar

The top menu bar provides access to all functions:

- **File**: Import/export operations and the recent-files list
- **Edit**: Image transformations and labels
- **Tools**: Batch processing, depth map
- **Settings**: GPU acceleration, language, tile, registration, and thread configurations
- **Help**: Environment info, contact information, update check, and the log folder

---

## 4. Importing Images and Video

### Opening an Image Folder

1. Go to **File → Open Folder** or press `Ctrl+O`
2. Select a folder containing your image sequence
3. A loading-scale dialog appears — choose the scale (100% = full resolution)
4. Click OK to load the images

Tick **Don't ask again** in that dialog to always load at the chosen scale
without a prompt; **Settings → Ask for Loading Scale on Import** turns the
prompt back on. A loaded stack can still be downscaled later with
**Edit → Resize**.

The software will automatically:
- Detect and sort image files by filename
- Display the first image in the source panel
- Populate the source images list with all frames

### Opening Video Files

1. Go to **File → Open Video** or press `Ctrl+Shift+O`
2. Select a video file (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. The video will be automatically decoded into individual frames
4. Each frame becomes part of the image stack

### Project Files

File → Save Project stores input references and reproducible settings in an
`.ofproj`: exact frame order (including TIFF pages and duplicate frames),
exclusions, rotations/flips, load scale, per-method rendering parameters,
labels, scale-bar style and current frame. Result and retouch pixels must be
exported separately. Operations that cannot be reconstructed from the source
files require saving a new input stack first.

File → Open Project rebuilds the recorded frames before replacing the current
stack. If sources moved, choose the new source root and confirm the complete
old/new path mapping. Save the project again to retain the relocated references.

### Demo Stack

The welcome dialog offers **Load Demo Stack**: six bundled sample photos
from a real focus stack, so you can try the full workflow immediately
without your own images. Also handy for verifying an installation.

### Sharpness Curve

A small curve above the source navigation slider shows the per-frame
sharpness (normalized). Out-of-focus frames appear as dips marked with a
red dot. Click anywhere on the curve to jump to that frame; hovering
shows the frame number.

### Multi-page TIFF Stacks

Microscopy software (ZEN, NIS-Elements, ImageJ/Fiji) often exports a whole
Z-stack as ONE multi-page TIFF file. OpenFocus reads every page of such a
file as an individual frame of the image stack — drop or open the file and
the stack appears complete (frames are named `<file>_page_001` and so on).
Single-page TIFFs behave exactly as before. The reverse works too:
**File → Save Stack → (Registered / Input) → Save as Multi-page TIFF**
writes the entire stack back into a single `.tif`, preserving 16-bit depth.

### Scale Bar (Publication Figures)

Tools → Scale Bar… configures a scale bar burned into exported images.
The pixel size (µm/px) is auto-detected from TIFF metadata — ImageJ's
`unit=micron` convention, OME-TIFF's `PhysicalSizeX`, or an explicit
centimetre resolution unit — so microscopy stacks (including multi-page
TIFFs) calibrate themselves. Printer resolution (72/300 dpi and friends)
is deliberately *not* treated as calibration, and a manual value you type
always wins over detected metadata: a plausible-looking wrong bar in a
figure is worse than none. Choose the corner
and color (Auto inverts against the background); the dialog shows a live
preview. With the bar enabled, every export (single image, folder stacks,
multi-page TIFF, GIF, export-all, batch) carries a publication-style bar:
a rounded 1-2-5 length (~12% of the image width) with stroked, readable
text. The CLI supports it via `--scale-bar` (auto calibration) and
`--scale-um <um/px>`.

The preview draws at the target's native pixel geometry before fitting it to
the viewport. Fit, 100%, pan and frame navigation change only the view. Apply
commits one undoable edit; Cancel preserves the previous calibration. Historical
outputs retain their own calibration and style when another source stack loads.

Remove it again from **Tools → Scale Bar → Remove scale bar**: *From the
source stack* and *From the result* narrow the dialog's "Apply to" scope to
the remaining canvas, while *From everywhere* turns the bar off. Both
canvases and the A/B wipe redraw immediately; removing a canvas that never
showed the bar leaves the setting unchanged.

### Depth Map (Focus Position)

**Tools → Depth Map…** estimates focus positions using the selected method's
activity measure. This is not a measured surface or an exact pixel-source map:
pyramid/wavelet methods estimate activity and AI reports an expected frame
position. Choose either the current checked frames or the selected output's
recorded source. Missing historical provenance is never replaced with the
current stack. The source and texture stay bound to that snapshot.

All six methods — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C and StackMFF V3 —
require an explicit **Compute** click. Opening the dialog or changing the measure,
source, registration or parameters never starts computation. The parameter editor
appears directly below the measure selector: DCT median kernel and GFG initial kernel
use odd values 1–51 (default 7); OpenFocus B uses a variance window and OpenFocus C a
band-pass energy window, both odd 1–21 px (default 11). DTCWT has no level control
here: it follows the main window's decomposition depth, as in a render. StackMFF V3
shows its own AI output mode (Soft blend by default, or Hard pick) and an explanation
instead of a numeric kernel: the map reports a continuous frame position, not a
physical distance. Soft uses the probability-weighted position; Hard uses the most
probable frame per pixel.
Drafts start from the matching output's parameters or the main window's remembered
method settings. They are remembered per source and method while this dialog is
open; editing them does not change the main window, project or recorded output.

Changing calculation settings marks the previous 2D map as **not current**, clears
the 3D surface and blocks every export, including colorized, raw, physical and OBJ,
until a new computation succeeds. An active calculation is cancelled cooperatively;
after cancellation finishes, press **Compute** again. No next calculation is queued.
Compute stays visible and is disabled while computing, cancelling or exporting.

- **Color map**: grayscale (publication standard) plus 15 scientific
  palettes (Turbo, Viridis, Magma, …) and
  your own — *New Custom Scheme…* builds a gradient from any number of
  color stops (double-click a stop to edit position and color); custom
  schemes are saved and offered in the dropdown.
- **Invert / Gamma / Smoothing**: flip the depth axis, re-balance tone
  (gamma < 1 brightens low indices), and remove speckle with an
  edge-aware pass.
- **Overlay**: available only for the associated output with matching coordinates.
- **Raw export**: 16-bit TIFF stores a normalized index encoded from 0–65535;
  it is not lossless float data or a physical distance. JPEG/BMP are refused.
- **Physical depth**: enter Z start, positive step, unit and acquisition direction,
  then confirm the original capture mapping. Excluded frames keep their gaps.
  Float32 TIFF stores micrometres, NaN for missing data, and source metadata.
  A physical OBJ also needs XY calibration; it uses a 1:1 physical Z scale.
- **3D model / 3D texture**: choose the grid's longest side (600 vertices by default).
  Each newly opened dialog defaults to **Automatic (true color)**: use the associated
  output only when its identity and coordinates/ROI match the completed map;
  otherwise use the first participating frame from that completed calculation,
  after registration and any frame reordering. The resolved output or frame name
  is shown. You can explicitly choose the matching output, a participating frame,
  or **Depth pseudocolor**; unavailable images are reported, never replaced with
  unrelated outputs. This texture is shared by the 3D preview and the OBJ texture
  PNG. The 2D map keeps its own color map and overlay: changing its palette or Gamma
  does not recolor a true-color texture. Explicit pseudocolor follows the palette,
  Gamma and inversion. The settled preview and export share the sampled height
  field without hidden extra smoothing. Relative relief is in image units.
- **Model export**: OBJ, MTL, texture PNG and JSON metadata are a file group;
  keep them together. Export runs in the background. Conflicts are checked for
  every member, preparation is staged, cancellation keeps existing files, and
  commit failures roll back replacements. Saving a model does not export its
  parent fusion image or remove that image's unsaved protection.

Computation runs in the background with progress and cancellation; the AI
method reuses the same memory-safe tiling as AI fusion, so deep stacks
work. The CLI equivalent is `--depth-map [path]` with `--depth-colormap`.

### Crash Auto-Recovery

OpenFocus continuously keeps a snapshot of your working session (stack,
settings, labels, current frame). If the application closes unexpectedly — a crash, power loss, or a forced kill — the next launch detects it and
offers **Restore Session** to return exactly to where you were. Clean
exits never show the prompt. Recovery data lives next to your settings
(portable installs keep it beside the executable, so it travels with you).

### Recent Files

Every stack you open is remembered under **File → Open Recent** (up to 8 entries). Press `Alt+1`–`Alt+8` to reopen an entry directly. Reopening the same stack moves it to the top; entries whose files no longer exist can be removed with a prompt, and **Clear Recent List** empties the menu.

### Drag and Drop

You can also drag a folder directly onto the application window to import images. Dropping more files onto an existing stack appends them (after a size check).

### EXIF Orientation

Camera JPEG/TIFF files with an EXIF Orientation tag are rotated/flip-corrected
automatically during import, so portrait shots load upright.

### 16-bit Sources

16-bit PNG/TIFF inputs are detected automatically and preserved end-to-end:
fusion runs at full depth and results can be saved back as 16-bit PNG/TIFF
(JPG/BMP exports are converted to 8-bit, as those formats cannot store 16 bits).

Camera RAW files are developed to 16-bit (camera white balance, sRGB gamma)
with auto-brightening disabled, so every frame of a stack gets the same
exposure mapping; per-frame auto exposure would make brightness flicker
between frames. Save as TIFF or PNG to keep the 16-bit data.

### Clearing the Stack

To remove all loaded images, go to **File → Clear Stack** or press `Ctrl+W`.

---

## 5. Viewing and Navigating Images

### Navigating the Image Stack

- **Slider**: Drag the slider below the source panel to scroll through frames
- **File List**: Click any filename in the right panel's source images list
- **Keyboard**: Use arrow keys (Left/Right) to navigate

### Zooming and Panning

- **Zoom In/Out**: Mouse wheel over the image
- **Fine Zoom**: Hold `Ctrl` + mouse wheel for smaller increments
- **Fit to Window**: Double-click the image to toggle between fit and 100% view
- **Pan**: Click and drag to move the image within the panel

### Synchronized Navigation

When you navigate in the source panel, the result panel shows the corresponding result frame if available.

---

## 6. Wipe Compare View

The Wipe view overlays two images in a single frame with a draggable vertical divider, making misalignments and quality differences directly visible across the split.

### Opening the Wipe View

1. Render at least one result (fusion or registration)
2. Press the **Wipe** toggle in the top-right corner of the result panel title bar
3. The result area switches to the compare view; press **Wipe** again to return

### Reading the View

- **Side A (left of the divider)**: a source frame — by default it follows the source panel slider
- **Side B (right of the divider)**: the most recent render
- **White divider + round handle**: drag horizontally to move the split
- Dragging edges that jump across the divider reveals registration errors that are hard to see in side-by-side panels

### Choosing What to Compare

The bar at the bottom of the compare view has one group per side:

- **Side A** — check **Follow source** (default) to track the source panel slider, or uncheck it to lock side A to any frame with its own slider
- **Side B** — check **Latest result** (default) to always show the newest render, or uncheck it to scrub through your output history with its slider

Both sides update live: new renders, source-slider moves, and frame deletions are reflected immediately.

### Zooming and Panning

The compare view shares one transform for both images:

- **Zoom**: mouse wheel (zooms around the cursor)
- **Pan**: click and drag anywhere except the divider
- **Reset**: double-click

Wipe mode and ROI mode share the result panel, so entering one exits the other. Clearing the stack also exits wipe mode.

---

## 7. Image Registration

Image registration corrects spatial misalignment between frames in your image stack. This is essential when images have slight shifts or perspective changes.

### Registration Methods

#### ECC (Enhanced Correlation Coefficient)

- Uses optimization to find the best alignment based on correlation
- Suitable for subtle misalignments
- Provides sub-pixel accuracy
- Slower than Homography but more precise

#### Homography

- Based on feature point matching (SIFT/ORB features)
- Handles larger geometric transformations
- Faster than ECC for initial alignment
- Good for images with significant perspective changes

#### OpenFocus (fast)

- Bounded similarity alignment (translation + rotation + scale),
  with per-frame brightness equalization and auto-crop
- Sub-second per frame — much faster than ECC or Homography
- Frames shifting more than **8% of the frame size are left unaligned**
  rather than partially corrected. Override with the CLI
  `--align-shift-limit`
- For non-rigid or extreme geometry, pick ECC or Homography instead

### Enabling Registration

1. In the right panel, locate the **Registration** group
2. Pick one method from the dropdown: **No alignment**, **OpenFocus (fast)**,
   **Align (ECC)** or **Align (Homography)**
3. The OpenFocus fast alignment can also be chained in front of ECC or
   Homography (coarse global pass first, then the precise local pass)

### When to Use Registration

- Your images have visible misalignment
- Images were captured handheld
- The scene has parallax (closer objects shift between frames)
- Results show ghosting or doubling artifacts

### Registration Settings

Access additional settings via **Settings → Registration**:
- **Downscale Width**: Control preprocessing resolution (default: 1024px)
- Lower values = faster processing, potentially lower accuracy
- Higher values = slower processing, potentially higher accuracy

---

## 8. Image Fusion Methods

OpenFocus offers eight fusion algorithms (plus an alignment-only mode). Each has different characteristics suitable for various image types.

### Guided Filter (Default)

- **Algorithm**: Uses guided filtering for edge-preserving fusion
- **Best for**: General purpose, balanced speed and quality
- **Advantages**: Good edge preservation, fast processing
- **Parameter**: Kernel size (adjustable from 1-51, odd values only)
  - Smaller kernel: More local detail, potentially more noise
  - Larger kernel: Smoother results, better noise suppression

### DCT (Discrete Cosine Transform)

- **Algorithm**: Frequency-domain fusion using DCT coefficients
- **Best for**: Texture-rich images, scientific imaging
- **Advantages**: Preserves fine textures, mathematically sound
- **Parameter**: Kernel size controls the processing window

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algorithm**: Multi-scale wavelet domain fusion
- **Best for**: Complex scenes with multiple detail levels
- **Advantages**: Excellent multi-scale analysis, good directionality
- **Fusion rule**: one decision per pixel and level for all orientations and
  colour channels (N-way, majority-consistency checked); the low-pass band
  follows the coarsest decision instead of averaging every frame
- **Parameter**: Wavelet levels (1–6, default 4) in the right panel. There is
  no spatial kernel, so the kernel slider is hidden for this method.

### GFG-FGF (Gaussian-like Four-neighbourhood Gradient + Fast Guided Filter)

- **Algorithm**: implementation of Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): a blur-difference focus map |I − mean(I)| is enriched
  with the Gaussian-like four-neighbourhood gradient (GFG) operator, refined
  by a fast guided filter, decided per pixel (ties broken by a 3×3 Sobel
  small-region measure), and the binary decision maps are smoothed by the
  fast guided filter again before the weighted blend.
- **Best for**: Microscopy and images with clear focus regions
- **Advantages**: Fast, sharp focus-region boundaries; memory does not grow
  with the number of frames
- **Parameter**: Kernel = size of the mean filter of the coarse focus map

### StackMFF-V3 (Deep Learning)

- **Algorithm**: Pixel-wise source classification (requires PyTorch). The
  network predicts, for every pixel, which input frame it comes from, then
  either blends the frames by those probabilities or picks one frame
- **AI output mode** (right panel, only for this method):
  - **Soft blend (weighted)** — default; each pixel is a probability-weighted
    mix of the frames. Smooth transitions, no hard seams; matches the paper's
    reported configuration
  - **Hard pick (single frame)** — each pixel is copied from the most
    probable frame. Crisper selection with no cross-frame interpolation; a
    pixel-level seam can appear where two frames tie
- **Best for**: Highest quality results when GPU is available
- **Advantages**: Order-invariant (frame order never changes the result),
  handles any stack length, no per-method tuning, automatic optimization
- **Requirements**: PyTorch installation, optional CUDA GPU (no torchvision)
- **Note**: May be unavailable if PyTorch is not installed

### OpenFocus A / B / C

- **Algorithm**: Variance-weighted average, variance depth-map pick, and
  Laplacian-pyramid fusion
- **OpenFocus A**: variance-weighted average with a last-frame prior —
  fast, colour-stable on short stacks
- **OpenFocus B**: variance depth-map pick — selects the sharpest frame
  per pixel; best on smooth surfaces and microscopy
- **OpenFocus C**: Laplacian-pyramid per-band blend — natural look on
  textured subjects, boosts local contrast
- **Parameters**: A and B take a Variance window (kernel slider); C has no
  radius. All three take a Smoothing sigma.

**Smoothing**: one spin box under the kernel slider, with the meaning of the
selected method (range 0–8, step 0.2):

| Method | Label | Default | Effect |
|---|---|---|---|
| OpenFocus A | Blend smoothing (σ) | 2.0 | Gaussian smoothing of the per-frame weight map — how widely the sharp regions of neighbouring frames are blended |
| OpenFocus B | Depth smoothing (σ) | 1.2 | Gaussian smoothing of the depth map — width of the transition between frames |
| OpenFocus C | Blend smoothing (σ) | 3.0 | Gaussian smoothing of each pyramid band's weight map |

0 keeps the crispest per-pixel choice (finest detail, but speckles on noisy
stacks); larger values soften transitions and halos. Each method remembers
its own value across sessions, and batch processing uses the same rule.

C has no Radius: Helicon's Method C exposes only Smoothing, so the kernel
slider is hidden for C instead of silently doing nothing.

CLI: `--depth-smoothing SIGMA` (B) and `--blend-smoothing SIGMA` (A, C).

**Brightness options are per method.** *Match exposure before fusion* and
*Remove brightness drift* are remembered separately for each fusion method,
because they answer a per-stack question rather than a global preference.
Both default to on for OpenFocus A/B/C (which have no built-in exposure
calibration); the other methods keep the historical defaults (exposure
matching on, drift removal off). Switching methods shows that method's own
values, and those values survive a restart. **Reset** clears the remembered
brightness, smoothing and Radius values of every method.

### Alignment only (no fusion)

Select **Alignment only (no fusion)** to register the stack and add one stack
entry to the output list. The frames stay in memory until you export them;
alignment-only does not create an extra cache or write beside the source files.
Save the entry to create one folder inside your chosen destination. Original
filename stems are retained, with the extension of the format you select.

### Selecting a Fusion Method

1. In the right panel, locate the **Fusion** group
2. Pick a method from the **dropdown** (exactly one is always selected;
   the first entry registers without fusing)
3. Adjust kernel size if applicable — the label shows what the kernel
   means for the selected method
4. Click **Start Render** to process

### Algorithm Comparison Guide

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| Guided Filter | Fast | Good | General use |
| DCT | Medium | Good | Textures |
| DTCWT | Medium | Very Good | Complex scenes |
| GFG-FGF | Fast | Good | Focus regions |
| AI Fusion (StackMFF-V3) | Slow (GPU) | Excellent | Best quality |
| OpenFocus A | Very Fast | Very Good | Short stacks, colour stability |
| OpenFocus B | Very Fast | Very Good | Smooth surfaces, microscopy |
| OpenFocus C | Fast | Very Good | Textured subjects, natural look |

Open **Tools → Method comparison** to adjust several methods independently.
Each method's name, parameter labels and controls stay together when the window
narrows. Changes update the preview automatically. **ROI preview (fast)** fuses
the selected area with surrounding context; **Full-image fusion** checks the
whole-image result and lets you inspect another area without rendering again.
ROI results can differ from full-image results because exposure statistics and
some algorithms depend on the image extent.

Use **Preview size (px)** above the right panes to choose a square size from
160 to 960 pixels, by dragging the slider or entering a number. All methods
use the same size and wrap to fit the available width. The size is remembered
for the next opening. Changing it keeps the current magnification and adjusts
the inspection square around its centre, clamped to the image boundaries.

Press **F11** or the **Fullscreen (F11)** button to use the whole screen; the
button text switches to **Exit fullscreen (F11)**. The title bar can also
maximise the window. **Esc** leaves fullscreen first and only closes the dialog
on a second press. Returning from fullscreen restores the previous window size,
and the result grid reflows to the available width either way.

Left-drag the source image to move the white inspection square; releasing the
button keeps it in place. The right panes show exactly that region from the
same aligned stack. Use the wheel to change the region's magnification,
right-drag to pan the source, **Ctrl+wheel** to zoom the source, and **F** to fit.
Switching source frames or resizing the window preserves the selection.
Cancel stops further work; closing waits for the current native operation to
return safely.


### Quick Preview (Draft Render)

Tuning parameters on large stacks is slow at full resolution. Check
**Quick preview** above the render button and OpenFocus first renders a
downscaled draft (longest side 1200 px) — typically several times faster,
including registration. The preview is displayed with a "Preview" note in
the status-bar summary after the render and is **not** added to the output history. Uncheck
the box and render again to produce the full-resolution result. Preview
mode does not apply to ROI renders and never pollutes the full-resolution
alignment cache.

---

## 9. Image Transformations

### Rotation

1. Go to **Edit → Rotate**
2. Choose rotation type:
   - **90° Clockwise**: Rotate all images 90 degrees right
   - **90° Counter-Clockwise**: Rotate all images 90 degrees left
   - **180°**: Flip images upside down

### Flipping

1. Go to **Edit → Flip**
2. Choose flip type:
   - **Horizontal Flip**: Mirror left-to-right
   - **Vertical Flip**: Mirror top-to-bottom

### Resizing

1. Go to **Edit → Resize**
2. A dialog will appear for resizing options
3. Enter new dimensions or scale percentage
4. All images in the stack will be resized proportionally

### Transformation Order

Transformations are applied to the entire image stack, maintaining alignment between frames.

---

## 10. Retouching, Labels and Annotations

### Retouching the Result (Edit → Retouch Result…, `Ctrl+T`)

Fusion can still leave a local defect — a halo around a high-contrast edge,
a bristle taken from the wrong frame, noise in a flat area. The workspace
copies pixels from the exact frames that produced the selected result:

1. Select one output in the output list (right-click → *Retouch Result…*
   works too) and press `Ctrl+T`. With no selection, a chooser lists every
   result and marks which ones have recorded source frames.
2. The source frame is on the left, the result on the right. Both share zoom
   and position: wheel zooms, right/middle-drag (or Space+drag) pans, `F`
   fits, `1` shows 100 %.
3. Pick a thumbnail, drag the frame slider, or press `,` / `.`. A stroke is locked
   to that frame and the current brush settings; the frame cannot change
   until the stroke ends.
4. Choose **Source Brush** (`B`) or **Restore Brush** (`E`). Diameter,
   hardness and opacity have both sliders and numeric fields; `[` / `]`
   adjusts the diameter. The restore brush returns pixels to the image you
   opened, not to an older output.
5. Hold `Tab` to see the result before this session and `Alt` to peek at the
   source frame. The sharpness reference is optional, covers only the source
   canvas and uses one shared scale for every frame.
6. `Ctrl+Z` / `Ctrl+Y` undo and redo whole strokes. **Restore All…** is one
   undoable step. **Apply as New Output** adds an entry named
   `<name> · Retouched N` and selects it; the original stays untouched. The
   new entry is marked *Not exported* until you export it from the main
   window, and OpenFocus protects it from automatic history eviction and
   from quitting without a decision.

Provenance travels with the output: the aligned frames, ROI crop/paste
offsets, DCT edge trimming and bit depth are recorded when the result is
rendered. Batch files, legacy results and align-only folder entries have no
recorded source, so they are listed as unavailable instead of being guessed
from the current stack. Unapplied strokes and undo history are session-only;
saving a project does not store them.

### Managing Labels

Open **Tools → Labels** after loading images or producing a result. Each label
has its own target: the source stack, the registered stack, or output results.
Output labels can apply to all results or to the selected result by its name;
renaming or reordering that result does not change the association.

Use **Add** to create another label, select an entry to edit it, and use its
checkbox to enable or disable it. Enter fixed text or insert `{value}` for a
frame number. The number is the starting value plus the zero-based frame index
times the interval; a single output always uses the starting value.

Choose all frames, the current frame, or a custom range such as `3-5,8`.
Invalid or unavailable frame numbers prevent applying the draft. Current and
custom ranges follow the selected source identities when the stack is reordered;
all-frame labels also apply to appended frames. Missing targets stay associated
with their original identities until you explicitly choose a new target.

The preview uses the actual target. Navigate with Previous/Next or type its frame
number, choose **Fit** or **100%**, and drag the image to pan at 100%. Position and
font size use image pixels. A fallback font or an unsuccessful draw is reported.
Labels preserve the original bit depth when exported as PNG or TIFF.

**Apply and Close** commits the complete draft as one undoable change. Cancel,
Esc and closing the dialog discard it. Selecting another label keeps your draft
edits. **Delete** removes only the selected entry, including the last entry.
The menu removal actions clear source labels, registered-stack labels, output
labels, or all labels independently and can also be undone. Applying labels changes the session display;
export an image to write the annotations to a file.

---

## 11. Settings and Configuration

Open **Settings → Advanced Settings** for performance, registration, StackMFF AI and startup preferences. Changes remain drafts until **OK**; Cancel, Esc and closing discard them. Reducing the tile size immediately updates the visible overlap limit to half a tile. AI output mode affects StackMFF only; registration width can change alignment accuracy.

Automatic update downloads run while idle and never install automatically. A completed download remains ready until you explicitly choose **Restart and Update**. Restart protects active edits and unsaved work, then writes recovery data before launching replacement. Cancelling a download or preparation waits for the current safe boundary. An existing complete, newer staged build can be installed offline.

Flat-field correction changes session working pixels without overwriting source files. Select the white reference and a preview frame, then generate the preview again after changing source or scale. Reference and working/base images must share a coordinate domain and size. Applying prepares both stacks before committing; optional reference-frame removal belongs to the same undo step. Export corrected source images when you need to preserve these pixels outside the session.

**Help → Send feedback** opens local logs. Copy copies the visible session log; the feedback ZIP contains the session/full application logs and environment information. Review the package before submitting it yourself. Opening GitHub only opens the tracker. Export cancellation leaves files unchanged; an error keeps the previous destination file and shows the failed path.

### Tile Settings (Memory Optimization)

For large images, OpenFocus uses tile-based processing to avoid memory issues.

Access via **Settings → Advanced Settings → Performance**:

- **Tile Block Size**: Size of each processing tile (default: 1024px)
- **Tile Overlap**: Overlap between tiles for seamless blending (default: 256px)
- **Tile Threshold**: Image size above which tiles are used (default: 2048px)

**Recommendations**:
- Smaller block size: Lower memory usage, slower processing
- Larger block size: Higher memory usage, faster processing
- Increase overlap if you see visible seams in results

### Registration Settings

Access via **Settings → Advanced Settings → Registration**:

- **Downscale Width**: Preprocessing resolution for registration
  - Default: 1024px
  - Lower for speed, higher for accuracy

### Performance Reference

See [PERFORMANCE.md](./PERFORMANCE.md) for measured throughput and memory
numbers on large real stacks, and practical guidance for deep stacks and
high-resolution input.

### Thread Count Settings

Control CPU parallel processing:

Access via **Settings → Advanced Settings → Performance**:

- `0` selects automatic scheduling and is the default for new settings.
- Positive values are upper limits; saved limits survive upgrades.
- CPU capacity, resource mode and memory headroom may lower concurrency.
  More threads do not necessarily make a render faster.

### Resource Mode

**Settings → Advanced Settings → Performance → Resource mode** decides how
much of the machine a render may use. It never changes the fused image.

| Mode | Behaviour |
|---|---|
| Balanced (default) | Keeps roughly 20% of the physical cores and a memory reserve free for the desktop |
| Responsive | Uses about half the cores and a smaller in-flight window; slower renders, most responsive machine |
| Maximum | Uses every physical core the memory budget allows |

The planner considers physical cores, CPU affinity and quotas, and runs one
parallel layer at a time. Heavy jobs share admission through preparation and
publication; cancelling waits for the active native call to finish. Unknown or
insufficient memory produces a resource error. The selected tile grid stays
unchanged; choosing smaller tiles yourself can change output pixels.

### GPU Acceleration Toggle

OpenFocus uses the GPU (CUDA/MPS) for the AI neural model (StackMFF-V3); classical algorithms and application registration run on CPU. Explicit low-level CuPy registration is a separate numerical backend.

- Toggle via **Settings → GPU Acceleration**
- Unchecking forces everything onto the CPU — useful when GPU drivers misbehave
- The status panel shows `CPU*` while the GPU is disabled
- The choice takes effect on the next render and is remembered

### AI Batch Size

Access via **Settings → Advanced Settings → AI**:

- The saved batch limit defaults to 2; the current execution policy caps actual
  inference at one tile per call on every device.
- Increasing this limit does not enable unvalidated GPU batching or reduce
  spatial tile size.

### Themes

Two interface themes are available: **Dark (default)** and **Light**. Switch via
**Settings → Theme**; the choice is applied immediately (the Windows title
bar follows) and remembered. First-launch users choose a theme in the
welcome dialog.

### Portable Mode

Create an empty file named `OpenFocus.portable` next to `OpenFocus.exe`
(the portable zip ships with one) and all user data — settings, logs,
recent files — is stored in an `OpenFocusData` folder beside the app
instead of your user profile, so the installation can travel on a USB
stick. Delete the marker to switch back to the normal profile.

### EXIF Metadata

Source metadata (camera model, exposure, orientation and so on) is kept:
orientation is applied on load, and PNG/TIFF/WebP exports embed the
original EXIF automatically. JPG/BMP exports cannot store full EXIF and
are written without it.

### HEIC/HEIF Input

iPhone photos load directly. Requires the optional `pillow-heif` package
(`pip install pillow-heif`; included in the packaged builds).

### Updates

Project files (`.ofproj`) behave like a document: the title bar shows the
project name with a `*` while there are unsaved changes, and closing the app
or opening another project asks before discarding them. The project stores
reconstructable input and settings; it does not store output pixels or remove
their export protection. Source images are referenced, not copied.

Help → Check for Updates compares your version against GitHub Releases.
**Download and Install** fetches the new installer in-app (with progress)
and launches it; Open Downloads Page goes to the releases list. The check
runs silently at most once a day.

### Registration Cache

OpenFocus can store registration results in a `.openfocus_cache` folder
next to the source stack. Re-opening the same stack (files unchanged)
then skips alignment entirely. Toggle via **Settings → Registration
Cache**; stale entries are replaced automatically.

### Persisted Settings

All of the following survive an application restart automatically:
thread count, tile parameters, registration downscale width, AI Fusion (StackMFF-V3)
batch size, the GPU acceleration toggle, the registration cache toggle,
interface language, window size/panel layout, the drag-out export format,
the per-method brightness options, smoothing sigma and Radius values,
and your recently opened stacks. Settings dialogs save the moment you confirm them;
everything is also saved when the application closes. Preferences are
stored per user in `OpenFocus/OpenFocus.ini` (QSettings INI format).

### Interface Language

Switch anytime via **Settings → Language → English / 中文**. The change
applies immediately and is remembered for future sessions.

### Automatic Update Check

Once per day, a few seconds after launch, OpenFocus silently checks GitHub
for a newer release. When one exists, a banner appears above the image
panels: the version, **Update now** (downloads and restarts in one step),
**What's new** (release notes) and a dismiss button. Dismissing hides the
banner for that version only — the next release brings it back, and
**Help → Check for Updates** re-arms it immediately.

While the download runs in the background the status bar shows progress.
You can also check manually anytime via **Help → Check for Updates**.

---

## 12. Batch Processing

Process multiple folders automatically:

1. Go to **Tools → Batch Processing**
2. In the batch dialog:
   - **Add Folders**: Select folders containing image sequences
   - **Drag folders in**: Drop one or more stack folders anywhere on the
     dialog (or on the folder list) to queue them all at once; dropping
     switches to multi-folder mode and the thumbnails appear as they decode
   - **Remove Folders**: Remove selected folders from the list
   - **Clear All**: Remove all folders
3. Configure output settings:
   - **Save to Subfolder**: Creates results in each input folder
   - **Save to Same Folder**: Saves alongside input while preserving existing files
   - **Custom Folder**: Specify output location
4. Review current fusion/registration settings
5. Click **Start Batch** to begin

The batch dialog shows real-time progress. You can cancel processing at any time.

### Batch Processing Notes

- Starts from a snapshot of the main window settings; batch controls override it.
- Each folder or split group is processed independently through the same pipeline.
- **What lands where, by mode.** With **align only**, the selected destination
  receives one `<stack>_aligned` folder per stack and the output panel lists
  that folder. With **fusion + save aligned**, the fused image is written as
  `<stack>.<ext>` in the destination and its aligned frames go to
  `<stack>_aligned` beside it; the output panel lists only the fused result.
  With **fusion and no alignment export**, only the fused image is written.
  The output panel names folders exactly as they are on disk.
- Aligned frames keep the original filename stems. Duplicate names and existing
  folders receive numeric suffixes. Source files are never replaced.
- Leftover `*_registered*` copies from older builds are not loaded as frames
  when their original file is present in the same folder; the load message
  reports how many were skipped.
- PNG/TIFF preserve 16-bit data; explicitly choosing JPG/BMP converts to 8-bit.
- Cancellation waits for the worker to stop. Committed files remain available;
  partial, failed, cancelled and unstarted tasks are counted separately.

---

## 13. Exporting Results

### Saving a Single Result

1. Select the result you want to save in the output list
2. Go to **File → Save** or press `Ctrl+S`
3. Choose format and location
4. Click Save

### Drag-out Export Format

Dragging results from the output list writes image files using the format
chosen under the output list's context menu (**Drag-out format**: JPG,
PNG or TIFF). The choice is remembered.

### Saving the Registered Stack

1. Go to **File → Save Stack → Registered Stack**
2. Choose save format:
   - **Save as Folder**: Choose a destination and encoding; each stack gets one new subfolder containing its original frame names
   - **Save as GIF**: Creates an animated GIF of the aligned sequence

### Saving the Input Stack

1. Go to **File → Save Stack → Input Stack**
2. Choose save format:
   - **Save as Folder**: Saves original frames as separate images
   - **Save as GIF**: Creates animated GIF of input sequence

### Export Formats

- **JPG**: Compressed, good quality-to-size ratio
- **PNG**: Lossless, preserves 8-bit and 16-bit image data
- **BMP**: Uncompressed, converts 16-bit image data to 8-bit
- **TIFF**: Preserves 8-bit and 16-bit data; multi-page export keeps the whole stack in one file
- **GIF**: Animation with an 8-bit palette; cancellation leaves an existing destination intact

---

## 14. Command Line Interface

OpenFocus can run without the GUI — useful for scripted pipelines and CI.
From a Python environment (or next to the packaged executable):

```bash
# Fuse a folder with the default Guided Filter
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT with ECC registration, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 forced to CPU with a custom tile size
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explicit file list; video files are also accepted as input
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Single multi-page TIFF Z-stack with a metadata-calibrated scale bar
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Option | Description |
|--------|-------------|
| `--input, -i` | Image folder, video file, or one or more image files |
| `--output, -o` | Output image path (extension sets the format) |
| `--method, -m` | `stackmffv3` (default), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (default), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | `openfocus` align only: max accepted shift as a fraction of the frame size (default 0.08; frames above it are left unaligned) |
| `--exposure-match` | Match every frame's mean to the last frame before fusion (default: on; `--no-exposure-match` disables). Whole-stack gains, so tiled output stays seamless; applies to every method and to the OpenFocus aligner's brightness reference |
| `--base-frame` | Optional post-fusion step: match the output's low frequencies to the `last`/`first` frame (`none` default) |
| `--sort` | Frame order for `openfocus_a/b/c`: `asc` (default, file order), `desc` (reverse a far-to-near sweep), `off` |
| `--dust-map WHITE` | Flat-field (dust map) correction from a defocused white-field shot |
| `--kernel, -k` | Kernel size (forced odd; for the openfocus family it is the source of the variance-window radius) |
| `--depth-smoothing SIGMA` | `openfocus_b` only: Gaussian sigma of the frame-to-frame depth transition (default 1.2; 0 keeps the finest detail) |
| `--blend-smoothing SIGMA` | `openfocus_a` / `openfocus_c` only: Gaussian sigma of the per-frame weight map (defaults 2.0 for A, 3.0 for C; 0 keeps the crispest per-pixel choice) |
| `--threads, -t` | Worker thread upper limit; `0` = automatic (default), positive values cap concurrency |
| `--resource-preference` | `balanced` (default), `responsive`, or `maximum`; controls resource scheduling |
| `--batch-size` | AI tile batch upper limit (default 2); actual inference currently uses one tile per call |
| `--cpu` | Force CPU even if a GPU is available |
| `--downscale` | Registration feature-detection downscale width |
| `--tile-size` | Tile block size for large images |
| `--output-dir, -d` | Batch mode: fuse every input folder, writing `<folder>.png` per stack |
| `--scale-bar` | Draw a scale bar (calibration auto-detected from TIFF metadata) |
| `--depth-map [path]` | Also save a colorized focus-position (depth) map (default: `<output>_depth.png`) |
| `--depth-colormap` | Colormap for `--depth-map`: turbo, viridis, jet, gray, … (default turbo) |
| `--scale-um` | Pixel size in µm/px, overriding metadata (implies `--scale-bar`) |

Exit codes: `0` success, `1` processing error, `2` usage error. Running
`python main.py --help` prints all options; launching with a plain path
(`python main.py C:\stack`) still opens the GUI with that folder loaded.

---

## 15. Keyboard Shortcuts

### File Operations

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open Folder |
| `Ctrl+Shift+O` | Open Video |
| `Ctrl+S` | Save Result |
| `Ctrl+Shift+S` | Save Registered Stack |
| `Ctrl+W` | Clear Stack |
| `Ctrl+Q` | Exit |
| `Ctrl+R` | Start Render |
| `Ctrl+Shift+R` | Compare All methods |
| `Ctrl+T` | Retouch Result |
| `Ctrl+M` | Depth Map |
| `Ctrl+E` | Toggle Wipe compare |
| `Ctrl+Shift+B` | Batch Processing |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo (stack edits) |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Left Arrow` | Previous frame |
| `Right Arrow` | Next frame |
| `Mouse Wheel` | Zoom in/out |
| `Ctrl + Wheel` | Fine zoom |
| `Double-click` | Toggle fit/100% view |

### General

| Shortcut | Action |
|----------|--------|
| `Space` | Pan mode toggle |
| `Delete` | Delete selected item |

---

## 16. Troubleshooting

### Common Issues

#### "No Images Loaded"

- Make sure you've imported an image folder or video first
- Check that the folder contains supported image formats

#### Fusion Results Look Blurry or Ghosted

- Enable image registration (pick one from the dropdown: Homography / ECC / OpenFocus fast)
- Ensure images are properly aligned before fusion
- Try a different fusion algorithm
- Increase kernel size for smoother results

#### AI Unavailable

- Install PyTorch: `pip install torch torchvision`
- For GPU support, install CUDA-enabled PyTorch
- Check via **Help → Environment Info**

#### Out of Memory Errors

- Enable tile processing in **Settings → Tile**
- Reduce tile block size
- Use smaller kernel sizes
- Resize images to smaller dimensions

#### Registration Fails

- Try only one registration method at a time
- Reduce the downscale width in **Settings → Registration**
- Ensure images have sufficient contrast and features

#### Slow Processing

- Increase thread count in **Settings → Thread Count Settings**
- Use a more powerful CPU
- Reduce image resolution
- Use simpler fusion algorithms (Guided Filter instead of AI Fusion)

#### Windows SmartScreen Warning on First Launch

The released binaries are unsigned. Click **More info → Run anyway**.
The first launch after unzipping may take 30–60 seconds.

#### macOS Says the App "Cannot Be Opened" or "Is Damaged"

The app is unsigned and not notarized. After moving `OpenFocus.app` to
Applications, run:

```bash
xattr -cr /Applications/OpenFocus.app
```

or right-click the app and choose **Open**. Released macOS builds are for
Apple Silicon (M1/M2/M3/M4); Intel Macs can run from source.

### Getting Help

1. **Environment Info**: Go to **Help → Environment Info** to check your installation
2. **Contact Support**: Go to **Help → Contact Us** for support information

---

## Appendix: Fusion Algorithm Details

### Guided Filter

The guided filter performs edge-preserving smoothing using a guidance image. It computes local linear transformations between the guidance and input images, then applies these to produce the output. This method excels at preserving edges while reducing noise.

### DCT (Discrete Cosine Transform)

DCT transforms images to the frequency domain where fusion decisions are made based on coefficient magnitudes. High-frequency components (details) are preserved while low-frequency components (smooth regions) are merged based on local energy.

### DTCWT (Dual-Tree Complex Wavelet Transform)

DTCWT provides multi-scale, multi-directional decomposition of images. It captures features at different scales and orientations, allowing for sophisticated fusion rules that preserve both texture and structure information.

### GFG-FGF

Implements Fu et al. (2024): the coarse focus map |I − mean(I)| is enriched with the Gaussian-like four-neighbourhood gradient — the 3×3 template T = [1 2 1; 2 4 2; 1 2 1]/16 gives P = I⊗T, and G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2 with values below T0 = 0.005 dropped — then refined by a fast guided filter. Pixels pick the frame with the largest focus value (ties: 3×3 Sobel small-region measure), and the binary decision maps are smoothed by the fast guided filter before the weighted blend.

### AI Fusion (StackMFF-V3)

A deep learning approach using a neural network trained on large datasets of multi-focus image pairs. The network learns optimal fusion strategies automatically, providing state-of-the-art results when properly configured.

---

*OpenFocus is an open-source project. For updates, bug reports, and feature requests, please visit the project repository.*

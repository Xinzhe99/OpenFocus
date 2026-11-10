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
   - [Batch image conversion](#batch-image-conversion)
13. [Exporting Results](#13-exporting-results)
14. [Command Line Interface](#14-command-line-interface)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Introduction

OpenFocus is a professional multi-focus image fusion desktop application designed for researchers, engineers, and image processing professionals. The software processes multi-focus image sequences using advanced registration and fusion algorithms to generate all-in-focus images with full depth of field.

### Key Features

- **Multi-Focus Image Fusion**: Combine multiple images with different focus points into one fully focused image
- **Multiple Fusion Algorithms**: Ten methods — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF V1/V2/V3 and OpenFocus A/B/C — plus alignment-only mode
- **Image Registration**: Align misaligned frames with ECC, homography, homography followed by ECC, or the fast OpenFocus aligner
- **Auto background**: replace a messy stacked background with one real frame, and see where no frame saw the scene clearly
- **Wipe Compare View**: Inspect alignment and compare results with an A/B divider view
- **Project Files**: restore input references and reproducible settings; export result pixels separately
- **Processing Templates**: save and reapply fusion, registration and resource settings to any loaded stack
- **Multi-page TIFF**: single-file Z-stacks load page-by-page and export back into one file
- **Depth Map**: per-pixel focus-position maps with custom pseudo-color schemes
- **Scale Bar**: publication-grade scale bars with automatic µm/px calibration
- **Watermark**: visible tiled attribution text burned into exported copies
- **Crash Auto-Recovery**: the last working session is offered for restore after a crash
- **Multi-language**: twelve UI languages — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — with automatic system-language detection
- **Headless and Agent CLIs**: Script headless pipelines for CI, or drive a running window through the local control channel
- **Flexible Export**: Save results as individual images, folders, or GIF animations
- **Image Transformations**: Rotate, flip, and resize image stacks
- **Label Addition**: Add text labels to registered and input stacks
- **Persistent Settings**: Your preferences, language, and recently opened stacks survive restarts

### Supported Input Formats

- Image files: JPG/JPEG, PNG, BMP, TIFF, WebP, and other common formats
- Video files: MP4, AVI, MOV, MKV, WMV, FLV and WEBM (decoded into image sequences)
- Folders containing numbered image sequences
- Camera RAW files (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2, RWL, FFF, MEF, DCR and other common
  RAW formats). Decoding uses `rawpy` (LibRaw), which `pip install openfocus`
  installs and the Windows/macOS builds bundle.
  Camera models and compression support depend on the installed LibRaw version.
  RAW is decoded to a 16-bit image at its original dimensions before the
  selected loading scale is applied: a smaller scale reduces later memory use
  but does not eliminate the full-size decode peak. Grouped fusion does not support RAW.

Transparent PNG stacks can be fused in the desktop application, desktop batch
processing, or the command-line interface, without grouped fusion. Every frame
must be an 8-bit or 16-bit PNG of the same dimensions and bit depth, with the same alpha mask.
The mask must be zero outside one nonempty rectangle and have one constant,
nonzero opacity throughout the rectangle. Partially transparent constant
opacity is allowed. The visible rectangle is fused independently, so hidden
colors cannot influence focus measurement or tile boundaries. PNG output
retains the original canvas size and opacity; hidden output colors are zero.
Desktop source and result views show transparency against a checkerboard.

Load at full size, choose no alignment, turn off Quick Preview and ROI, then
render and save as PNG. Batch requires a fusion method and PNG output without
saving an aligned stack. Alignment, resizing or downsampling, flat-field
correction, depth maps, method comparison, retouching and scale bars are
unavailable for transparent data, and labels cannot be burned into an export.
A fusion method that changes the rectangle's dimensions is
rejected. Varying or feathered masks, different frame masks, transparent TIFF
input and non-PNG output, including GIF and multi-page TIFF, are unsupported.
The CLI rejects sensor cleanup and depth maps for transparent input as well.

For example, the CLI equivalent is `python main.py --input ./png-stack
--output ./result.png --method guided_filter --align none`.

### Supported Output Formats

In **Save As**, ordinary TIFF saves the result as a single image; layered TIFF includes the source frames as layers. Linear DNG is always listed, but saving it requires an unbaked camera-linear result with camera calibration.

- Individual images: JPG, PNG, BMP, TIFF, WebP
- Image sequences: Saved as folders
- Animations: GIF format

---

## 2. Getting Started

### Downloading and Starting the Application

The preferred way to install a packaged build is the project website,
**https://openfocus.top** — it links the current Windows installer, macOS build
and Linux package, with install scripts for terminal use. Prefer it over
rebuilding: those artifacts are the exact release pipeline output.
`pip install openfocus` installs only the headless command-line interface (the
wheel deliberately does not depend on PyQt6); use a desktop build or a source
checkout for the GUI.

After installation, launch OpenFocus by:

1. **Windows**: Double-click the OpenFocus shortcut or executable file
2. **macOS**: Open `OpenFocus.app` from Applications
3. **Python Source**: Run `pip install -r requirements.txt`, then
   `python main.py` in your activated environment

The first launch opens the welcome dialog, where you can choose the theme and
load the bundled demo stack. A fresh profile defaults to the light theme; the
choice is remembered.

On first launch and after an update, the optional account screen offers email,
GitHub and Google sign-in. You can continue without an account or network. To
sign in, review the current service terms and privacy policy and accept them.
Use **Help → OpenFocus account** to open the screen again, sign out or delete
the account. Signing in can enable automatic error reports; you can turn them
off under **Settings → Advanced Settings → Diagnostics**.

### Quick Start Workflow

1. Import your image sequence (File → Open Folder or drag-and-drop)
2. Configure fusion settings in the right panel
3. Registration starts on **ECC**; choose **No alignment** if the frames are
   already aligned
4. Adjust kernel size if needed
5. *(optional)* Open **Render settings** and check **Quick preview** to test parameters on a downscaled draft first
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
3. **Action Buttons**: Start Render, plus reset, **Render settings**, and render all methods. The render settings dialog contains quick preview, frame order, exposure matching, brightness drift removal, and output sharpening
   Exposure matching is enabled by default; the other options are unchecked for new settings and after reset. Saved choices are preserved.
4. **Source Images List**: Shows all loaded images with filenames
5. **Output List**: Shows generated results

### Menu Bar

The top menu bar provides access to all functions:

- **File**: import/export operations, projects, processing templates and the recent-files list
- **Edit**: Undo/redo, reset the current stack, image transformations and retouching
- **Tools**: Image stitching, batch processing and conversion, method comparison, depth map, flat-field correction, scale bar, watermark and labels
- **Settings**: Advanced Settings (performance, registration and alignment diagnostics, sensor cleanup, output naming, AI, diagnostics, startup), GPU acceleration, theme and language
- **Help**: Environment info, contact information, update check, and the log folder

**Tools → Stitch Images…** opens a separate workspace for already-fused
microscope regions and camera panoramas; it does not modify the current stack.
The **Project assistant** tab gathers import, alignment and control points;
**Source images** holds the list, grid and manual placement; **Panorama editor**
holds seam, blend, exposure/color, output, projection, camera and HDR settings
plus preview overlays. Use **Add files**, **Open folder**, or **Add results** to
copy chosen outputs from the current session, or drag files or folders onto the
list or canvas. Folder import includes supported files directly inside the
folder, not subfolders. Imports use natural filename order, ignore duplicate
paths, and add multi-page TIFF pages as separate images. Select an image and
press Delete or Backspace to remove it from the workspace without deleting the
file.

**Auto-align** finds shared features on bounded thumbnails, ignoring masked and
transparent pixels, and fits each connected overlap. Planar mode estimates
rotation, scale and translation; panorama mode also fits the camera and lens
model. Images without enough common features remain marked unmatched. In planar
mode, drag them or enter X/Y in pixels, scale as a percentage and rotation in degrees,
including coordinates derived from a known stage. Image content cannot infer
the position of a featureless, non-overlapping region, and automatic matching
is not a physical calibration. For a declared stage grid, set rows, columns,
capture order, direction, start corner and estimated overlap, then press
**Apply grid positions**. The cell count must equal the image count, and grid
placement applies only to planar mode. Editing the controls leaves current
positions untouched until Apply, which replaces them and clears match
connections. In planar mode, inspect every overlap and adjust X/Y, scale and rotation before
saving.

**Control points** opens a two-preview editor. Choose image A and B, click the
same feature in each preview, and press **Add point pair**; stored coordinates
are native source pixels even though previews and markers are scaled. The table
lists each pair's enabled state, images, A/B coordinates, manual or automatic
source, and residual in pixels. Correct coordinates, clear **Enabled** on
suspect points, or use **Delete selected**, **Delete worst 10%**, and **Delete
non-adjacent pairs**. **Optimize** fits all enabled pairs and reports alignment
quality: active points, weak pairs, disconnected images and p95 residual in
pixels. A pair needs at least four enabled points to constrain a fit. Automatic
points survive manual edits; verify repeated texture, parallax and specimen
motion. Control-point edits do not create a new full preview: press **Preview
blend** after editing or optimizing to inspect the result.

**Source mask** edits one image's native-resolution mask. Paint with a radius
expressed in source pixels; **Exclude** marks pixels red and **Include**
restores them; **Clear mask** removes the base; **Import mask…** accepts one
PNG (alpha, grayscale or BGR) or TIFF without alpha (grayscale or BGR), whose
dimensions exactly match the source. **Apply** commits the strokes, while
**Cancel** keeps the previous mask.
Excluded pixels do not contribute to seams, feature matching or output pixels,
and a mask or RGBA source makes transparency available in the output.

The **Panorama editor** controls the actual blend. Under **Find optimum
seams**, **Content seam (DP)** searches content seams; **Graph cut** is offered
only when the OpenCV build provides it; **Feather** skips content-seam search.
Seam precision 1/2/4/8, transition half-width, color or color-and-gradient cost
and overlap-loop refinement tune that search; precision, cost and loop
refinement are disabled for Feather. **Blending** is a true multiband pyramid
(1–7 levels, default 5), feather, or zero-overlap hard selection. **Exposure
and color** offers off, exposure compensation, color balance, or a local gain
field, with optional **Blend in linear light**. Per-image lens fields are
enabled in panorama mode: yaw, pitch, roll, horizontal field of view, radial
a/b/c distortion and lens-center shifts. Vignetting k1–k3 and exposure time in
seconds (0 keeps the frame's metadata when available) apply to any selected
image. These are user-supplied or fitted estimates, not inferred physical
calibration.

**Projection** chooses **Planar microscope mosaic**, which uses affine
X/Y/scale/rotation and supports the grid, or a **Rectilinear**, **Cylindrical**
or **Equirectangular** panorama that maps every source through its camera yaw,
pitch, roll, field of view and lens model; panorama scale is pixels per radian,
and **Optimize field of view** may adjust the camera FOV. Rectilinear errors if
the panorama crosses its horizon, and cylindrical errors at a pole; choose
equirectangular for those scenes. Under **HDR / exposure fusion**, **Exposure
fusion** blends each complete bracket of consecutive different exposures, while
**HDR (Debevec + Reinhard)** calibrates a response and tone-maps it. Set the
number of consecutive exposures per bracket, exposure times (0 uses each file's
metadata when available) and the fusion width/target or HDR exposure
scale/white point. Debevec requires a positive exposure time for every source
and exposure compensation Off; HDR cannot be combined with linear-light
blending, and incomplete brackets or incompatible photometric combinations
report an error.

**Preview blend** captures the current global plan and renders a bounded
preview (up to 1024 pixels on the long edge). **Show seams**, **Show image
numbers** and **Show excluded areas** overlay the chosen labels and mask, and
selected control points appear as markers. Source opacity affects only the
workspace preview and never enters the exported blend. X/Y, scale and rotation
define the actual placement in planar mode. For an unmatched panorama image,
adjust its camera yaw, pitch, roll and field of view instead.
The preview opacity value is retained across
imports and layout changes. Any source, placement, mask, control-point or
editor change invalidates the plan, so press **Preview blend** again after edits
or optimization. Save reuses the current captured plan only when sources and
settings are unchanged; otherwise it plans and renders again. Output controls
choose Automatic, 8-bit or 16-bit depth (Automatic is 16-bit for any 16-bit
source or HDR, otherwise 8-bit), a 0.01–10 output size, normalized crop,
Automatic/On/Off transparency (Automatic enables it for panorama projection,
masked or RGBA sources, and an uncovered canvas), and a color-profile policy:
Automatic preserves a common profile and converts mixed profiles to sRGB, sRGB
converts, keep first copies the first tag without converting pixels, and no
profile omits the tag. Mixed 16-bit profiles cannot be converted by this
backend, so select keep first or no profile — there is no silent 8-bit
reduction. **Keep source EXIF** embeds sanitized
camera/lens/exposure/date/GPS EXIF from the first source, dropping pixel-layout
tags and resetting orientation. Save as TIFF or PNG. TIFF uses disk-backed
tiled composition, strip-based writing (4 MiB target strip, one full output row
minimum) and BigTIFF when needed; large PNGs are encoded in row bands. Allow
adequate temporary and output disk space and RAM: the full-size output is
streamed to disk, but planning and previews still use memory. Enough shared
features or manual points are required, and no image-only method infers
physical scale. Source images and settings are frozen while an operation runs;
use **Cancel** and wait for it to finish.

**Edit → Reset Current Stack…** restores the imported images, their original
order and import scale, including frames later removed or excluded. It clears
all outputs (including unsaved work), labels, scale bars and the watermark on
every target, ROI and registration state, and the undo/redo history. The confirmation defaults
to Cancel. Files on disk are not deleted. Wait for active operations to finish
before resetting; an empty stack cannot be reset.
After reopening a project, reset uses the frames and loading scale saved in
that project. Frames deleted before the project was saved are not in the
project and cannot be recovered by resetting.

---

## 4. Importing Images and Video

### Opening an Image Folder

1. Go to **File → Open Folder** or press `Ctrl+O`
2. Select a folder containing your image sequence, or type or paste its path
   in the address bar and press Enter to navigate before choosing it
3. Choose a loading percentage or a 2K, 4K or 8K preset based on the first
   frame's longest side (100% = full resolution)
4. Click OK to load the images

File and folder selection uses the system dialog and its navigation controls.

The status bar shows processed file counts while loading. The count may stay
unchanged while one RAW file is being decoded; cancellation takes effect after
the current file finishes decoding.

Tick **Don't ask again** to reuse the chosen percentage for later imports;
presets are not recalculated for later images. **Settings → Advanced Settings → Startup & updates → Ask for
Loading Scale on Import** turns the prompt back on. A loaded stack can still be
downscaled later with **Edit → Resize**.

The software will automatically:
- Detect and sort image files by filename
- Display the first image in the source panel
- Populate the source images list with all frames

Hover over a source frame to see its working dimensions, file path, original
dimensions and available camera metadata. File details load in the background.

### Opening Selected Image Frames

Choose **File → Open Image Frames…** and select the image files to include.
You can type or paste a folder path into the address bar. After choosing the
loading scale, the selected files form a new stack in natural filename order.

The Windows installer can add **Import into OpenFocus** to File Explorer. Select
a folder or multiple image files and choose the command; on Windows 11 it may
appear under **Show more options**. Imports append to an existing stack.

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

A project behaves like a document: the title bar shows its name with a `*` while
there are unsaved changes, and closing the app or opening another project asks
before discarding them. Source images are referenced, not copied.

### Processing Templates

A processing template carries settings, never source paths. **File → Save
Processing Template…** writes an `.oftemplate` containing the current fusion
method and parameters, registration method and parameters, per-method
brightness options, thread count, tile settings, registration downscale width,
StackMFF AI batch size, sharpening, reverse order and the sensor-denoise flag.
A defect-mask path is deliberately not stored.

**File → Apply Processing Template…** applies such settings to the stack that is
already loaded; it does not replace the current sources. The picker also accepts
a `.ofproj` project (settings are extracted without requiring its sources) and a
recipe `.json` that contains project settings. Applying a template marks the
project dirty and can change algorithm parameters and resource settings, so
review the right panel before rendering. Recipes remain the headless format and
do include a defect-mask path plus a fingerprint.

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
The pixel size (µm/px) is auto-detected from TIFF or Linear DNG metadata —
ImageJ's `unit=micron` convention, OME-TIFF's `PhysicalSizeX`, or an explicit
centimetre resolution unit — so microscopy stacks (including multi-page
TIFFs) calibrate themselves. DNG export writes the result's calibration back
into the file, so re-importing the Linear DNG keeps the same physical scale.
Printer resolution (72/300 dpi and friends)
is deliberately *not* treated as calibration, and a manual value you type
always wins over detected metadata: a plausible-looking wrong bar in a
figure is worse than none. Choose the corner
and color (Auto inverts against the background); the dialog shows a live
preview. With the bar enabled, every export (single image, folder stacks,
multi-page TIFF, GIF, export-all, batch) carries a publication-style bar:
a rounded 1-2-5 length (~12% of the image width) with stroked, readable
text. The CLI supports it via `--scale-bar` (auto calibration) and
`--scale-um <um/px>`.

When the selected target has no usable metadata, the dialog starts in manual
calibration mode and waits for a valid pixel size.

The preview draws at the target's native pixel geometry before fitting it to
the viewport. Fit, 100%, pan and frame navigation change only the view. Apply
commits one undoable edit; Cancel preserves the previous calibration. Historical
outputs retain their own calibration and style when another source stack loads.

Remove it again from **Tools → Scale Bar → Remove scale bar**: *From the
source stack* and *From the result* narrow the dialog's "Apply to" scope to
the remaining canvas, while *From everywhere* turns the bar off. Both
canvases and the A/B wipe redraw immediately; removing a canvas that never
showed the bar leaves the setting unchanged.

### Watermark (Visible Attribution)

**Tools → Watermark → Add or edit…** burns a visible, tiled text mark into
exported copies. Choose the text (up to 80 characters), font, size (1–12% of
the image short side, default 3.5%), opacity (5–90%, default 28%), white or
black ink with an automatic opposite-colour outline, angle (−60° to 60°,
default −25°) and density (sparse / standard / dense). The pattern is laid
out in the rotated text coordinate system, anchored at the image centre and
scaled from each output's own pixel rectangle, so equal-ratio resizes keep
the same relative size, spacing and phase. Source files and in-memory arrays
are never modified.

For a new watermark, **Add ©** is selected and places the symbol before the
text. Turn it off or choose the position after the text. Existing watermarks
retain their original text and appearance.

Scope is independent per target: source frames, registered frames and
fusion results (defaults: source + result), each keeping its own text and
style. Editing one scope never rewrites another; checking several applies
the same draft to each checked scope. The text is stored only in an explicitly
saved project or the local recovery snapshot, never in global preferences.
The editor previews the real
selected frame or output and shows its pixel size and bit depth; switching
the preview does not change the scope. Apply commits one undoable edit;
Cancel, Esc and closing the window leave the committed configuration
untouched. A newly loaded source stack does not inherit the previous stack's
watermark text.

Every GUI export path burns the frozen configuration at export start: single
save/Save As, output-batch and drag-out, source/registered frame folders,
multi-page TIFF, GIF, GUI batch output and aligned stacks, and representable
Linear DNG. 8-bit and 16-bit images keep their depth; blending uses the
original 16-bit values and leaves every pixel outside the anti-aliased mask
bit-identical. PNG alpha is preserved and transparent pixels stay
transparent. Set the angle to 0 for horizontal rows.

Already-exported batch results whose pixels contain a watermark are marked
in the output tooltip and cannot receive another one. Layered TIFF cannot
include the live watermark and is blocked while one is active. If the text
cannot be drawn legibly (tiny image, no installed glyph coverage, text too
long for the chosen size), the dialog and the affected export report the
reason instead of writing a file without the mark.

### Depth Map (Focus Position)

**Tools → Depth Map…** estimates focus positions using the selected method's
activity measure. This is not a measured surface or an exact pixel-source map:
pyramid/wavelet methods estimate activity and the StackMFF variants report an
estimated frame position. Choose either the current checked frames or the selected output's
recorded source. Missing historical provenance is never replaced with the
current stack. The source and texture stay bound to that snapshot.
Depth maps are unavailable for transparent PNG stacks, in the GUI and the CLI.

All seven depth-capable methods — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C,
StackMFF V2 and StackMFF V3 — require an explicit **Compute** click. StackMFF V1
has no depth map. Opening the dialog or changing the measure,
source, registration or parameters never starts computation. The parameter editor
appears directly below the measure selector: DCT median kernel and GFG initial kernel
use odd values 1–51 (default 7). OpenFocus B uses Radius, Smoothing and Feather;
OpenFocus C's depth export uses Smoothing without a radius. DTCWT decomposition levels
are editable here as a local draft, initialized from the source parameters. StackMFF V3
shows its own AI output mode (Soft blend by default, or Hard pick) and an explanation
instead of a numeric kernel: the map reports a continuous frame position, not a
physical distance. Soft uses the probability-weighted position; Hard uses the most
probable frame per pixel. StackMFF V2 has no parameters and reports the focal
layer its network samples.
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
- **Index TIFF and alternative image TIFF**: export the current depth estimate as
  unsigned 16-bit frame indices or as an image assembled from source pixels. Index
  values 1–N refer to the exact frame order used as input to the depth calculation,
  after any registration and reordering; 0 marks invalid pixels. At most 65,535
  frames are supported. Fractional estimated positions are rounded to the nearest
  frame, with halfway values rounded upward. These indices estimate focus position;
  they are not the true pixel provenance of a fused result. The alternative image
  selects the pixel from the corresponding depth input frame at each valid
  location, with invalid pixels set to zero. When any input is 16-bit, 8-bit
  values are promoted to 16-bit by multiplying by 257. Exports use the depth
  map's ROI and preserve invalid areas as zero.
- **Physical depth**: enter Z start, positive step, unit and acquisition direction,
  then confirm the original capture mapping. Switching source clears this
  confirmation; confirm the capture order again. Excluded frames keep their gaps.
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
methods reuse the same memory-safe tiling as AI fusion, so deep stacks
work. The CLI equivalent is `--depth-map [path]` with `--depth-colormap`; it
writes only the colorized map (default `<output>_depth.png`), works only for
the depth-capable methods above, and is rejected for transparent or grouped
input. The GUI additionally exports raw depth, frame indices, alternative
images, physical depth and 3D models.

### Crash Auto-Recovery

OpenFocus continuously keeps a snapshot of your working session (stack,
settings, labels, current frame). If the application closes unexpectedly — a crash, power loss, or a forced kill — the next launch detects it and
offers **Restore Session** to return exactly to where you were. Clean
exits never show the prompt. Recovery data lives next to your settings
(portable installs keep it beside the executable, so it travels with you).

### Recent Files

Every stack you open is remembered under **File → Open Recent** (up to 8 entries, numbered in the submenu). Click an entry to reopen it; there is no global keyboard shortcut for recent files. Reopening the same stack moves it to the top; entries whose files no longer exist can be removed with a prompt, and **Clear Recent List** empties the menu.

### Drag and Drop

You can also drag a folder directly onto the application window to import images. Dropping more files onto an existing stack appends them (after a size check).

### EXIF Orientation

Camera JPEG/TIFF files with an EXIF Orientation tag are rotated/flip-corrected
automatically during import, so portrait shots load upright.

### 16-bit Sources

16-bit PNG/TIFF inputs are detected automatically and preserved end-to-end:
fusion runs at full depth and results can be saved back as 16-bit PNG/TIFF
(JPG/BMP exports are converted to 8-bit, as those formats cannot store 16 bits).

Camera RAW files are developed to 16-bit (camera white balance and the sRGB color space and transfer curve)
with auto-brightening disabled, so every frame of a stack gets the same
exposure mapping; per-frame auto exposure would make brightness flicker
between frames. When prompted, choose **Standard color image** for normal
fusion and JPEG/PNG/TIFF export. Save as TIFF or PNG to keep the 16-bit data.

For a camera-linear workflow, open a folder or files containing only camera
RAW frames and choose **Camera-linear RAW (Linear DNG)** when prompted. DNG files are also
accepted, including DNG files named `.tif` or `.tiff`; ordinary TIFF images are not camera RAW.
Support for a particular camera depends on the installed LibRaw decoder. All
frames must share camera calibration and white balance; unsupported or
incompatible files are rejected. There is no need to convert vendor RAW files
to DNG before importing them.

Align and fuse the stack as usual, then select the result and choose **Save As →
Linear DNG** to continue editing its color and white balance in a compatible RAW
editor. The fused Linear DNG contains demosaiced camera-linear RGB pixels, not
the original sensor mosaic. OpenFocus retains 16-bit camera-linear pixels during
alignment, fusion and retouching and displays developed previews. To save the
developed appearance at 16 bits, choose ordinary **TIFF** in Save As. PNG and
other ordinary exports also use the developed appearance. When the camera profile
allows it, DNG annotations are burned into the exported pixels. If an annotation color cannot be represented
in the camera color space, DNG export fails; choose a different color or export
a developed TIFF or PNG. Regular RAW import continues to produce developed sRGB.
For camera-linear Bayer RAW at a loading scale of 50% or less, OpenFocus uses
half-size decoding before the final resize to reduce memory use. Its pixels can
differ slightly from full-size decoding followed by downsampling; decoding still
requires workspace based on the native sensor dimensions.
For DNG inputs with complex camera profiles, the saved Linear DNG retains the
supported color profile metadata for a DNG-aware editor. OpenFocus uses a simplified D65 matrix
for its preview and developed TIFF/PNG; it does not apply DNG profile transforms
or baseline exposure. Their appearance may therefore differ from that editor.
For a complex profile, annotations and scale bars cannot be baked into Linear
DNG because their colors could be incorrect in a RAW editor. Remove them to
save the image as DNG, or export TIFF/PNG with the annotations instead.

RAW decoding still requires full-resolution working memory even when the stack
is loaded at a reduced scale.
If the camera model cannot be identified, only a single RAW can be opened;
its DNG uses a generic camera identity and does not preserve the manufacturer's
model or proprietary profile.

### Color Profiles

Single-image PNG, TIFF, JPEG and WebP exports retain the RGB ICC profile when
every source frame has the same embedded profile. Mixed or untagged stacks are exported
without an ICC profile. OpenFocus preserves the profile; it does not convert
source pixels into a common color space or color-manage the preview. Convert
mixed-profile images to a common color space before loading them. Regular RAW
import is developed to sRGB and uses a generated sRGB profile rather than a
camera ICC profile. Native RAW mode uses camera calibration for its developed
preview and Linear DNG export.

For a multi-page TIFF, all pages in the file must have the same valid RGB
profile; otherwise its frames are treated as untagged. Video frames are also
treated as untagged. Appended frames and the frames selected for a render all
contribute to its profile, which stays with that result for later exports.

### Clearing the Stack

To remove all loaded images, go to **File → Clear Stack** or press `Ctrl+W`.

---

## 5. Viewing and Navigating Images

### Navigating the Image Stack

- **Slider**: Drag the slider below the source panel to scroll through frames
- **File List**: Click any filename in the right panel's source images list;
  while the list has focus, Up/Down move the selection and the source panel
  follows
- **Keyboard**: There is no global previous/next-frame shortcut; focus the
  source list first

### Reviewing similar frames

Click **Review frames** in the source header, immediately left of **Select all**, to find adjacent frames
whose pixels are nearly identical. The button changes to **Cancel review**
while analysis runs. Each suggested row names its reference frame and difference;
inspect both before using the source checkbox to exclude a frame from fusion.
Suggestions never change checkboxes or delete files. This conservative
comparison can miss similar frames; changes confined to a small focused region
are not averaged away. Results are discarded if the stack changes.
Suggestions are shown only when both the suggested frame and its reference
are included; changing a checkbox updates the displayed suggestions.

Clearing the stack removes review status and disables the review button.

### Zooming and Panning

- **Zoom In/Out**: Mouse wheel over the source or result image, including with `Ctrl` held
- **Result zoom controls**: In the single-result view, use **−** and **+** below the image (or focus a button with Tab and press Space). The value is relative to the fit-to-window size; the buttons are disabled when no result is displayed or the corresponding limit is reached. Zooming keeps the view centre in place where pan bounds allow.
- **Magnifier**: Hold the left mouse button over either image. While holding it,
  use the wheel to change magnification or `Ctrl` + wheel to resize the square.
  The size is retained between presses, independently for each panel.
- **Fit to Window**: Middle-click the image
- **Pan**: Right-drag or hold Space and left-drag

For large source images, the magnifier starts at native 100%; its percentage
describes device pixels per original pixel. The main view can reach 800%.
JPEG and RAW stacks open with bounded previews when available. Original pixels
load on a worker when closer inspection needs them, and the view receives only
the visible region. The selected original is retained for repeated inspection;
changing or clearing the source releases it. The first original decode still
requires enough RAM and can take longer than opening its preview.

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

Historical outputs use their own source frames; they cannot follow a current
source stack that does not match the output.

If the selected output has no available source image, Wipe reports that the
source is unavailable. When the position of a registration crop is unknown,
the source and aligned images are fitted separately while preserving their
proportions, and Wipe reports that the crop position is unknown.

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

- Bounded similarity alignment (translation + rotation + scale) with auto-crop.
- Searches sampled image differences and warps with Lanczos interpolation.
  The translation search range defaults to 3% of the frame size and can be
  changed with `--align-shift-limit`. Successive estimates can accumulate beyond
  that range; it is not a guaranteed rejection threshold.
- For non-rigid or extreme geometry, pick ECC or Homography instead
- **Rejected frame handling** defaults to continuing with the unaligned frame
  when estimation fails. Choose **Stop on rejected frame** to fail the render
  or batch item instead. Frames are numbered from 1 in the processed stack;
  geometry defaults to frame 1 and successive estimates follow the adjacent
  aligned frame. Low texture, repeated patterns, occlusion, large motion, or
  focus changes can still produce an inaccurate match without an error.
  Inspect the result.
- In the CLI, select the same policy with `--align openfocus --align-param
  'failure_policy="fail"'` (use shell-appropriate quoting). Batch items use
  the selected policy independently. The default is `"continue"`.
- Set **Reference frame** to a 1-based number in the included, processed stack
  order to anchor geometry to that frame. Frame order is preserved; brightness
  equalization uses its own reference. Every batch stack must contain that
  number or the item fails. CLI example: `--align openfocus --align-param
  reference_frame=3`. Choosing a reference cannot correct parallax or a poor
  match; inspect the result.

Alignment reports identify the reference and each frame's outcome. A fallback
means the method could not establish the alignment or depends on an earlier
failed estimate; the reported reason distinguishes weak estimates and motion
outside the limits when known. Quality values are method-specific, not a
guarantee of accuracy.
Transforms map source coordinates to the registration stage's reference canvas
before the common crop; composite methods report their stages separately.
Homography and ECC also accept the opt-in `failure_policy="fail"` registration
parameter to stop when their algorithms detect failure. The default remains
`"continue"`; inspect results even when alignment succeeds.

To inspect a render, enable **Settings → Advanced Settings → Registration →
Collect alignment diagnostics** and render again. This recomputes alignment
instead of reusing cached images. The completion
status highlights fallbacks; open **Settings → Advanced Settings → Registration → Alignment diagnostics** for the
per-frame report. It shows available failure reasons, method-specific scores,
match counts, referenced frames and transforms. Reference numbers and names
refer to the original source list, including when frames are excluded. Batch
and grouped renders do not provide this report.

### Enabling Registration

1. In the right panel, locate the **Registration** group
2. Pick one method from the dropdown: **No alignment**, **OpenFocus (fast)**,
   **ECC**, **Homography**, or **Homography → ECC**
3. The defaults work without further adjustment. For OpenFocus (fast), adjust
   the reference frame and search limits under **Settings → Advanced Settings → Registration**.
   Set **Rejected frame handling** there for all registration methods.
4. **Collect alignment diagnostics** now lives in Advanced Settings (see above)

### When to Use Registration

- Your images have visible misalignment
- Images were captured handheld
- The scene has parallax (closer objects shift between frames)
- Results show ghosting or doubling artifacts

### ROI (Region) Rendering

**Preview ROI** prepares the included frames with the current registration
method and its parameters, and uses the current fusion method's exposure-matching setting
to choose the brightness reference. With **No alignment**, ROI selection uses the original
frames directly. Changing the stack, included frames, registration method or
parameters, downscale width, frame order, sensor cleanup, or exposure-matching
setting discards the old ROI stack; entering ROI again prepares a new one. The
selected rectangle is then fused by the normal render path.

### Registration Settings

Access additional settings via **Settings → Advanced Settings → Registration**:
- **Downscale Width**: Control preprocessing resolution (default: 1024px)
- Lower values = faster processing, potentially lower accuracy
- Higher values = slower processing, potentially higher accuracy
- **Collect alignment diagnostics** and **Registration cache** sit in the same
  section

---

## 8. Image Fusion Methods

OpenFocus offers ten fusion algorithms (plus an alignment-only mode). A fresh
desktop profile uses **OpenFocus B** with **ECC** registration; both are marked
as recommended. Existing user selections are preserved. CLI defaults are separate.

### Guided Filter

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

### StackMFF (V1, deep learning)

- **Algorithm**: a single-scale attention network blends source luma with softmax weights; Cb/Cr come from the source frame closest to the fused luma
- **Best for**: neural fusion of a complete focus stack
- **Advantages**: no parameters; colour stays in gamut. Fused luma is invariant to frame order; chroma may differ when source luma distances tie.
- **Note**: about 156,172 parameters; larger stacks benefit from GPU acceleration

### StackMFF V2 (deep learning)

- **Algorithm**: a single-scale attention network normalizes the expected focus depth and samples the nearest source frame. Depth indices depend on stack order. **The frames must stay in their original focus order** — reversing is disabled for this method
- **Depth maps**: supported; reports the sampled focal layer
- **Best for**: continuous focus sweeps where a very small, fast model matters
- **Requirements**: ONNX Runtime and the bundled ONNX model.

### StackMFF V3 (Deep Learning)

- **Algorithm**: Pixel-wise source classification (ONNX Runtime). The
  network predicts, for every pixel, which input frame it comes from, then
  either blends the frames by those probabilities or picks one frame
- **AI output mode** (right panel, only for this method):
  - **Soft blend (weighted)** — default; each pixel is a probability-weighted
    mix of the frames. Smooth transitions, no hard seams; matches the paper's
    reported configuration
  - **Hard pick (single frame)** — each pixel comes from one source frame.
    In tile overlaps, the more interior tile wins instead of blending pixels.
    This can reduce mixed-color halos on fine edges but can also create jagged
    edges or tile seams; it does not guarantee correct occlusion. Compare a
    region and use retouching to repair local defects.
- **Best for**: Per-pixel selection or blending of focus frames
- **Advantages**: Order-invariant (frame order never changes the result),
  handles stack lengths within available resources, no per-method tuning, automatic optimization
- **Requirements**: ONNX Runtime and the bundled ONNX model.


For large stacks, leave space in the temporary directory for source frames and
any additional aligned or corrected frames. If no directory is specified,
OpenFocus estimates needs from known inputs and scratch data. If the system
temporary disk lacks space, it may consider writable fixed local disks. An
explicitly selected directory stays in the settings. If it becomes unavailable,
OpenFocus warns and temporarily uses the system temporary directory; each new
job retries the saved choice. A full selected disk is not replaced automatically.
Choose an existing writable folder under **Settings → Advanced Settings →
Performance → Temporary directory** if a storage error occurs. Free-space
checks are estimates, not reservations; space may run out later. Existing
frames remain in their original directory until released.

Before a full render starts, OpenFocus estimates the temporary space it will
need and compares it with the space left after its safety reserve. When space
looks low, **Low temporary storage space** shows the directory, the space
available and the estimate. **Choose another directory** opens a folder picker;
the folder must exist and be writable, and its space is checked again before
the render starts. You can keep the new folder for future renders or use it for
this render only. **Continue anyway** is available when the estimate does not
establish that the required storage is insufficient. When automatic selection
could use more than one volume, OpenFocus warns if all available volumes look
short of space. If their capacity differs, the result may remain unknown until
the run starts. On macOS, automatic volume discovery runs in the background
before rendering, keeping the interface responsive. A chosen directory is
also checked before rendering. Caches may be disabled and processing can still fail if storage
fills. Quick previews,
ROI renders and **Compare all** do not ask. On the command line, warnings go to
standard error and do not stop the run; batch warnings identify each folder.
Grouped fusion skips this estimate. Use `--temp-dir DIR` to choose a directory
for that invocation. The command line
does not use the GUI's Temporary directory setting.

Each neural tile uses every frame. Source frames and alignment scratch can be disk-backed, but free disk space and available memory still limit stack size. V2 spatial tiling approximates whole-image inference.

### OpenFocus A / B / C

- **OpenFocus A**: uses a multiscale focus response and blends low-detail backgrounds; ties keep the first frame, and 16-bit precision is preserved.
- **OpenFocus B**: blends neighboring frames guided by a smoothed depth map.
- **OpenFocus C**: combines source detail across a multilevel weighted pyramid.
- **Parameters**: A and B expose Radius and Smoothing; B also exposes Feather.
  The defaults are R8/S4 for A, R8/S4/F3 for B and S4 for C. A's Smoothing
  control currently has no effect, unlike the reference method. B smooths its
  depth map; C smooths its multilevel fusion weights. C has no Radius control.

ECC remains the default registration method for A/B/C even when experimental
features are enabled. OpenFocus alignment is used when explicitly selected;
saved explicit choices are preserved when allowed by the experimental setting.

Each method remembers its own settings. For example, use `--method openfocus_b
--param radius=8 --param smoothing=4 --param feather=3` in the CLI. Existing
saved values are converted to the new units when read.

**Brightness options are per method.** *Match exposure before fusion* defaults
on for every fusion method. OpenFocus A/B/C match the first frame in processing
order; other methods use the last frame. *Remove brightness drift* is optional
and defaults off. It adjusts the fused result's low frequencies toward the first
processed frame for OpenFocus A/B/C, or the last for other methods. Reversing
the stack changes processing order and therefore the reference frame. Explicit
saved choices remain in use. **Reset** clears remembered choices and restores
the method defaults. The CLI defaults exposure matching on and accepts
`--no-exposure-match` to disable it.

**Sharpen fused output** applies a mild unsharp mask after fusion, before the
result is converted to its output bit depth. It is off by default and can be
enabled in the main fusion options or batch options; the equivalent CLI flag is
`--sharpen-output`. This affects the fused image, not registered frames or
depth maps. It can make edges more distinct, but may amplify noise and halos.
It is saved with the project and remembered between sessions.

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
| StackMFF (V1) | Slow (GPU) | Not benchmarked here | Luma is order-invariant; chroma may differ on luma-distance ties |
| StackMFF V2 | Fast (GPU) | Not benchmarked here | Continuous sweeps with a tiny model |
| StackMFF V3 | Slow (GPU) | Excellent | Best quality |
| OpenFocus A | Depends on stack | Depends on stack | Multiscale focus response and low-detail blending |
| OpenFocus B | Depends on stack | Depends on stack | Depth-guided blending of neighboring frames |
| OpenFocus C | Depends on stack | Depends on stack | Source blending using the final depth map |

Open **Tools → Compare methods…** to adjust several methods independently.
Each method's name, parameter labels and controls stay together when the window
narrows. Changes update the preview automatically. **ROI preview (fast)** fuses
the selected area with surrounding context; **Full-image fusion** checks the
whole-image result and lets you inspect another area without rendering again.
ROI results can differ from full-image results because exposure statistics and
some algorithms depend on the image extent.
Full-image fusion is selected by default.

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

The workspace displays a selection hint: click a result, or focus it with `Tab`
and press `Enter` or `Space`, to select or deselect it.

Left-drag the source image to move the white inspection square; releasing the
button keeps it in place. The right panes show exactly that region from the
same aligned stack. Use the wheel to change the region's magnification,
right-drag to pan the source, **Ctrl+wheel** to zoom the source, and **F** to fit.
Switching source frames or resizing the window preserves the selection.
Cancel stops further work; closing waits for the current native operation to
return safely.


### Quick Preview (Draft Render)

Tuning parameters on large stacks is slow at full resolution. Open **Render settings**
and check **Quick preview**; OpenFocus first renders a draft
at half the linear size (never below 320 px on the long edge) — typically
several times faster, including registration. The completion message (and the
newest result's tooltip) reports **Quality: Preview**, and the draft is **not**
added to the output history. Uncheck the box and render again to produce the
full-resolution result. Preview mode does not apply to ROI renders and never
pollutes the full-resolution alignment cache.

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

1. Go to **Edit → Resize**.
2. Choose a 2K, 4K or 8K preset, or enter a scale percentage. Presets specify
   the longest side in pixels and preserve the aspect ratio.
3. Presets larger than the available base images are disabled; the available
   resolution option restores those base images without upscaling.

### Transformation Order

Transformations are applied to the entire image stack, maintaining alignment between frames.

---

## 10. Retouching, Labels and Annotations

### Retouching the Result (Edit → Retouch Result…, `Ctrl+T`)

Fusion can still leave a local defect — a halo around a high-contrast edge,
a bristle taken from the wrong frame, noise in a flat area. The workspace
copies pixels from the exact frames that produced the selected result:

The workspace opens as soon as the result and any available thumbnails are
ready. Thumbnails use existing source previews where present, and the selected
frame's thumbnail is replaced with its processed pixels once the frame resolves.
Only the source frame you select is resolved, in the background; at most two
resolved frames stay cached. Some actions still read every frame on demand, for
example **Auto background**, checkpoint save/load, or turning on the sharpness
reference.

1. Select one output and click **Retouch Result…** in the output header, or press
   `Ctrl+T`. You can also right-click an output and choose the same action. The
   menu and shortcut require one selected output.
2. The source frame is on the left, the result on the right. Both share zoom
   and position: wheel zooms, right/middle-drag (or Space+drag) pans, `F`
   fits, `1` shows 100 %.
3. Pick a numbered frame, drag the frame slider, or press `,` / `.`. The
   selected source loads in the background; its thumbnail appears when it is
   ready. You can navigate or close the workspace while it loads. Painting is
   available once the selected frame is ready. A stroke is locked to that
   frame and the current brush settings until the stroke ends.
4. Choose **Source Brush** (`B`) or **Restore Brush** (`E`). Diameter,
   hardness and opacity have both sliders and numeric fields; `[` / `]`
   adjusts the diameter. Hold `Ctrl` and scroll up to increase the diameter or
   down to decrease it; the wheel alone zooms. The restore brush returns pixels
   to the image you opened, not to an older output.
5. Hold `Tab` to see the result before this session and `Alt` to peek at the
   source frame. The sharpness reference is optional, covers only the source
   canvas and uses one shared scale for every frame.
6. `Ctrl+Z` / `Ctrl+Y` undo and redo whole strokes. **Restore All…** is one
   undoable step. **Apply as New Output** adds an entry named
   `<name> · Retouched N` and selects it; the original stays untouched. The
   new entry is marked *Not exported* until you export it from the main
   window, and OpenFocus protects it from automatic history eviction and
   from quitting without a decision.

**Open shortcut help**, next to **Before** in the toolbar, lists navigation,
frame selection, brush and editing shortcuts. Hover over a control for its help.

Apply the retouch as a new output, then select it and choose **Save As**. Select layered TIFF in the format list to save the result with its source frames.
Live labels and scale bars are not included; unsaved annotation changes remain
protected. An active watermark blocks this export; remove it or export a normal
image/image stack instead. Source geometry and validity masks are retained. The result and
sources must be opaque RGB or grayscale images with matching 8-bit or 16-bit
depth, and each source rectangle must fit within the result. Export supports
up to 128 sources; source names are limited to 255 characters and
cannot contain NUL or characters outside the Basic Multilingual Plane, such as
many emoji. It refuses unsafe contexts and jobs estimated to exceed 512 MiB
of working memory or the available RAM budget, reporting the failure instead
of writing an incomplete export.
GIMP 3.2.6 on Windows imports the 8-bit and 16-bit layers, including their
offsets and transparent registration fill. The result is visible and source
layers are hidden; GIMP also adds a visible Background layer. Other editors
have not been tested.

For a local tonal correction, select **Exposure brush**, set its amount in
hundredths of a stop (100 doubles the pixel values, -100 halves them), and
paint on the result. The preview updates as you paint. Each stroke adjusts the
current result once, even where its dabs overlap; another stroke builds on it.
Use the brush opacity and hardness to control the blend. Highlights can clip
when brightening. Hold **Before** to compare with the result as opened, press
Escape during a stroke to cancel it, or use undo and redo after the stroke.
These are pixel edits, not adjustable layers. Apply as New Output and export
the new entry to save the edited pixels; a project does not store those pixels.

**Result adjustments…** changes the entire current retouch result. Controls are
on the left and the sampled preview is on the right. Adjust exposure, contrast,
highlights, shadows, saturation and color temperature; enable **Wipe comparison**
and drag the divider to compare before and after. The wheel zooms, dragging the
image pans and double-clicking resets the view. **Reset Default** clears the
adjustments. Extreme values can clip highlights or shadows.
**OK** applies the adjustment as one undoable edit; **Cancel** leaves the
result unchanged. During processing, Cancel requests a rollback. The full
image must fit within the retouch history's memory limit for this undoable
operation. Publish and export the result as described above.

Provenance travels with the output: the aligned frames, ROI crop/paste
offsets, DCT edge trimming and bit depth are recorded when the result is
rendered. Batch files, legacy results and align-only folder entries have no
recorded source, so they are listed as unavailable instead of being guessed
from the current stack. Unapplied strokes and undo history are session-only;
saving a project does not store them.

For an existing image file, choose **Edit → Retouch Image Files…**, then select
the result and one or more source images. The sources must already be aligned
to the result on the same pixel grid; matching dimensions alone cannot prove
alignment. Single-page PNG, TIFF, JPEG and BMP images with matching upright
dimensions, channels and 8-bit or 16-bit depth are accepted; EXIF orientation
is applied when loading. Alpha and multipage images are not supported. The
selected frames are user supplied, not recorded render provenance. Apply
creates a separate unexported output; save it from the output list. The chosen
files are protected against overwriting during this session, including depth-map
and 3D mesh exports.
As with other results, project files do not store these pixels or brush history.

### Replacing the Background (Auto background)

Stacking sharpens the background along with the subject. The background of a
stacked image is often a patchwork of different blurs, with a halo around the
subject. **Auto background…** in the retouch workspace replaces it with one real
frame from the same stack. The subject stays exactly as stacked. The background
is derived from a frame and the stacked result; processing may reconstruct,
adjust, and blend its pixels.

1. Open the result in the retouch workspace and click **Auto background…**.
2. Choose the background:
   - **Automatic** (recommended) uses the soft background, unless the stack
     recorded much of the background in focus. Then it uses the sharp one.
   - **Soft background** uses the frame where the background is blurriest.
   - **Sharp background** uses the frame focused farthest away.
3. Leave **Clean edge halos** on to remove halos along the subject's edges. It
   never goes beyond what the frames recorded. Turn it off to keep the stack's
   own edge look.
4. Tick **Keep soft parts attached to the subject** when parts of the subject
   sit just outside the depth of field, such as the knob at the end of a spine.
   It can also keep background objects next to the subject as stacked, so check
   the result.
5. Click **Replace background**. A 20-megapixel stack of 15 frames takes about
   a minute and needs about 3.5 GB of free memory. **Cancel** stops it.

The change is one undoable step, and strokes you painted on the subject before
are kept. Use the **Restore Brush** to bring back stacked pixels wherever you
prefer them, for example an out-of-focus grass blade in front of the subject.
When the subject fills nearly the whole frame, as in a ground-level forest
scene, the image is left as stacked and a note says so.

If you open Retouch on an output that was already retouched, Auto background
analyzes the stacked result it came from and lays the new background onto your
retouched image. Brushing on the subject is kept, and an earlier background
replacement is never analyzed again, so running it twice does not build up
halos.

After a run, **Show missing data** tints the places no frame saw cleanly. These
are the places to check or retouch by hand:

- magenta: background no frame recorded without the subject in the way
- cyan: subject hidden from the camera
- yellow: subject beyond the focus range of the stack
- slate: areas too smooth to judge

**Save missing-data map…** writes the tinted preview and one black-and-white
mask per colour. Auto background needs the frames that made the result and is
not available for region (ROI) results. New background checkpoints restore the
exact image, reachable history and diagnostics without recomputing the background.

The command line does the same without the GUI: add `--auto-background` to a
fusion run (see [Command Line Interface](#14-command-line-interface)). It uses
the same defaults (Automatic, edge halos cleaned, soft parts not kept), and
`--save-stacked` and `--missing-data-map` also write the stacked result and the
missing-data map.

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

**Enable experimental features** in **Settings → Advanced Settings** is off by
default and is remembered for this installation. With it off, desktop fusion
offers OpenFocus A/B/C and No fusion; registration offers ECC and No alignment.
Other fusion and registration methods and image stitching remain visible as
experimental but cannot be used. Enable the option and click **OK** to unlock
them; required dependencies may still limit availability. The experimental
features are under development and may have performance, quality or availability
limitations. Projects and templates do not override this preference. Restoring
default settings turns it off again.

Open **Settings → Advanced Settings** for performance, registration, sensor
cleanup, output naming, StackMFF AI, diagnostics and startup preferences. Changes remain
drafts until **OK**; Cancel, Esc and closing discard them. Reducing the tile
size immediately updates the visible overlap limit to half a tile. The AI
section's saved batch limit is retained, but inference currently submits one
tile per call; registration width can change alignment accuracy.
Alignment-diagnostic collection is a checkbox in the Registration section.

**Settings → Advanced Settings → Share usage statistics** is enabled by
default when no preference exists; an explicit previous opt-out is preserved.
OpenFocus sends only a random installation ID, application version and operating
system to openfocus.top, at most once per day after a successful send. No images,
paths, hardware details or logs are sent. The ID is pseudonymous: reports with
the same ID can be linked. Turn the switch off and choose **OK** to stop future
reports and delete the local ID and send timestamp; enabling it again creates
a new ID. Cancel, Esc and closing discard the draft without changing reporting.
There is no onboarding choice or extra consent dialog. Closing the application
cancels future reports; an already dispatched request may still complete.

An active installation count for approximately 30 days means distinct IDs
that reported during that period, optionally grouped by version. It is an
estimate, not a count of people or all users: offline installations and users
who disable reporting are missing. Failed sends retry while the application stays
open or on a future startup. Updates retain the ID with your settings. Copying
portable settings copies the ID, so
copies may count as one installation; reinstalling retains it only if the
existing settings are kept.

Automatic update downloads run while idle and never install automatically. A completed download remains ready until you explicitly choose **Restart and Update**. Restart protects active edits and unsaved work, then writes recovery data before launching replacement. Cancelling a download or preparation waits for the current safe boundary. An existing complete, newer staged build can be installed offline.

Flat-field correction changes session working pixels without overwriting source files. Select the white reference and a preview frame, then generate the preview again after changing source or scale. Reference and working/base images must share a coordinate domain and size. Applying prepares both stacks before committing; optional reference-frame removal belongs to the same undo step. Export corrected source images when you need to preserve these pixels outside the session.

Batch processing accepts an optional white-field image in its processing options.
Choose a defocused reference at the original frame resolution; it is scaled with
the input stack and applied before registration, using the default dust size.
An unreadable reference or a size mismatch fails the affected stack visibly.
The reference is excluded from folder inputs, including grouped processing;
at least two source frames must remain for fusion. Flat-field correction cannot
be combined with transparent input. Leave the field empty to process without
correction. The CLI also accepts the reference with `--dust-map WHITE`.

Sensor cleanup is optional and runs on the original sensor-coordinate frames
before registration. In the desktop application it lives in **Settings →
Advanced Settings → Sensor cleanup**; the batch dialog has the same controls.
Select an explicit defect mask to correct known hot or
dead pixels; OpenFocus does not infer a mask from specimen images. The mask must
be a single-channel 8-bit or 16-bit image with the same dimensions as the source:
zero means unaffected and either 1 or the maximum channel value marks defects.
Review its extent before processing. Masks covering more than 10% of the frame,
or defects without an unmasked neighbor, are rejected. Keep the mask at the
original resolution and do not select it as a source image.
For GUI processing, load source frames at full scale when using a defect mask;
the mask is never resized. Sensor cleanup (including denoise) cannot be applied
to a GUI stack after a rotation, flip, flat-field correction, or other source
change: reload the original frames or disable cleanup. This also applies to
restored projects with such source histories. Pre-aligned ROI processing cannot
be combined with sensor cleanup; leave ROI mode and render the full stack.
Cleanup settings belong to the current session and saved project, not global
application preferences. Batch and CLI process their original inputs separately,
with sensor cleanup before flat-field correction and registration.
Defect masks require source files with neutral EXIF orientation. Oriented files,
camera RAW files, and sources whose orientation cannot be established are
rejected because their decoded pixels may no longer match sensor coordinates.
Use source images and a mask in the same known pixel orientation, with neutral
orientation metadata. Denoising alone does not require a defect mask.

The optional sensor denoise is independent of the defect mask and is disabled
by default. Inspect the result at full size to decide whether its smoothing is
appropriate for the specimen. Sensor cleanup preserves the source files and
8-bit or 16-bit image depth; it is separate from flat-field correction.
The CLI accepts `--defect-mask-path MASK` and `--sensor-denoise`; recipes and
the Python recipe API capture these options. Recipes record a mask fingerprint
and reject a missing or changed mask. CLI cleanup cannot be combined with
transparent input. Grouped processing applies cleanup to original frames, not
to intermediate group results.

Recipes require their output location to be outside every input folder, and
reject an explicitly selected input as output. Choose a separate output location
to protect the source images and avoid including an earlier result on a rerun.
A project's processing settings can be reused as a template or recipe even if
its original sources have moved: **File → Save Processing Template…** stores
the settings without source paths and **File → Apply Processing Template…**
applies them to a loaded stack, while `openfocus recipe create --project`
writes a headless recipe that still needs explicit `--input` paths. Templates
carry the sensor-denoise flag but not a defect-mask path; recipes store the mask
path and its fingerprint. Opening the project itself still
requires its sources and any recorded flat-field reference.

When a flat-field file reference is applied in **Tools → Dust Map
(Flat-field)** at the stack's base scale, a saved
project records its absolute path, content hash and dust size, and reapplies it
once to the original sources on open. Keep the reference file unchanged and in
place: a missing or changed reference prevents the project from opening. A
reference selected from the stack, or correction at a different working/base
scale, still requires exporting the corrected source stack before saving a
project. If the working scale changes after correction, restore it to the
correction scale before saving or export the corrected stack.

**Help → Send feedback** opens local logs. Copy copies the visible session log;
the local feedback ZIP can contain file paths, so review it before sharing.
**Review error report** shows the sanitized diagnostics that will be sent;
nothing is uploaded until you choose **Send to OpenFocus**. Automatic reports
are sent only after a session with an error when enabled. A previous native
crash requires confirmation on the next launch. Reports contain no images,
user file names or paths. See [Privacy Policy](PRIVACY.md) for details.

Each launch writes its own rotating log, so concurrent GUI/CLI jobs keep separate
records. Task IDs link parameters, elapsed time and completed/cancelled/failed
outcomes; unexpected errors include a full traceback. The feedback ZIP also
includes the current crash report and dependency versions. Use **Help → Open Logs
Folder** or the path shown on the feedback page to find the files. Logs rotate at
1 MB with three backups per process; inactive session logs older than 14 days are
removed. For detailed support logs, enable **Settings → Advanced Settings →
Diagnostics → Detailed logging** (see [Diagnostics](#diagnostics)), or set
`OPENFOCUS_LOG_LEVEL=DEBUG` before launch; `OPENFOCUS_LOG_DIR` optionally selects
a different log folder.

### Tile Settings (Memory Optimization)

Methods supporting external tiles can use them above the configured threshold.
OpenFocus A/B/C process the whole image at the default threshold. For A and B,
changing the threshold explicitly enables their external tile path; C does not
support external tiles. Their bounded row/frame
passes still reduce working memory while retaining global depth and reconstruction.

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

The status panel shows used and total physical memory when the operating
system can provide both readings; otherwise it shows `N/A`.

### Diagnostics

Access via **Settings → Advanced Settings → Diagnostics**. Both switches are
off by default, are remembered for this installation only (not in projects or
templates) and do not change the rendered image.

- **Profile rendering performance** times the stages of each following render,
  preview and comparison render. **Performance report** shows the most recent
  report with its finish time: where the time went, the stage tree in execution order with CPU time
  and peak-memory growth, tile geometry, thread counts and cache activity. A
  cancelled or failed render still reports the part that ran. **Save as JSON…**
  exports the full report; reports are otherwise kept only until the
  application closes. Collecting the report adds negligible time to a render.
- **Detailed logging** records why a render was planned as it was: resource
  plans, tile size and count, the temporary directory chosen, memory before
  and after, and the time of every tile. It takes effect immediately, including
  for the next render, and makes log files grow faster. **Open Logs Folder**
  opens the same folder as **Help → Open Logs Folder**.

Turn both on before reproducing a slow render, then attach the saved report
and a feedback package. [PERFORMANCE.md](./PERFORMANCE.md#diagnosing-a-slow-render)
explains how to read them.

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

StackMFF V1/V2/V3 use ONNX Runtime. DirectML on Windows and CoreML on macOS may accelerate inference; otherwise it runs on CPU. Provider availability alone does not prove GPU execution. Wider GPU support requires validation.


- Toggle via **Settings → GPU Acceleration**
- Unchecking forces neural methods onto the CPU — useful when GPU drivers misbehave
- The status panel shows `N/A` when no accelerator is used or available, and
  `CPU*` when an accelerator exists but the selected method is CPU-only
- The choice takes effect on the next render and is remembered

### AI Batch Size

Access via **Settings → Advanced Settings → AI**:

- The saved batch limit defaults to 2; the current execution policy caps actual
  inference at one tile per call on every device.
- Increasing this limit does not enable unvalidated GPU batching or reduce
  spatial tile size.

### Themes

Two interface themes are available: **Light (default)** and **Dark**. Switch via
**Settings → Theme**; the choice is applied immediately (the Windows title
bar follows) and remembered. First-launch users choose a theme in the
welcome dialog; a saved choice always wins.

### Portable Mode

Create an empty file named `OpenFocus.portable` next to `OpenFocus.exe`
(the portable zip ships with one) and all user data — settings, logs,
recent files — is stored in an `OpenFocusData` folder beside the app
instead of your user profile, so the installation can travel on a USB
stick. Delete the marker to switch back to the normal profile.

### EXIF Metadata

PNG/TIFF/JPEG/WebP exports keep sanitized camera, lens, exposure, date and
GPS metadata and set Orientation to 1. BMP output embeds no EXIF. If a JPEG's
EXIF or ICC payload exceeds the format limit, OpenFocus reports a clear error;
choose PNG or TIFF instead. If a source's sanitized EXIF exceeds 16 MiB,
OpenFocus refuses to load it; remove or reduce that file's EXIF first.

### HEIC/HEIF Input

iPhone photos load directly. Requires `pillow-heif`, which
`pip install openfocus` installs and the packaged builds bundle.

### Updates

**Help → Check for Updates** compares your version with the release information
at `openfocus.top`. A newer release is downloaded from the same site in the
background while the app is idle; once the build is staged, the banner and
status area offer **Restart and Update**, which applies it offline. Builds that
cannot self-update open the website download page instead. Automatic checks run
quietly at most once a day.

### Registration Cache

OpenFocus can store registration results in a `.openfocus_cache` folder
next to the source stack. Re-opening the same stack (files unchanged)
then skips alignment entirely. Toggle it in **Settings → Advanced Settings →
Registration**; stale entries are replaced automatically.

### Persisted Settings

All of the following survive an application restart automatically:
thread count, resource mode, tile parameters, registration downscale width, AI
Fusion (StackMFF V3) batch size, the GPU acceleration toggle, the registration
cache and alignment-diagnostic toggles, interface language, window size/panel
layout, the drag-out export format, the per-method brightness options,
smoothing and Radius values, and your recently opened stacks. Settings dialogs
save when you confirm them; everything is also saved when the application
closes. Preferences are stored per user in `OpenFocus/OpenFocus.ini`
(QSettings INI format).

### Interface Language

Switch anytime via **Settings → Language**, which lists all twelve packs
(English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português,
Italiano, Nederlands, Polski). The change applies immediately and is remembered
for future sessions.

### Restore default settings

Choose **Settings → Restore default settings…** and confirm the change. The
confirmation defaults to **Cancel**; cancelling preserves every preference.
Restoring resets processing and interface preferences while retaining loaded
sources, outputs, the current project, recent files, language and export folders.
Finish running jobs and close dialogs before restoring.

### Automatic Update Check

Once per day, a few seconds after launch, OpenFocus quietly checks `openfocus.top`
for a newer release. When one exists, a banner appears above the image
panels: the version, **Update now** (applies a staged build, or starts the
download), **What's new** (release notes from `openfocus.top`) and a dismiss button. Dismissing
hides the banner for that version only — the next release brings it back, and
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
- **Grouped fusion** is an optional way to process long stacks in multiple-folder
  mode. Tick **Process large stacks in groups**; the **Group size** and
  **Overlap** controls appear only while it is on. Group size must be at least 2
  and overlap between 0 and the group size minus 2. The desktop overlap starts
  at 1 and is limited to the allowed range when the size changes. Groups
  are fused in rounds using temporary disk space; the progress display identifies
  each round. Fusion, exposure matching, base correction and registration use
  each group's own frames and references, including intermediate rounds, so the
  result approximates processing the complete stack at once. Grouping requires a
  fusion method; original-frame depth, provenance and aligned-frame exports are
  unavailable. Split-folder mode is not supported, so the checkbox is rejected
  there.
  Registered groups use their first frame as the coordinate reference. Each
  registration removes a conservative border for interpolation, and later
  rounds intersect the surviving rectangles, including any single-frame tail.
  The final image is therefore cropped. Each group being registered must be
  larger than 8 pixels in both dimensions; difficult rotations or projective
  transforms may have no safely usable rectangle and produce an error. Use
  pre-aligned inputs without registration, or disable grouping, if this occurs.
- Grouped input must have matching oriented dimensions and consist of single-frame
  files with headers readable by Pillow. Camera RAW, multipage files, and files
  with alpha channels or transparency are unsupported, including transparent
  PNG stacks. Remove alpha or disable grouping to process those stacks. CLI
  dry runs check these restrictions. For the command
  line, an example is `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`.
  Grouped CLI input must be a folder and cannot be combined with `--depth-map`.
  A different stack-wide OpenFocus alignment reference cannot be combined with
  grouping. A flat-field reference is applied to original frames before
  registration; output sharpening is applied once after the final group.
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

### Batch image conversion

Open **Tools → Batch Image Converter** independently of the current stack. Add files or all supported images in a folder, choose or enter an output folder and a format, then select **Convert**. The converter starts with PNG (lossless and 16-bit) and restores the last format you explicitly chose; an example filename below the selector shows the actual extension. This tool converts individual files and is separate from fusion Batch processing, which has its own output format. Missing output folders are created when conversion starts. The list shows output paths and errors per source; the summary reports successful and failed sources. Cancel stops between operations; already completed files remain available. Closing during conversion requests cancellation and closes the window when processing stops.

Drag local image files or folders onto the queue or dialog to add them. Folder
imports include direct children only. Files are naturally sorted and deduplicated;
the import summary reports added and skipped entries. Convert becomes available
when the queue contains images and the output path names an existing directory
or a new directory. Quoted paths and environment variables are supported.
The selected format's relevant limitations appear below the format selector;
hover over the selector for the full conversion guidance.

The fixed 96×72 thumbnail sits to the left of each row. Select multiple rows and
press **Delete** or **Backspace**, or choose **Remove**. These keys edit text
normally when the output field has focus. Import and removal remain disabled
until conversion or cancellation actually finishes. Thumbnails run in a separate,
bounded queue and
use JPEG decoder scaling or small raster pages. Large RAW files use bounded
embedded JPEG ranges from TIFF-based RAW or RAF containers; a format without a
safe preview, including some CR3 files, shows a placeholder instead of unpacking
the sensor. Missing thumbnails do not prevent conversion.
Thumbnail memory admission includes the original JPEG geometry: progressive
JPEG still needs full-image DCT coefficient storage even when decoder scaling
reduces the displayed pixels. A preview exceeding the resource budget shows
the same placeholder before pixel decoding.

Choose **DNG** for RAW-to-DNG conversion without installing Adobe DNG Converter.
Supported three-color Bayer and X-Trans sensors retain the exact active CFA
samples, their black levels, white level, as-shot white balance, D65 camera
matrix, crop and orientation. Masked optical borders are omitted; active pixels
are neither resized nor developed. Samples use a 16-bit TIFF container while
retaining the original 12/14/16-bit white level. Existing DNG files are copied
byte-for-byte, preserving their profiles, processing opcodes and proprietary
metadata. Non-CFA three-color sensors use **LinearRaw**, containing camera-linear
RGB rather than a sensor mosaic. Ordinary JPEG/PNG/TIFF inputs cannot become
camera RAW by choosing DNG and are reported as errors.

LibRaw supplies the camera calibration for vendor RAW. Its Python API does not
expose every proprietary correction, spatial black table or manufacturer profile;
conversion does not promise identical rendering to the manufacturer's software.
Unknown CFA layouts, four-color sensors, non-square/diagonal layouts, missing
calibration, unequal channel saturation levels and CFA payloads exceeding standard
TIFF's 4 GiB limit fail explicitly. No Adobe SDK or converter is required at runtime.
Synthetic Bayer, X-Trans and LinearRaw samples are covered by independent LibRaw
decoding tests; support for individual real cameras depends on LibRaw.

The seven small synthetic files (Bayer and X-Trans at 12/14/16-bit sensor ranges,
plus one 16-bit LinearRaw) also completed **Adobe DNG SDK 1.6 dng_validate**
with exit code 0. That validator build disabled XMP and external lossy-JPEG
support; the generated files contain neither. The six CFA samples emit the SDK's
"Too little padding (possible interpolation artifacts)" warnings at their four
edges because masked borders were removed and ActiveArea covers the stored
pixels. Border interpolation may therefore differ between consumers. Pixels
are not fabricated or padded to hide this warning. This is validation of these
synthetic files, not certification of every camera or Adobe product.

Parallel conversion shares the application's resource budget, reserves memory
for each job's largest TIFF page, and bounds pending jobs to the worker count.
TIFF admission walks page headers once with one open file and checks cancellation
before reading or estimating each page; cancellation also interrupts this preflight.
Each job runs native codecs with one thread. Cancellation stops new admissions
and checks between pages, decode/write stages and DNG strips; native codec calls
finish cooperatively. Completed files remain available and incomplete temporary
files are removed. Output-name reservation is atomic across concurrent jobs.

For ordinary image output, camera RAW is developed with LibRaw using camera white balance and without automatic brightening. Available RAW support depends on the installed LibRaw version. Multipage TIFF is exported as separately numbered images. JPEG is lossy, and JPEG, BMP and WebP reduce 16-bit pixels to 8-bit; PNG and TIFF retain 16-bit pixels. Transparent images require PNG. LZW TIFF pages with alpha, unsupported TIFF color layouts and sample types are reported as errors. Existing files and inputs are never replaced: name conflicts receive numbered suffixes. Every image still requires enough free memory and disk space for its decoder and encoder. Safe camera EXIF and compatible RGB profiles are retained where supported; some formats or pages cannot carry all metadata.

RAW conversions to ordinary image formats use the sRGB transfer curve and embed an sRGB profile where
the output format supports it. For a multi-page TIFF, the converter preserves
each page's own valid RGB profile independently.

---

## 13. Exporting Results

### Saving a Single Result

1. Select the result you want to save in the output list
2. Go to **File → Save** or press `Ctrl+S`
3. Choose format and location
4. Click Save

Choose **File → Export All Results to Folder** and select a destination to
export every output-list entry. Image results are saved as PNG and aligned
stacks as folders. Existing files and folders are not overwritten.

In **Settings → Advanced Settings → Output naming**, an optional filename
prefix helps identify newly rendered results. The example updates as you type;
use up to 40 letters, numbers, underscores or hyphens. Leave it empty for the
original naming. Confirm with **OK** to remember it, or cancel to discard the
change. The prefix becomes part of new output-list names and their suggested
export names; existing results keep their names. You can still edit a name in
the save dialog. This preference does not rename source files or change batch
output names.


### Sharing a Result

Right-click an output and use the **Share** submenu. **Copy preview image
(JPEG)** writes the current result at full resolution, including its labels and
scale bar, to the clipboard as a real JPEG and also provides a decoded bitmap
for desktop applications. **Copy file paths** copies the last saved path for
each selected result as both text and file URLs. Unsaved or missing files must
be saved first; this command never creates a file. If a result has changed
since saving, the copied path still points to the last saved version.
**Show in folder** opens Explorer, Finder or the desktop file manager with a
matching saved file selected. If the result is unsaved, changed or its file is
missing, it creates a temporary PNG without marking the result saved. Temporary
files may be removed by the operating system; use **Save As** to keep one.
Saving a result also copies its actual file path to the clipboard.
**Create private share link** stays disabled until hosted sharing exists.

### Drag-out Export Format

Dragging results from the output list writes image files using the format
chosen under the output list's context menu (**Drag-out format**: JPG,
PNG, WebP or TIFF). The choice is remembered.

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

Large ordinary TIFF, PNG and baseline JPEG saves automatically use row bands;
no special save mode is required. TIFF and PNG retain native 16-bit samples,
including their low bits. JPEG converts 16-bit samples to 8 bits and stages
grayscale or RGBX pixels in temporary disk storage before encoding. Allow up to
four bytes per pixel of JPEG scratch space in addition to the destination file.
Its mapped pages remain subject to the operating system's memory management.
Advanced progressive or optimized JPEG requests require additional image-sized
codec memory and are admitted only when the resource budget can cover it.
Other OpenCV JPEG parameters, including restart intervals, sampling factors and
independent luminance/chrominance quality, retain their requested settings through
the original codec after admission for its complete image-sized working set.
Cancellation of that native codec call takes effect at its boundaries; an existing
destination remains intact. BMP, WebP and other legacy encoders also require
image-sized memory buffers.

Ordinary TIFF automatically switches to BigTIFF when its pixel data and metadata
cannot safely fit classic TIFF offsets. Camera, exposure, lens, valid GPS and RGB
ICC metadata stay attached to the output, including BigTIFF; source layout tags
and MakerNotes are removed and Orientation is reset to 1. Multi-page exports with
known page geometry stay classic TIFF when the complete conservative bound,
including every page's metadata and strip tables, fits its offsets. Large stacks
and streams with unknown geometry use BigTIFF from the outset, requiring a
BigTIFF-capable receiving application. Page conversion and encoding use the same
strip writer. Desktop multi-page stack exports retain their uncompressed TIFF
default. Supported bounded TIFF compression modes are
uncompressed, LZW, Deflate and PackBits, subject to the installed OpenCV codec.
PNG retains transparency; ordinary non-PNG exports require opaque pixels.
Cancellation or a failed write leaves an existing destination file intact.

---

## 14. Command Line Interface

Installing with `pip install openfocus` includes the AI model and weights, HEIC and camera-RAW decoding. On Python 3.13+, upstream `dtcwt` is omitted because its metadata requires NumPy < 2. To enable DTCWT after installing OpenFocus, run `python -m pip install --no-deps dtcwt==0.14.0`. OpenFocus installs its prerequisites (NumPy, SciPy and six) and supplies the NumPy 2 compatibility adapter. Upstream's declared constraint remains unsatisfied, so `pip check` reports it; future upstream releases may resolve this conflict.

OpenFocus can run without the GUI — useful for scripted pipelines and CI. The
`openfocus` console script (installed by `pip install openfocus`) and
`python main.py` from a checkout expose the same commands. Fusion options work
with either form; the subcommands below are:

```bash
openfocus version                       # print the version
openfocus describe                      # JSON: methods, parameters, capabilities, control commands
openfocus gui [--control] [--socket NAME] [paths…]   # start the desktop app
openfocus ctl <command> [flags]          # drive a running window (see AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # replay a saved headless recipe
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` launches detached; `--control` waits for the control channel and
prints its connection JSON. `openfocus describe` and `version` need no input.
The agent-facing workflow is documented in
[AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) and
`skills/openfocus-control/SKILL.md`.

From a Python environment (or next to the packaged executable):

```bash
# Fuse a folder with the default method (StackMFF V3)
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT with ECC registration, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 forced to CPU with a custom tile size
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explicit file list; video files are also accepted as input
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Single multi-page TIFF Z-stack with a metadata-calibrated scale bar
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Unattended: align, stack, replace the background, and keep the stacked
# result and the missing-data map for review
python main.py -i ./burst -o result.tif -a ecc --auto-background --save-stacked --missing-data-map

# Inspect adjacent similar frames without fusing or changing the input
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` accepts a folder or an ordered list of single-frame
image files. It prints one JSON object with `files` and `suggestions` arrays;
each suggestion's `index` and `reference_index` are zero-based positions in
`files`, and `difference` is the mean normalized sample difference. Add
`--json report.json` to save the same object. This review mode does not combine
with `--output`, `--output-dir`, `--json-progress`, `--dry-run`, or sensor
cleanup options.
The report compares adjacent files in the supplied input order, without knowing
which frames you may later exclude. If you exclude a reference, reconsider its
successor or review the remaining files again in their intended order.

For example, add `--align openfocus --alignment-report alignment.json` to a
single render. The JSON object contains `method`, `input`, `frame_count`, and
`entries`. Each entry's `frame_index` is zero-based in decoded input order,
before fusion sorting; multipage images expand in page order and video frames
in decode order. Composite methods include separate entries for each stage.
The report is written atomically after successful processing and cannot replace
an input or another output. Batch, grouped and dry-run modes reject this option.

| Option | Description |
|--------|-------------|
| `--suggest-duplicates` | Report nearly identical adjacent frames for manual review; does not fuse or exclude them |
| `--input, -i` | Image folder(s), video file, one or more image files, or multi-page TIFF |
| `--output, -o` | Output image path (extension sets the format) |
| `--output-dir, -d` | Batch mode: fuse every input folder, writing `<folder>.png` per stack |
| `--resume` | Batch mode: skip folders whose output already exists |
| `--config FILE` | Batch mode: JSON per-folder overrides, e.g. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (default), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Set any fusion parameter; repeatable and overrides legacy flags |
| `--align, -a` | `none` (default), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Set any registration parameter; repeatable |
| `--align-shift-limit` | `openfocus` alignment search range as a fraction of the frame size (default 0.03) |
| `--alignment-report PATH` | Save per-frame alignment outcomes as JSON for a single, non-grouped render with alignment enabled; batch and grouped renders are not supported |
| `--exposure-match` | Match mean brightness to the first processed frame for OpenFocus A/B/C, or the last for other methods (default: on; `--no-exposure-match` disables). Gains are computed before tiling. |
| `--base-frame` | Optional post-fusion step: match the output's low frequencies to the `last`/`first` frame (`none` default) |
| `--sort` | Frame order for `openfocus_a/b/c`: `asc` (default, file order), `desc` (reverse a far-to-near sweep), `off` |
| `--frame-order` | Folder input: `auto` (default) orders frames near to far by EXIF focus steps when every frame has them (OM SYSTEM focus bracketing starts mid-depth), otherwise by name; `name` always sorts by filename. Explicit file lists keep their order |
| `--sharpen-output` | Apply mild unsharp masking to the fused output (off by default) |
| `--dust-map WHITE` | Flat-field (dust map) correction from a defocused white-field shot |
| `--defect-mask-path MASK` | Sensor-coordinate defect mask repaired before registration |
| `--sensor-denoise` | Conservative sensor denoising before registration |
| `--kernel, -k` | Odd kernel size for methods that declare `kernel_size`; OpenFocus A/B use `--param radius=...` |
| `--ai-mode {soft,hard}` | StackMFF V3 output: probability-weighted blend (default) or most probable frame per pixel |
| `--threads, -t` | Worker thread upper limit; `0` = automatic (default), positive values cap concurrency |
| `--resource-preference` | `balanced` (default), `responsive`, or `maximum`; controls resource scheduling |
| `--batch-size` | AI tile batch upper limit (default 2); actual inference currently uses one tile per call |
| `--cpu` | Force CPU even if a GPU is available |
| `--downscale` | Registration feature-detection downscale width |
| `--tile-enabled` / `--no-tile-enabled` | Enable tiled fusion when needed (default: on) |
| `--tile-size` | Tile block size for large images (default: automatic) |
| `--tile-overlap` | Tile overlap in pixels (default: automatic) |
| `--tile-threshold` | Minimum image size that triggers tiling (default: automatic) |
| `--group-size` | Fuse a folder in groups of this many frames; `0` disables grouping (default). Folders only; no `--depth-map`, RAW/alpha/multipage input, or non-default OpenFocus reference frame |
| `--group-overlap` | Frames shared by consecutive groups; default `0`, at most group size minus 2 |
| `--scale-bar` | Draw a scale bar (calibration auto-detected from TIFF/DNG microscopy metadata) |
| `--scale-um` | Pixel size in µm/px, overriding metadata (implies `--scale-bar`) |
| `--depth-map [path]` | Also save a colorized focus-position map (default `<output>_depth.png`); depth-capable methods only, not grouped or transparent. In batch mode only the default path is allowed |
| `--depth-colormap` | Colormap for `--depth-map`: turbo, viridis, jet, gray, … (default turbo) |
| `--auto-background [preset]` | Replace the stacked background with an aligned frame's, as Retouch's Auto background does: `auto` (default), `soft` (blurriest background) or `sharp` (far-focus frame). Needs the aligned frames, so not with grouped fusion |
| `--background-mode` | Auto background: `clean` (default) also removes the stack's edge halos, within what the frames recorded; `faithful` keeps them |
| `--keep-soft-parts` | Auto background: keep soft parts attached to the subject that some frame focused on |
| `--save-stacked [path]` | With `--auto-background`, also save the result before the replacement (default `<output>_stacked<ext>`). In batch mode only the default path is allowed |
| `--missing-data-map [path]` | With `--auto-background`, also save the missing-data map and one 8-bit mask per class (default `<output>_missing.png`). In batch mode only the default path is allowed |
| `--json FILE` | Write a machine-readable result summary to FILE when done |
| `--json-progress` | Print one JSON progress object per step to stdout (NDJSON); prose goes to stderr |
| `--profile-performance [FILE]` | Time the render's stages for a single, non-grouped run. Prints a summary to stderr, adds the report to `--json` under `performance`, and writes it to FILE when given. See [PERFORMANCE.md](./PERFORMANCE.md#diagnosing-a-slow-render) |
| `--log-level LEVEL` | Detail of this run's log file: `debug`, `info` (default), `warning` or `error`. Overrides `OPENFOCUS_LOG_LEVEL` and prints the log file path to stderr |
| `--dry-run` | Validate inputs and print the plan without fusing |

There is no `--method none` value: the CLI always fuses. Use the desktop
application for alignment-only output.

### Processing recipes

A recipe is a portable JSON file of CLI processing options with paths relative
to the recipe. Create one from the options you would pass to a normal run, then
replay it:

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif`
captures a project's processing settings against explicitly chosen inputs; the
project's sources may have moved. Recipes reject transient controls such as
`--dry-run`, `--resume`, `--json` and `--config`, and require the output to sit
outside every input folder (and differ from an explicitly selected input). A
defect mask is stored by relative path with a SHA-256 fingerprint, so a missing
or changed mask fails the run. Project source edits (rotation, flip, flat
field) cannot be captured; export the corrected stack first.

Project capture transfers fusion, registration, brightness, order, sharpening
and tile settings into the CLI recipe and embeds GUI processing settings. Frame
exclusions, load scaling, labels and edits are not transferred; prepare the new
input stack accordingly. Grouping options can be captured from CLI arguments.

The version 1 JSON format contains `version`, an `input` array, either `output`
or `output_dir`, and an `options` object. Paths are relative to the recipe file;
move the recipe together with its input files to preserve those relationships.
Creation captures the CLI settings and resolved algorithm parameters. Unknown
fields, invalid values and unsupported versions are rejected. A dry run checks
the input without writing result files. Existing project files remain unchanged.

In the GUI, File → Save
Processing Template saves settings without source paths. File → Apply Processing
Template applies a template, project, or project-backed recipe to the current
stack without replacing its inputs, transforms, exclusions or labels. A template
is not itself an executable recipe. The GUI settings embedded in a recipe are
independent of its CLI options; if editing the JSON, keep both consistent.

The same headless API is available without importing PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # validated dictionary with relative paths
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # use False to process
```

`capture_recipe` interprets its arguments as ordinary CLI paths relative to the
current directory; `save_recipe` interprets stored paths relative to the recipe.
The recipe records settings and references, including a content fingerprint for
an explicitly selected sensor defect mask. It does not record the input pixels,
software versions, hardware, or checksums of the source stack; keep those
separately when provenance matters. Identical settings do not guarantee
identical pixels across software versions,
drivers or devices; GPU neural operations may be nondeterministic. Use `--cpu`
when a CPU execution is required, and retain the installed software and inputs.

### Replay a retouch checkpoint

Use the retouch workspace's checkpoint save/load controls with the original base
and processed sources. Auto-background sessions, including successful analyses
that change no pixels, save lossless v2 `.ofretouch` ZIP snapshots with JSON
metadata and NumPy payloads: the exact working image, reachable undo/redo tiles,
diagnostic states and source provenance. Loading v2 does not recompute the
background. Ordinary brush/tone sessions may use compact JSON v1; older JSON v1
background checkpoints remain replayable.
Legacy v1 operations recorded as using the original stacked image require that
image to be available in the retouch session with the same shape; otherwise
loading fails without changing the current edits. The legacy record does not
fingerprint that image, so a different image of the same shape cannot be
detected. The headless replay command does not supply a separate stacked image.

Source verification runs in a cancellable background operation. Before replacing
the destination, saving reloads the candidate file and checks exact equality of
the current image, reachable undo/redo tiles and diagnostic states. A source
mismatch, failed verification, cancellation or insufficient memory leaves your
current edits and any existing checkpoint unchanged.

If the checkpoint recording limit prevents another edit, apply the current
edits and start a new retouch session to continue.

Save a retouch checkpoint from the retouch workspace, then supply the original
base image and every processed source frame in the same order as the session:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Both formats still require the original base and every actual source frame used
for retouching, in session order, after alignment and any ROI crop or other
processing, with their original pixel values and bit depth. Loading reconstructs
the recorded frame offsets and alignment behavior and verifies the base, ordered
frames and validity masks; incompatible pixels, geometry or masks are rejected.
Each path must contain one image. Output must be PNG or TIFF and preserves the
base bit depth. The checkpoint and input images cannot be used as the output.

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
| `Ctrl+Alt+O` | Open Project |
| `Ctrl+Alt+S` | Save Project |
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
| `Up` / `Down` (source list focused) | Previous / next frame |
| `Mouse Wheel` | Zoom the view; while holding the left button, change magnifier zoom |
| `Ctrl + Wheel` while holding the left button | Resize the magnifier |
| `Middle-click` | Fit the image to the window |

### General

| Shortcut | Action |
|----------|--------|
| `Space` (hold, pointer over an image) | Temporary pan mode |
| `Delete` | Delete the focused list item; also clears the ROI when the pointer is over the source image |

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

- Check that ONNX Runtime and the bundled model are installed.
- Check via **Help → Environment Info**

#### Out of Memory Errors

- Enable tile processing in **Settings → Advanced Settings → Performance**
- Reduce tile block size
- Large stacks retain source and aligned images in system RAM; GPU memory does
  not replace those allocations
- Use smaller kernel sizes
- Resize images to smaller dimensions

#### Registration Fails

- Use **OpenFocus (fast)** for bounded rigid motion, or enable **Stop on
  rejected frame** to fail early instead of accepting unaligned frames
- Reduce the downscale width in **Settings → Advanced Settings → Registration**
- Ensure images have sufficient contrast and features
- Enable **Collect alignment diagnostics**, render again, and read
  **Settings → Advanced Settings → Registration → Alignment diagnostics** for the failing frames

#### Slow Processing

- Leave threads at automatic (`0`); CPU capacity, resource mode and memory
  headroom already cap concurrency
- Use Maximum resource mode for an unattended batch, or Guided Filter /
  GFG-FGF instead of an AI method
- Reduce image resolution or use **Quick preview** while tuning
- To see which stage takes the time, enable **Settings → Advanced Settings →
  Diagnostics → Profile rendering performance**, render again and open
  **Performance report**

#### Windows SmartScreen Warning on First Launch

Windows code-signing status varies by release; see the release notes. Signing
is not guaranteed. Download the build from **https://openfocus.top** rather
than a third-party mirror. If SmartScreen warns, click **More info → Run
anyway**. The first launch after unzipping may take 30–60 seconds.

#### macOS Says the App "Cannot Be Opened" or "Is Damaged"

macOS builds are not notarized. Download the Apple Silicon build from
**https://openfocus.top**. If you installed from the DMG by moving
`OpenFocus.app` to Applications, run:

```bash
xattr -cr /Applications/OpenFocus.app
```

If you installed from a terminal with `install.sh`, the app is in
`~/Applications`; run `xattr -cr ~/Applications/OpenFocus.app` instead.

On macOS 15 Sequoia and later, right-clicking the app and choosing **Open** no
longer bypasses Gatekeeper. If you prefer the UI, try to open the app once,
then go to **System Settings → Privacy & Security** and click **Open Anyway**.
Released macOS builds are for Apple Silicon (M1/M2/M3/M4); Intel Macs can run
from source.

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

### AI Fusion (StackMFF V3)

A deep learning approach using a neural network trained on large datasets of multi-focus image pairs. The network learns optimal fusion strategies automatically, providing state-of-the-art results when properly configured.

---

*OpenFocus is an open-source project. For updates, bug reports, and feature requests, please visit the project repository.*

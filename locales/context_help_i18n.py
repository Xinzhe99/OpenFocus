"""Localized plain-language and illustrated context help entries."""
from __future__ import annotations

_EN_TOPICS: dict[str, tuple[str, str, str]] = {
    "overview": (
        "Interactive Context Help",
        "Drag the ? handle in the top-right corner onto any panel, button, or parameter — or hold the H key anywhere in OpenFocus and click — to open an illustrated explanation without triggering the control.",
        "• Drag ? onto a control to inspect it in one motion.|• Click ? once to enter pick mode, then click any target.|• Press and hold H anywhere (including sub-dialogs) and click any item.",
    ),
    "registration": (
        "Image Alignment (Registration)",
        "Corrects lens breathing, focal shift, magnification change, and small camera movement between slices before focus stacking so sharp details line up pixel-for-pixel.",
        "• ECC (Recommended): Sub-pixel perspective alignment for general macro and handheld stacks.|• Homography: Fast feature-based perspective alignment when textures are clear.|• None: Use only for pre-aligned or locked-rail stacks.",
    ),
    "fusion": (
        "Focus Stacking Algorithm",
        "Selects the sharpest regions across all slices and blends them into one extended-depth-of-field photograph.",
        "• OpenFocus - B (Recommended): Depth-map weighting; crisp detail with smooth transitions and minimal halos.|• OpenFocus - A: Fast weighted average for low-contrast textures.|• OpenFocus - C: Multi-scale pyramid fusion for complex overlapping edges and hairs.",
    ),
    "align_area": (
        "Alignment Detection Region",
        "Restricts registration feature matching and motion estimation to the area where your main subject sits, ignoring moving background leaves, water, or specular reflections.",
        "• Drag a rectangle over the static parts of your subject.|• Clear the region to return to full-frame alignment.|• Ideal when background bokeh shifts differently from your subject.",
    ),
    "align_preview": (
        "Preview Alignment Result",
        "Previews how the aligned source frames line up before you commit to a full-resolution focus stack render.",
        "• Scrub the source slider while this toggle is active to inspect edge stability across frames.|• Turn off or clear the detection region if frames still drift.",
    ),
    "source_canvas": (
        "Source Stack Canvas & Loupe",
        "Displays the currently selected input slice. You can zoom, pan, draw an alignment region or output ROI, and use the magnifying loupe to inspect fine focus transitions.",
        "• Drag or scroll the slider below to step through slices from front to back.|• Hold Right-Click (or toggle Magnifier) to inspect pixel-level sharpness.|• Drag files or folders directly onto the canvas to load a stack.",
    ),
    "sharpness_curve": (
        "Per-Slice Sharpness Curve",
        "Plots relative high-frequency focus energy across each slice in the stack so you can see where focus peaks and whether tail frames are empty bokeh.",
        "• Blue line: Relative sharpness per slice (current slice shown by vertical marker).|• Red dots: Frames below the relative sharpness threshold — usually out-of-focus background frames that can be excluded cleanly.",
    ),
    "result_canvas": (
        "Merged Output Canvas",
        "Shows the stacked result or registration preview. Supports lossless pan/zoom, 1:1 inspection, pixel loupe, and split-screen comparison against source frames.",
        "• Click 1:1 or Fit on the bottom zoom bar, or scroll anywhere to zoom smoothly.|• Click items in the Output list on the right to compare past renders side by side.",
    ),
    "wipe_compare": (
        "Split-Curtain Compare",
        "Divides the output canvas with a draggable vertical curtain so you can inspect the fused result directly against any original source slice or another rendered output.",
        "• Drag the vertical handle left or right to sweep across edges and hairs.|• Use the A/B controls below the view to pick which frames are compared.",
    ),
    "render_action": (
        "Start Render",
        "Runs the selected alignment and focus fusion pipeline across all included source frames using your current parameters and Render Settings.",
        "• Click once to start; click again while running to cancel cleanly.|• Tune Radius / Smoothing / Feather first in Live Preview for fast feedback on large stacks.",
    ),
    "live_preview": (
        "Interactive Live Color Preview",
        "Builds a fast aligned proxy stack so you can tune OpenFocus A/B/C parameters, inspect source vs. fused split views, and trim substack depth ranges in real time.",
        "• Dragging sliders updates in tens of milliseconds without waiting for a full-resolution render.|• Live Preview uses proxy settings focused on OpenFocus A/B/C parameters and substack trim; final full-res Brightness/Output options come from Render Settings when you render or apply.",
    ),
    "render_settings": (
        "Render Settings",
        "Configures full-resolution workflow options applied when running Start Render or adding to the Render Queue: Quick Preview draft scaling, Reverse Stack Order, Brightness/Exposure Matching, Base Correction, and Output Sharpening.",
        "• Reverse Order flips front-to-back traversal for OpenFocus depth weighting.|• Exposure Matching & Base Correction stabilize brightness drift across slices.|• Quick Preview renders a fast downscaled draft for checking alignment and halos.",
    ),
    "reset_defaults": (
        "Reset to Defaults",
        "Restores alignment method, fusion method, algorithm parameters, and render options back to OpenFocus factory defaults.",
        "• Source images and already rendered outputs in your workspace are kept safe.",
    ),
    "render_queue": (
        "Queue, Batch & Compare Methods",
        "Opens the Queue & Batch Render menu: add the current configuration to the Render Queue, open the Queue window to run multiple jobs sequentially, or run Render All Methods to compare algorithms on the same stack.",
        "• Enqueue lets you queue multiple Radius/Smoothing presets and walk away.|• Render All Methods runs every available fusion backend for side-by-side comparison.",
    ),
    "source_list": (
        "Source Image Stack",
        "Lists all loaded input frames in focus order. Checkboxes include or exclude individual frames from rendering, and rows can be dragged to reorder.",
        "• Uncheck blurred or blank end frames instead of deleting files from disk.|• Drag rows up or down if frames were imported out of sequence.",
    ),
    "output_list": (
        "Rendered Outputs",
        "Stores all rendered results from this session. Select any output to view it, compare outputs, retouch artifacts, or export to TIFF, JPEG, PSD/PSB, HTML5 animation, or 3D formats.",
        "• Keep multiple renders in the list to compare parameter tweaks visually.|• Right-click an output for quick export, retouch, or cleanup actions.",
    ),
    "status_panel": (
        "Session & Hardware Status",
        "Summarizes loaded frame count and pixel dimensions alongside active compute device (GPU/CPU) and system memory usage.",
        "• Watch memory headroom when stacking large full-resolution RAW/TIFF sequences.",
    ),
    "radius": (
        "Radius",
        "Controls the neighborhood window size (in pixels) used when evaluating local focus contrast for each pixel.",
        "• Smaller values (4–8): Preserve ultra-fine hairs and crisp micro-textures.|• Larger values (12–25): Smooth noisy regions and reduce speckled artifacts on smooth surfaces.",
    ),
    "smoothing": (
        "Smoothing",
        "Regularizes transitions between neighboring depth choices so textured boundaries stay clean without blotchy patches.",
        "• Lower values react faster to abrupt depth jumps.|• Moderate values (3–6) suppress isolated noisy pixels along shallow transitions.",
    ),
    "feather": (
        "Feather",
        "Softens the mask boundaries where adjacent slices meet inside OpenFocus - B.",
        "• Increase slightly if hard halo seams appear between foreground and background layers.|• Set near 0–3 when crisp cutoffs along dark edges are preferred.",
    ),
    "kernel": (
        "Filter Kernel Size",
        "Sets the odd square window (N × N px) for spatial contrast filtering and local energy estimation.",
        "• Larger kernels stabilize noisy high-ISO shots at the cost of fine local separation.",
    ),
    "levels": (
        "Pyramid Decomposition Levels",
        "Controls how many frequency bands the transform pyramid uses from fine textures up to broad shading.",
        "• 4–5 levels balance fine detail with clean halo-free tonal blending on most photos.",
    ),
    "ai_mode": (
        "AI Blend Mode",
        "Chooses between soft probability weighting and hard winner-take-all frame selection in neural fusion backends.",
        "• Soft: Smoother gradients and less noise.|• Hard: Crisp single-slice selection when textures overlap cleanly.",
    ),
    "reference": (
        "Reference Frame",
        "Chooses which slice serves as the geometric anchor frame that all other slices warp to match.",
        "• Middle frame (default): Minimizes maximum scale change from front to back.|• First / Last frame: Anchors perspective strictly to the nearest or farthest plane.",
    ),
    "tiles": (
        "Tile Processing & Overlap",
        "Splits ultra-large images into overlapping rectangular tiles so peak memory stays bounded.",
        "• Keep generous overlap (e.g. 128–256 px) so tile seams remain invisible.",
    ),
    "registration_width": (
        "Registration Downscale Width",
        "Caps the working image width used while estimating alignment transforms, greatly speeding up ECC/Homography without hurting final full-resolution warp quality.",
        "• 1024–1600 px is fast and accurate for most 24–60 MP stacks.",
    ),
    "align_limits": (
        "Alignment Motion Limits",
        "Rejects erratic frame alignments if detected horizontal shift, vertical shift, rotation, or scale change exceeds realistic camera/rail motion.",
        "• Widen limits slightly for handheld stacks; tighten them to guard automated batches.",
    ),
    "align_auto_crop": (
        "Auto-Crop Aligned Edges",
        "Trims black border margins introduced where warped outer frames no longer cover every corner of the reference canvas.",
        "• Enabled by default for clean borderless renders; disable if you prefer keeping the full reference frame dimensions.",
    ),
    "align_interpolation": (
        "Warp Interpolation",
        "Selects the resampling kernel used when applying alignment transforms to source pixels.",
        "• Lanczos: Maximum sharpness preservation.|• Cubic / Linear: Slightly softer, faster warping.",
    ),
    "live_preview_histogram": (
        "Depth Histogram & Substack Trimmer",
        "Visualizes how much each slice contributes to the live fused preview and lets you drag the start/end handles to drop unwanted front or rear slices.",
        "• Drag either blue handle inward to exclude out-of-focus bokeh slices and refresh instantly.|• Click any histogram bar to jump the stack browser to that slice.",
    ),
    "live_preview_split": (
        "Live Split Curtain",
        "Splits the live preview canvas vertically between the live fused result (left) and the selected aligned/unaligned source slice (right).",
        "• Drag the center knob left or right, or double-click the divider to recenter.",
    ),
    "live_preview_size": (
        "Live Preview Proxy Resolution",
        "Controls the downscaled proxy working size inside Live Preview (Small / Medium / Large) to trade speed for finer edge inspection.",
        "• Medium (720 px) responds in milliseconds; switch to Large when checking hair halos.",
    ),
    "live_preview_stack": (
        "Live Slice Browser & Peek",
        "Steps across individual source slices in the aligned or original domain and lets you hold Space to peek directly at source pixels.",
        "• Toggle Show Aligned to verify alignment precision slice-by-slice.",
    ),
    "opt_quick_preview": (
        "Quick Preview Draft",
        "Renders a downscaled draft on full-pipeline runs so you can verify composition and parameters quickly without saving to Output history.",
        "• Uncheck before final export when you want full-resolution output added to the Output list.",
    ),
    "opt_reverse_order": (
        "Reverse Stack Order",
        "Reverses near-to-far traversal ordering for OpenFocus depth weighting when a stack was captured back-to-front.",
        "• Useful when depth order affects OpenFocus weighting or exported depth maps.",
    ),
    "opt_exposure_match": (
        "Exposure Matching",
        "Normalizes subtle frame-to-frame brightness and exposure flicker across the image sequence before alignment and fusion.",
        "• Recommended ON for LED/flash flicker or natural outdoor light changes.",
    ),
    "opt_base_correct": (
        "Base Brightness Correction",
        "Anchors final tonal brightness toward the base reference frame to prevent gradual drift along long stacks.",
        "• Helpful on very deep stacks where average luminance slowly shifts.",
    ),
    "opt_sharpen_output": (
        "Output Sharpening",
        "Applies controlled post-fusion sharpening after stacking finishes to restore micro-contrast lost to resampling.",
        "• Adjust strength slider carefully; excessive sharpening can emphasize halos or sensor noise.",
    ),
    "batch_split": (
        "Single-Folder Stack Splitting",
        "Automatically splits a single folder containing multiple shoots into individual focus stacks before alignment and fusion.",
        "• Adaptive (Auto): Detects scene transitions between neighboring frames without tuning thresholds.|• Fixed / Time / Exposure: Split by constant count, capture gap, or EV steps.",
    ),
    "batch_output_format": (
        "Batch & Watch Output Format",
        "Chooses how stacked results are written. Use TIFF/PNG for 16-bit lossless output or DNG when stacking camera RAW sequences.",
        "• DNG exports 16-bit camera-linear LinearRaw DNG files with camera color metadata preserved (requires RAW/DNG inputs).|• JPG/BMP always convert down to 8-bit; TIFF/PNG/DNG keep full 16-bit precision.",
    ),
    "watch_folder": (
        "Watch Folder Automation",
        "Continuously monitors an input folder for newly captured stacks (tethered shooting or shared network folder), aligns, fuses, and saves automatically.",
        "• Settle delay waits briefly after the last file write finishes before starting a stack.|• Processed stacks are recorded so restarting never duplicates work.",
    ),
    "retouch_toolbar": (
        "Retouch History, Checkpoints & Inspect",
        "Undo/redo brush strokes, save or reload retouch checkpoints, hold Before to inspect original output, or press F9 to jump straight to the slice under the cursor.",
        "• F9 Locate Contributing Frame finds which source photo supplied the detail at the cursor.|• Checkpoints save intermediate retouch progress without flattening.",
    ),
    "retouch_tools": (
        "Retouch Tools (Brushes & Stamp)",
        "Switch between painting pixels from any source frame, restoring original fusion pixels, dodging/burning exposure locally, or cloning texture.",
        "• Source Brush (B): Paint sharp detail from the currently selected slice.|• Restore Brush (E): Revert brushed areas back to the unretouched result.|• Exposure Brush: Lighten or darken halos locally.|• Clone Stamp (S): Sample nearby clean texture (Alt+Click origin) to heal dust or stray hairs.",
    ),
    "retouch_brush_params": (
        "Brush Diameter, Hardness & Opacity",
        "Controls stroke footprint size, edge softness, and per-stroke blending strength.",
        "• Use [ / ] or Ctrl+Wheel to resize diameter quickly while inspecting the canvas.",
    ),
    "retouch_source_options": (
        "Source Brush Precision & Local Offset",
        "Refine how Source Brush blends onto the target: restrict by color similarity, match brightness, or shift pixels to fix local parallax ghosting.",
        "• Color Tolerance protects high-contrast hairs or edges from spilling into neighboring colors.|• Auto Brightness & Local Auto-Align stabilize exposure or sub-pixel shift at the brush point.",
    ),
    "retouch_clone_options": (
        "Clone Stamp Origin & Mode",
        "Choose Area mode for continuous offset cloning or Pattern mode for repeating background texture around the Alt+Clicked origin.",
        "• Hold Alt and click on the result canvas to set the clone source point before painting.",
    ),
    "retouch_overlays": (
        "Diagnostic Overlays & Global Actions",
        "Visualize per-frame sharpness or depth ownership directly on the result canvas, run automatic background cleanup, adjust global tones, or restore everything.",
        "• Sharpness / Depth overlays paint live over the canvas without modifying pixels until you stroke.",
    ),
    "retouch_filmstrip": (
        "Source Slice Filmstrip",
        "Step quickly through every slice in the stack using thumbnails, the slider, or the frame number box.",
        "• Use ',' and '.' keys (or scroll/drag the bottom slider) to hunt for the sharpest reference slice.",
    ),
    "adv_experimental": (
        "Experimental Features",
        "Unlocks preview registration and fusion options (such as OpenFocusAlign) that are still evolving ahead of default release.",
        "• Keep disabled for routine production stacks if you prefer locked defaults.|• Changes take effect after clicking OK.",
    ),
    "adv_legacy": (
        "Legacy Features",
        "Restores older deprecated algorithms and workflows strictly for reproducing historical projects.",
        "• Recommended defaults supersede legacy options with faster processing and lower memory usage on new stacks.|• Keep disabled unless an archived project specifically requires an older method.",
    ),
    "adv_privacy": (
        "Crash Reports & Usage Telemetry",
        "Controls whether OpenFocus submits anonymized crash diagnostics and lightweight feature-usage statistics.",
        "• Crash reports help diagnose driver or decoder failures across GPUs and OS releases.|• You can toggle either option at any time.",
    ),
    "adv_storage": (
        "Working File Storage",
        "Sets the scratch folder used for decoded frame caches, aligned proxy buffers, and temporary render files.",
        "• Choose a fast local NVMe/SSD drive with generous free space when stacking large 16-bit or 45MP+ sequences.",
    ),
    "adv_naming": (
        "Output Naming & Completion Sound",
        "Defines an optional prefix, custom filename template with dynamic tokens, and a completion chime after stacking finishes.",
        "• Click any token chip to insert [METHOD], [ALIGN], [DATE], [FRAMES], or other metadata placeholders into the template.|• The live preview underneath updates immediately as you type.",
    ),
    "adv_export_color": (
        "Export Metadata & Color Management",
        "Controls EXIF metadata stripping on export and ICC profile conversion for color-accurate preview inside the OpenFocus workspace.",
        "• Keep ICC Color-Managed Preview ON when working with AdobeRGB, Display P3, or ProPhoto images.|• Enable Strip EXIF only when sharing images that must omit camera/GPS metadata.",
    ),
    "adv_raw_decode": (
        "Camera RAW Fallback Decoder",
        "Points to an optional external DNG converter CLI used automatically only when built-in LibRaw cannot decode a brand-new camera model.",
        "• Click Auto-Detect to locate an installed system converter, or click Open RAW Develop Settings to tune white balance, highlights, and exposure.",
    ),
    "adv_resources": (
        "Resource Budget & Auto-Tiling",
        "Selects CPU/GPU concurrency aggressiveness and configures optional tiled fusion bounds for memory-constrained or legacy neural workflows.",
        "• Balanced suits most workstations; Responsive keeps UI scrubbing snappy during heavy batches.|• OpenFocus A/B/C manage tile memory automatically without manual tile tuning.",
    ),
    "adv_startup": (
        "Startup, Loading & Updates",
        "Controls restoring the previous stack on launch, prompting for downscale import size, and automatic update download preferences.",
        "• Uncheck Prompt Import Scale only if you always load full-resolution 100% frames or reuse a fixed default scale.",
    ),
    "adv_gallery": (
        "Community Gallery & Danmaku Settings",
        "Configures overlay danmaku density, screen position, font size, opacity, auto-translation, and local download/cache disk limits.",
        "• Clear Cache frees local thumbnail and preview storage without deleting exported works.",
    ),
    "adv_developer": (
        "Developer Access & Pre-Release Toggles",
        "Unlocks verified developer options and feature flags inside OpenFocus.",
        "• Enter your developer key and click Verify; click Lock to clear local developer authorization.",
    ),
    "sensor_cleanup": (
        "Sensor Defect Mask & Hot-Pixel Denoise",
        "Applies a grayscale/binary dust or hot-pixel defect mask and optional raw-sensor denoising consistently across every slice before alignment.",
        "• Changes save globally and apply on the next alignment or render.|• Generate a clean gain mask from Tool -> Dust Removal Map.",
    ),
    "dust_map": (
        "Dust Removal Map (Flat-Field Gain Extraction)",
        "Builds a reusable flat-field gain correction map from an out-of-focus reference frame or current slice to heal sensor dust spots and vignetting.",
        "• Adjust Defect Scale (sigma) to capture actual dust halos without erasing fine texture noise.|• Preview Before/After side by side before applying to the stack.",
    ),
    "depth_source_method": (
        "Depth Map Source & Estimation Method",
        "Chooses whether depth is measured directly from your current input stack or recovered from a rendered Output record, then selects the focus-metric algorithm.",
        "• When re-measuring raw input slices, OpenFocus first runs alignment before scoring depth.|• AI neural depth models produce smoother macro depth gradients with fewer blocky steps.",
    ),
    "depth_postprocess": (
        "Depth Appearance, Smoothing & Physical Z Calibration",
        "Customizes colormap, depth inversion, gamma contrast, median smoothing, canvas overlay opacity, and physical µm/mm Z calibration.",
        "• Use Invert Depth if near and far planes are swapped for your capture order.|• Calibrate Z Start, Step, Unit, and Direction before exporting physical depth TIFFs or scaled meshes.",
    ),
    "depth_3d_surface": (
        "Interactive 3D Surface Preview",
        "Renders an interactive GPU heightfield mesh right inside the dialog: drag to orbit, scroll to zoom, and tune relief amplitude and grid resolution.",
        "• Switch between Textured, Shaded Clay, Depth Color, and Wireframe modes from the top toolbar.|• Turn on Auto Motion or Capture Snapshot for quick inspection shots.",
    ),
    "depth_3d_lighting": (
        "3D Studio Lighting Controls",
        "Adjusts key light azimuth/elevation, key/fill intensities, and ambient bounce light to reveal subtle surface micro-relief.",
        "• Lower elevation angles accentuate fine surface ridges and chitin textures.|• Click Reset Lighting to restore balanced studio defaults.",
    ),
    "depth_export": (
        "Depth Map, Mesh & Interactive 3D Export",
        "Saves colorized or raw 16-bit depth maps, slice indices, contrast maps, 3D meshes (OBJ/PLY/STL), anaglyph/stereo pairs, turntable videos, or offline HTML5 WebGL 3D viewers.",
        "• Share Standalone HTML5 3D Viewer (.html) when recipients need drag-to-rotate 3D inspection in any browser with zero extra software installed.|• Adjust Mesh Grid Size before exporting OBJ/PLY/STL or HTML5 3D for finer geometry.",
    ),
    "stereo_3d": (
        "Stereo 3D & Turntable Export Settings",
        "Configures eye separation, zero-parallax convergence plane, parallel/cross-eyed/anaglyph mode, or turntable animation camera motion and FPS.",
        "• Moderate Eye Separation (2%–4%) preserves comfortable depth without ghosting.|• Turn on Invert Depth if foreground pops backward in stereo glasses.",
    ),
    "html_animation": (
        "Interactive HTML5 Focus Stack Exporter",
        "Bundles aligned source slices (and optional fused frame) into one self-contained HTML5 viewer so clients can scrub through focus planes in any browser.",
        "• WebP at 1280 px usually cuts file size dramatically with crisp web sharpness.|• Include Fused Frame adds an instant comparison toggle inside the browser viewer.",
    ),
    "raw_preview": (
        "Live Camera RAW Preview",
        "Previews demosaiced 16-bit RAW adjustments in real time on a chosen stack frame before batch decoding.",
        "• Switch Sample Frame from the dropdown or browse any RAW file to check highlight and shadow recovery.",
    ),
    "raw_wb_exposure": (
        "RAW White Balance & Exposure",
        "Sets Camera/Auto/Custom white balance, Temp/Tint shifts, EV exposure compensation, highlight recovery mode, and optional auto-brightness.",
        "• Highlights -> Rebuild recovers color details in partially clipped bright petals or metallic highlights.|• Keep Auto-Bright unchecked for consistent brightness across multi-frame focus stacks.",
    ),
    "raw_demosaic_color": (
        "Demosaic Algorithm & Output Color Space",
        "Selects Bayer demosaicing kernel (such as AHD/DCB/DHT) and 16-bit linear/working color space before alignment and fusion.",
        "• AHD or DCB delivers clean fine-edge demosaicing without zipper artifacts on macro details.|• Choose sRGB for standard sharing or AdobeRGB / ProPhoto / WideGamut for grading.",
    ),
    "batch_folders": (
        "Batch Stack Folder Queue",
        "Lists focus stack folders queued for unattended batch processing. Supports multi-folder drag-and-drop or splitting one folder into multiple shoots.",
        "• Add multiple shoot folders together or turn on Single-Folder Stack Splitting when all shoots share one directory.",
    ),
    "batch_methods": (
        "Batch & Watch Fusion / Alignment Selection",
        "Selects the focus stacking algorithm and registration mode shared across queued batch stacks or incoming watch-folder shoots.",
        "• ECC + OpenFocus - B offers high alignment resilience and clean edges across unattended jobs.",
    ),
    "batch_grouping": (
        "Grouped Large-Stack Processing",
        "Processes extra-deep stacks in overlapping sub-groups before merging to keep peak memory bounded cleanly.",
        "• Overlap 1–2 frames ensures continuous sharpness transitions between neighboring sub-groups.",
    ),
    "watch_settle": (
        "Watch Folder Settle Delay",
        "Waits this many seconds after the last file write stops before triggering alignment and fusion so tethered transfers never start on incomplete shots.",
        "• Increase settle delay when shooting large RAW bursts over slower Wi-Fi or network shares.",
    ),
    "substacks": (
        "Render Overlapping Substacks",
        "Splits a deep stack into sequential slice chunks (by frames per chunk or total chunk count) with configurable overlap so you can compare depth strata or restack them cleanly.",
        "• Overlap preserves continuity at depth handoffs.|• Check Auto-Restack to merge every chunk result automatically into a final stacked output.",
    ),
    "batch_converter": (
        "Batch Image Format & Size Converter",
        "Converts folders or files across PNG, TIFF, JPG, WebP, BMP, and camera RAW sources with optional long-edge or percentage resizing.",
        "• Choose Sibling or Subfolder output to avoid overwriting original camera masters.|• Drag files directly onto the queue list or use Delete to remove selected entries.",
    ),
    "compare_methods": (
        "Compare Algorithms Selection & Execution",
        "Pick multiple fusion algorithms, adjust shared registration or per-method parameters, and run them side by side against the same stack.",
        "• Switch between Full-Frame and ROI Compare when testing parameter tweaks rapidly on high-resolution stacks.",
    ),
    "compare_view": (
        "Compare Result Grid & Detail Crop Inspector",
        "Inspects synchronized 1:1 crop tiles across every candidate algorithm alongside the reference source frame.",
        "• Drag on the left Reference Canvas or switch View Mode (Detail vs. Full) to jump across critical hairs and edges.",
    ),
    "compare_adopt": (
        "Verify Full Resolution & Adopt Result",
        "Verifies an ROI winner at full frame size, adopts its algorithm + parameters into the main workspace, or imports the rendered image straight into Outputs.",
        "• Adopt Output reuses already-rendered pixels without waiting to re-stack if alignment matches.",
    ),
    "mosaic_layout": (
        "Panorama / Tile Grid Sources & Layout",
        "Imports tile frames or rendered stack outputs and arranges them manually or via row/column grid order and zigzag overlap parameters.",
        "• Set Start Corner, Capture Order, and Overlap % before clicking Apply Grid to snap microscope/macro X-Y rail captures quickly.",
    ),
    "mosaic_align": (
        "Mosaic Alignment & Canvas Controls",
        "Runs automatic feature alignment, fits tiles to view, or fine-tunes individual tile X/Y position, scale, rotation, and preview opacity.",
        "• Lower Preview Opacity briefly to visually inspect alignment seams against overlapping neighbors.",
    ),
    "mosaic_control_points": (
        "Manual Mosaic Control Points",
        "Lets you place matching point pairs between overlapping tiles and optimize global alignment when repetitive textures defeat automatic feature matching.",
        "• Pick pairs near the overlap zone corners, then click Optimize Alignment and inspect RMS residual errors.",
    ),
    "mosaic_mask": (
        "Mosaic Exclusion & Inclusion Brush Mask",
        "Paints exclusion or forced-inclusion masks over overlapping tile regions to suppress sensor edge artifacts, moving dust, or unwanted seam ghosting.",
        "• Paint Exclude over blurred tile borders so seam selection picks sharper neighbors cleanly.",
    ),
    "mosaic_export": (
        "Mosaic Seam, Blending, Projection & HDR Options",
        "Controls seam finder, multiband blending levels, photometric exposure/gain matching, spherical/planar projection, lens distortion, HDR bracketing, and final output depth/ICC.",
        "• Multiband blending smooths residual brightness differences across stitched macro tiles.",
    ),
    "scalebar": (
        "Calibrated Scale Bar Overlay",
        "Adds a physically calibrated micrometer/millimeter scale bar to previews and exported images using lens EXIF metadata or manual µm/px calibration.",
        "• Use Auto color to automatically pick crisp high-contrast black/white ink against the corner background.",
    ),
    "watermark": (
        "Tiled Text Watermark & EXIF Macros",
        "Composes semi-transparent diagonal or corner text watermarks supporting dynamic EXIF date, camera, and focal-length macros.",
        "• Click macro chips below the text box to embed shooting metadata cleanly across batch exports.",
    ),
    "retouch_export": (
        "Missing-Data Export Dialog",
        "Shows live progress while writing auxiliary retouch or missing-pixel diagnostic layers safely in the background.",
        "• Click Cancel at any point to abort cleanly without leaving partial output files.",
    ),
    "roi_options": (
        "Region of Interest (ROI) Output Handling",
        "Chooses whether an ROI render outputs only the cropped rectangle or pastes the sharply stacked patch back onto a chosen base frame.",
        "• Paste onto Base Frame preserves full-frame dimensions while speeding up localized test renders.",
    ),
    "font_size": (
        "Interface Text Size",
        "Adjusts global desktop UI font size (in points) with live side-by-side comparison against the default system scale.",
        "• Helpful on high-DPI 4K monitors when compact labels feel too small.",
    ),
}

_ZH_TOPICS: dict[str, tuple[str, str, str]] = {
    "overview": (
        "交互式图文帮助",
        "你可以按住右上角的“?”按钮拖动到界面任意区域、按钮或参数上查看说明；也可以在软件任意界面（包括弹窗里）按住 H 键并点击目标功能区，不会误触发原按钮功能。",
        "• 拖动右上角“?”：放到不懂的功能区或参数上松手即可看图文说明。|• 单击右上角“?”：进入帮助拾取模式，点击任意按钮或面板看说明（Esc 退出）。|• 按住 H 键：鼠标变为帮助指针，在主界面与子窗口里直接点击即可查看对应功能详解。",
    ),
    "registration": (
        "图像配准（对齐）",
        "聚焦堆栈拍摄时，微调焦距或相机呼吸效应会导致每一张照片的视角、放大倍率或位置有轻微差异。图像配准会先将所有原图逐像素对齐，再进行清晰度融合。",
        "• ECC（推荐）：亚像素级透视对齐，最适合微距呼吸效应与轻度手持偏移。|• Homography（单应性）：基于特征点快速对齐，纹理丰富时速度更快。|• 无配准（None）：仅适用于导轨锁死或已经预先对齐过的序列。",
    ),
    "fusion": (
        "焦点融合算法",
        "从对齐后的每一张切片中挑出最清晰的对焦区域，把整条景深范围合成一张全清晰的成品照片。",
        "• OpenFocus - B（推荐）：基于深度图权重，细节锐利、过渡自然、几乎无光晕。|• OpenFocus - A：加权平均法，速度快、适合低对比度平滑表面。|• OpenFocus - C：多尺度金字塔融合，擅长处理细毛发、交叉边缘与半透明细节。",
    ),
    "align_area": (
        "配准检测区",
        "把对齐特征计算限制在你框选的主体静态区域内，避免背景树叶晃动、水面波纹或大面积散景散斑干扰对齐结果。",
        "• 在左侧原图画布上拖出一个矩形覆盖稳定清晰的主体结构即可。|• 若背景大面积模糊或存在反光干扰导致画面错位，优先框选配准检测区。|• 点击清除按钮即可恢复全图参与配准。",
    ),
    "align_preview": (
        "预览对齐效果",
        "在正式渲染前，直接在左侧原图画布中预览对齐并裁剪后的画面，方便检查边缘是否还会跳动或偏移。",
        "• 开启后拖动底部切片滑块，对比不同切片在同一细节处是否完全重合。|• 若仍有漂移，可切换 ECC 或配合“配准检测区”重新尝试。",
    ),
    "source_canvas": (
        "源图像栈画布与放大镜",
        "显示当前选中的原始切片，可滚动浏览不同对焦位置，也支持局部放大镜检查每张切片的真实合焦边界。",
        "• 使用画布下方的滑块可逐帧切换前后景。|• 右键按住或开启放大镜工具，可直接查看 100%+ 像素清晰度。|• 支持直接把照片或文件夹拖入窗口导入。",
    ),
    "sharpness_curve": (
        "逐帧清晰度曲线",
        "统计堆栈里每张照片的高频清晰度分布，帮你一眼看出主体集中在哪些切片，以及前后两端是否存在全虚焦的废片。",
        "• 蓝线：各帧相对清晰度走势，竖线标记当前正在查看的帧。|• 红点：清晰度显著低于主峰的帧（通常是全虚焦的背景或前景），可在源图像列表里取消勾选跳过，节省渲染时间。",
    ),
    "result_canvas": (
        "输出成品画布",
        "展示融合后的全清晰图像或对齐预览，支持无级缩放、1:1 原像素查看、放大镜细节检视，以及多版本成果对比。",
        "• 底部工具栏支持一键 1:1、适应窗口与百分比滑块缩放。|• 在右侧“输出”列表中点击不同渲染记录，可快速横向对比不同参数效果。",
    ),
    "wipe_compare": (
        "分屏对比（卷帘对比）",
        "用可拖动的竖直分界线把中间画布一分为二，让你直观对比融合前后、或两个不同输出版本之间的细节差异。",
        "• 拖动画面中央的分界线把手，重点检查毛发边缘与暗部过渡是否出现光晕或重影。|• 勾选下方 A/B 联动滑块可随时更换对比帧。",
    ),
    "render_action": (
        "开始渲染",
        "按当前选定的配准算法、融合算法、右侧展开参数与“渲染设置”，对源图像列表中已勾选的切片执行完整渲染。",
        "• 渲染过程中按钮变为“取消渲染”，可随时安全中断。|• 大图集推荐先点右侧的“实时预览”快速调好参数再正式渲染。",
    ),
    "live_preview": (
        "实时预览（交互调参窗口）",
        "建立轻量化对齐代理栈后，可实时拖动 Radius / Smoothing / Feather、直观对比切片直方图并裁剪有效子栈范围（支持 OpenFocus A/B/C）。",
        "• 拖动参数滑块几十毫秒内即可看到彩色结果变化，还支持按住空格键瞬时对比原帧或拉动竖直分屏线。|• 说明：实时预览用于快速调试融合算法核心参数与子栈范围，因此不重复应用正式渲染的亮度校正、输出锐化等后处理开关；正式“开始渲染”时会自动生效“渲染设置”中的全部选项。",
    ),
    "render_settings": (
        "渲染设置（工作流与后处理选项）",
        "集中管理影响最终全分辨率渲染的选项：快速预览草稿、反转栈序、亮度匹配、基底亮度校正以及输出锐化强度。",
        "• 快速预览：先跑一张缩小草稿确认效果（不占输出历史）。|• 亮度匹配 / 基底校正：消除多帧之间轻微曝光闪烁或长堆栈亮度漂移。|• 输出锐化：合成结束后适度增强微反差（注意避免过高带来噪点与白边）。",
    ),
    "reset_defaults": (
        "重置默认设置",
        "一键把配准方式、融合算法、各项核心参数及渲染选项恢复为 OpenFocus 推荐默认值。",
        "• 重置不会清空已导入的源图像栈，也不会删除已经渲染好的历史输出。",
    ),
    "render_queue": (
        "队列、批处理与全部方法对比",
        "点击可打开更多渲染菜单：将当前配置“加入队列”、打开“渲染队列”统一排队跑多组参数，或执行“渲染全部方法”横向比较所有融合算法。",
        "• 需要尝试多组半径/平滑度组合时，用“加入队列”一次排队后台跑完最省心。|• “渲染全部方法”会依次调用所有可用融合算法生成结果，便于科研与评测对比。",
    ),
    "source_list": (
        "源图像列表",
        "按顺序展示当前导入的所有源图像。勾选框控制是否参与本次渲染，支持拖拽上下调整切片顺序。",
        "• 发现首尾纯虚焦帧或闭眼/晃动废帧时，取消勾选即可跳过，无需删除磁盘原图。|• 若导入顺序反了，可直接在“渲染设置”中开启反转栈序或拖拽重排。",
    ),
    "output_list": (
        "输出历史列表",
        "保存本次会话中已完成的渲染成果。点击任意条目可切换显示、对比细节，或进一步精修笔刷与多格式导出（TIFF / PSD / PSB / 交互 HTML / 3D 等）。",
        "• 建议保留多组不同参数成品，配合对比工具挑出细节最干净的一版。|• 右键列表项可直接打开精修、另存为或删除操作。",
    ),
    "status_panel": (
        "当前资源与状态栏",
        "实时显示已加载帧数、分辨率大小、当前负责运算的硬件设备（GPU 或 CPU）以及系统内存占用情况。",
        "• 处理几十张 45MP 以上大图时，可留意内存余量；如遇显存紧张，系统会自动协调工作。",
    ),
    "radius": (
        "半径（Radius）",
        "决定评估每个像素局部清晰度对比度时参考的邻域窗口大小（单位：像素）。",
        "• 数值较小（4–8）：保留极细毛发、纤毛与微小纹路，适合高频细节。|• 数值较大（12–25）：抗噪能力更强，平滑低纹理区域更容易干净稳定，不易起麻点。",
    ),
    "smoothing": (
        "平滑度（Smoothing）",
        "对相邻像素选中的焦平面编号进行空间平滑约束，避免物体表面出现零碎跳变的斑块或硬接缝。",
        "• 数值越低，景深边界响应越锐利；高景深跨度或高 ISO 噪点图建议适当提高到 3–8。|• OpenFocus - A 不使用该参数，因此自动隐藏。",
    ),
    "feather": (
        "羽化（Feather）",
        "控制 OpenFocus - B 中相邻切片蒙版拼接处的过渡羽化宽度。",
        "• 前景与背景交界处若出现生硬断层，可适当加大羽化值让过渡更柔和。|• 需要紧贴硬朗边缘减少泛白光晕时，可保持 0–3 较小值。",
    ),
    "kernel": (
        "滤波核大小（Kernel Size）",
        "空间梯度/导向滤波或 DCT 计算所使用的奇数正方形滑窗尺寸（N × N 像素）。",
        "• 滑窗越大对高噪点图像的抗干扰能力越强，但过大会略微模糊紧邻的细小局部焦点。",
    ),
    "levels": (
        "金字塔分解层数（N）",
        "设定小波或金字塔变换拆分的空间尺度层数，层数越多兼顾微细纹理与大范围光影渐变的尺度范围越广。",
        "• 默认 4 层在绝大多数摄影堆栈中兼顾清晰度与干净无光晕过渡。",
    ),
    "ai_mode": (
        "AI 融合权重模式",
        "控制神经网络融合在合并多层决策图时采用连续概率混合（柔和）还是硬性择优（锐利）。",
        "• 柔和模式（Soft）：渐变自然、抗噪稳定；硬选模式（Hard）：边缘明确时能保留最接近单张原图的对比度。",
    ),
    "reference": (
        "配准参考帧",
        "指定哪一张源图像作为几何基准锚点，其余所有切片都将经过对齐变换以贴合它。",
        "• 中间帧（默认）：可把整条堆栈前后端的放大倍率伸缩误差平摊到最小。|• 首帧 / 尾帧：需要严格以最近或最远一帧取景框为准时选用。",
    ),
    "tiles": (
        "分块并行与重叠区域",
        "把超高分辨率大图切成带重叠边的矩形块依次融合，显著降低峰值内存需求。",
        "• 保持足够重叠像素（如 128–256 px），可以避免块与块交界处出现接缝误差。",
    ),
    "registration_width": (
        "配准缩放分析宽度",
        "仅在估算对齐变换矩阵时把图像暂时缩放到该最大宽度；正式生成融合图像时依然使用原始全分辨率像素变形。",
        "• 默认 1024–1600 px 对绝大多数相机序列已具备亚像素精度，同时速度提升数倍。",
    ),
    "align_limits": (
        "配准位移 / 旋转 / 缩放上限",
        "设置每帧允许的最大平移比例、旋转角度与倍率变化阈值，防止某张无纹理或失焦严重的帧把整组画面拉偏。",
        "• 手持拍摄抖动较大时可适当放宽上限；高精度导轨堆栈可收紧上限作为异常保护。",
    ),
    "align_auto_crop": (
        "自动裁剪配准黑边",
        "对齐透视变形后，外围切片边缘会出现无法覆盖全画幅的黑边，开启后会自动内裁保留所有切片均有效的中心共同区域。",
        "• 日常建议开启以直接得到无黑边成品；若想完整保留某张基准边角取景再手动裁剪可关闭。",
    ),
    "align_interpolation": (
        "像素变形插值算法",
        "配准变换重采样像素时采用的插值核：Lanczos 最锐利细腻；Cubic 与 Linear 速度更快、边缘略柔。",
        "• 高像素微观细节推荐使用 Lanczos。",
    ),
    "live_preview_histogram": (
        "深度贡献直方图与子栈裁剪",
        "实时显示每一帧在当前融合结果中的贡献占比；拖动画布下方双滑块可快速剪掉首尾不需要的虚焦帧。",
        "• 向内拖动两端蓝色滑块即可实时屏蔽废片观察效果。|• 单击直方图中的任意柱子可直接跳转到对应源切片。",
    ),
    "live_preview_split": (
        "实时预览分屏对比",
        "在左侧“实时融合预览”与右侧“选定源切片”之间切出可拖动的竖直分屏界线，无需等待全图渲染即可核对毛刺与细节保留。",
        "• 拖动中间蓝色圆钮对比，双击分割线恢复 50% 对齐位置。",
    ),
    "live_preview_size": (
        "代理预览尺寸",
        "选择实时预览内部采用的缩放分辨率（小 / 中 / 大），用于在几十毫秒级极速响应与精细局部放大检视之间平衡。",
        "• 平时调参用“中（720 px）”最快；重点检查细小光晕时切到“大（1024 px）”。",
    ),
    "live_preview_stack": (
        "堆栈浏览器与原帧窥视",
        "在实时预览中前后浏览单张切片，并可切换“显示已对齐”核对呼吸效应对齐是否准确；按住空格键可瞬间切回原帧对比。",
        "• 如果开启“显示已对齐”后切帧依然晃动，可返回主界面先调整配准算法或配准检测区。",
    ),
    "opt_quick_preview": (
        "快速预览（草稿模式）",
        "正式点击“开始渲染”时先以较低分辨率快速走一遍完整算法与设定流程，草稿只临时查看、不写入输出历史。",
        "• 适合最终出图前花一两秒核对构图与对齐方向；准备保存成品时取消勾选。",
    ),
    "opt_reverse_order": (
        "反转栈序（倒序堆栈）",
        "反转 OpenFocus 族算法的前后景遍历与深度方向编号，适合从远景到近景拍摄的图像序列。",
        "• 也会同步影响生成深度图的近远黑白色阶映射方向。",
    ),
    "opt_exposure_match": (
        "曝光匹配",
        "在配准和融合前自动均衡各切片之间的轻微亮度起伏（如灯光频闪或自然光云影变化）。",
        "• 绝大多数堆栈建议保持开启，可大幅减少合成图上的亮度斑块。",
    ),
    "opt_base_correct": (
        "基底亮度校正",
        "将融合结果整体影调锚定回基准参考帧水平，避免超长堆栈合成后出现整体偏亮或偏暗偏移。",
        "• 适合高动态或上百帧长堆栈。",
    ),
    "opt_sharpen_output": (
        "输出锐化",
        "在焦点融合全部完成后对结果进行可控锐化补偿，恢复对齐插值与混合带来的轻微锐度损失。",
        "• 勾选后可调节锐化强度滑块；建议适中取值以防噪点放大或边缘白边。",
    ),
    "batch_split": (
        "单文件夹自动拆组设置",
        "当一个文件夹里连续拍摄了多组景深照片时，系统可自动识别分界并拆分成多组各自对齐融合。",
        "• 自适应（自动）：自动根据相邻帧内容相似度切分组别，通常无需手动调参数。|• 按数量/时间/曝光：当固定张数或拍摄间隔规律时可精确按条件分割。",
    ),
    "batch_output_format": (
        "批量与监控输出格式（含 DNG/RAW）",
        "选择融合成品保存的图像格式。输入相机 RAW/DNG 时可选择输出 16 位无损 TIFF/PNG，或选择 DNG 直接导出相机线性 RAW（LinearRaw DNG）。",
        "• 选择 DNG 时会读取相机原始线性像素并保留色彩校准矩阵，方便进后期软件继续以 RAW 流程调色（要求输入为相机 RAW/DNG）。|• JPG/BMP 为 8 位格式，重要作品推荐 TIFF、PNG 或 DNG。",
    ),
    "watch_folder": (
        "文件夹监控自动化",
        "持续监听指定输入目录，联机拍摄或共享目录中一有新堆栈写入稳定，便自动完成配准、融合与导出。",
        "• 写入稳定延迟：防止相机刚写一半文件就被提前读取。|• 已完成组会自动登记，避免重复渲染。",
    ),
    "retouch_toolbar": (
        "修修补补历史、检查点与帧定位",
        "包含撤销/重做、保存与恢复修图检查点、按住对比修饰前原图，以及 F9 一键定位当前鼠标下像素来自哪张源帧。",
        "• 鼠标悬停在成品的瑕疵处按 F9（或点准星图标），底部切片条会自动跳转到贡献该区域的源帧。",
    ),
    "retouch_tools": (
        "修图工具栏（四种核心画笔）",
        "通过顶部图标切换画笔：源帧画笔（用单张清晰原片修补鬼影）、还原画笔（擦除修补还原融合原图）、曝光画笔（局部提亮压暗）、仿制图章（Alt 取样修补瑕疵）。",
        "• 源帧画笔 (B)：修重影与错位最常用。|• 还原画笔 (E)：修过头时擦回融合结果。|• 曝光画笔：修正细微暗影或过亮反光。|• 仿制图章 (S)：先 Alt+左键取样，再涂抹清理杂物。",
    ),
    "retouch_brush_params": (
        "画笔直径、硬度与不透明度",
        "控制修补画笔的作用范围、边缘羽化过渡柔度和每次涂抹叠加的强度。",
        "• 快捷键 [ / ] 或 Ctrl+滚轮 可随时快速放大缩小画笔圆圈。",
    ),
    "retouch_source_options": (
        "源帧精细容差、亮度与局部对齐偏移",
        "用于解决高反差毛发误涂、源帧与成品轻微明暗不一致，以及局部微距漂移造成的重影错位。",
        "• 色彩容差：开启后只替换与落笔点颜色相近的区域，保护细毛发边缘不被糊掉。|• 局部自动对齐 / 偏移 (X/Y)：当选中源帧在该处有 1–5 像素微小错位时，一键自动对齐后再涂抹就不会有重影。",
    ),
    "retouch_clone_options": (
        "仿制图章取样与模式",
        "在区域平移取样与纹理平铺取样之间切换，并可通过 Alt+点击 在画布上拾取参考样点。",
        "• 区域模式适合复制固定位移的结构；平铺模式适合修补大面积均匀虚化背景。",
    ),
    "retouch_overlays": (
        "诊断叠层显示与全局操作",
        "可在成品画布上直接叠加显示切片清晰度热图或深度贡献分布图，也可一键自动背景净化、调整全局影调或全部还原。",
        "• 打开“显示深度/来源图”可直观看出哪一片区域由哪一帧合成而来。",
    ),
    "retouch_filmstrip": (
        "底部源切片胶片条与帧跳转",
        "用缩略图、滑块或右下角页码框快速切换源图像栈中的每一帧，方便挑选最合焦的那张作为画笔来源。",
        "• 快捷键逗号 `,` / 句号 `.` 或滚动/拖动滑块，可逐帧微调比对左右画面。",
    ),
    "adv_experimental": (
        "启用实验性功能",
        "解锁仍在持续演进中的实验性配准与融合选项（例如 OpenFocusAlign），点击“确定”保存后立即生效。",
        "• 常规日常生产如果希望保持稳定默认算法集合，保持关闭即可。",
    ),
    "adv_legacy": (
        "启用遗留功能",
        "仅用于复现历史工程时恢复已被替代的早期算法或旧版功能；常规新工程建议使用当前推荐方案以获得更快速度与更低内存占用。",
        "• 不必为了常规新堆栈开启此项；关闭时主界面与算法选择器会保持简洁。",
    ),
    "adv_privacy": (
        "隐私、错误报告与使用统计",
        "控制是否在遇到异常崩溃时自动上报诊断日志，以及是否分享匿名功能使用统计。",
        "• 错误报告有助于定位特定显卡驱动或解码器的兼容性问题；可随时按需关闭。",
    ),
    "adv_storage": (
        "工作文件临时目录",
        "设置解码帧缓存、对齐代理与临时渲染文件的磁盘存放位置。",
        "• 处理数十张 45MP 以上 16-bit 大图时，建议指向空间充裕的高速本地 SSD/NVMe 目录。",
    ),
    "adv_naming": (
        "输出命名模板与完成提示音",
        "配置可选文件名前缀、支持动态标签的输出文件名模板，以及全栈渲染完成时的声音提示。",
        "• 点击 [METHOD]、[ALIGN]、[DATE] 等标签按钮可直接插入模板，下方示例会实时刷新最终文件名。",
    ),
    "adv_export_color": (
        "导出元数据与色彩管理",
        "控制导出时是否移除 EXIF 拍摄/定位信息，以及是否在工作区预览时将内嵌 ICC 色域实时转换为 sRGB 显示。",
        "• 使用 AdobeRGB、Display P3 或 ProPhoto 素材时建议开启“ICC 色彩管理预览”，避免画面发灰。",
    ),
    "adv_raw_decode": (
        "相机 RAW 外部转换器回退与解码参数",
        "配置系统 DNG 转换器路径（仅在内置 LibRaw 暂不支持最新相机型号 RAW 时自动回退调用），并可快速进入 RAW 解码参数设置。",
        "• 可点击“自动检测”寻找已安装的 DNG 转换器，或打开“RAW 解码参数”调整白平衡、高光重建与色域。",
    ),
    "adv_resources": (
        "资源偏好与分块设置",
        "调节运算并发资源策略（平衡 / 响应优先 / 极致性能），以及特定算法在超大尺寸下的分块与重叠参数。",
        "• “响应优先”适合后台批量处理时仍需流畅查看窗口；OpenFocus A/B/C 会自动管理内存分块无需手调。",
    ),
    "adv_startup": (
        "启动加载与自动更新",
        "配置启动时是否恢复上次堆栈、导入图像时是否弹出降采样缩放提示，以及更新包自动下载与测试版通道选项。",
        "• 若总是以 100% 原分辨率或固定缩放导入，可取消“每次询问加载缩放”。",
    ),
    "adv_gallery": (
        "社区画廊弹幕与缓存设置",
        "自定义画廊弹幕开关、显示位置、字号、透明度、跨语言翻译策略，以及本地作品下载目录与缓存空间上限。",
        "• 点击“清理缓存”可立即释放画廊缩略图与临时预览缓存，不影响已导出成品。",
    ),
    "adv_developer": (
        "开发者选项与钥匙管理",
        "验证开发者密钥以解锁内部开发/社区测试功能入口。",
        "• 输入密钥后点击“验证并解锁”；不再需要时点击“锁定并清除”即可恢复普通模式。",
    ),
    "sensor_cleanup": (
        "传感器坏点/灰尘掩膜与降噪",
        "在对齐与融合之前，按传感器灰尘坏点掩膜文件自动修复污点，并可选开启传感器空间降噪。",
        "• 该设置全局保存并对所有后续堆栈生效；可通过“工具 → 生成灰尘修复图”快速提取掩膜。",
    ),
    "dust_map": (
        "传感器灰尘修复图（平场增益提取）",
        "从离焦纯白平场图或当前切片中提取灰尘暗斑与暗角增益场，生成可复用的传感器修正掩膜。",
        "• 调节“尘点尺度 (σ)”以准确囊括灰尘晕影而不误伤正常纹理；应用前可直接对比处理前后效果。",
    ),
    "depth_source_method": (
        "深度图来源、配准与测距算法",
        "选择基于当前源图像栈重新计算深度，或读取已渲染成品记录对应的深度，并设定清晰度评价算法与参数。",
        "• 当基于源栈重新计算时，系统会先按所选配准算法对齐各帧后再评估深度变化。",
    ),
    "depth_postprocess": (
        "深度外观、平滑度与物理 Z 轴标定",
        "切换伪彩色谱、深度反转、伽马对比度、中值平滑强度，支持叠加透明热图及设定 µm/mm 物理 Z 轴步长与方向。",
        "• 若焦栈采集方向与近远期望相反，勾选“深度反转”；导出定量深度 TIFF 或真实比例 3D 网格前请先标定步距与单位。",
    ),
    "depth_3d_surface": (
        "3D 交互曲面预览",
        "在窗口内实时渲染 GPU 三维高度场模型：左键拖拽旋转、滚轮缩放，顶部栏支持切换纹理/泥塑/深度色/线框等视图并实时调节起伏高度与细节。",
        "• 支持随时切换视角预设、开启自动运镜展示或截取高清 3D 画面快照。",
    ),
    "depth_3d_lighting": (
        "3D 虚拟多光源布光",
        "独立调节主光方位角、仰角以及主光/补光/环境光强度，凸显微观表面浮雕与纹理细节。",
        "• 调低“主光仰角”能以侧逆光强化几丁质、花粉或金属刻痕的立体质感。",
    ),
    "depth_export": (
        "深度图、3D 网格与单文件 HTML5 3D 导出",
        "导出伪彩/16 位原始深度图、切片索引、对比度图、OBJ/PLY/STL 三维网格、红青/左右立体图、旋转视频，以及零依赖的交互式单文件 HTML5 WebGL 3D 查看器。",
        "• 给客户分享可拖拽旋转的 3D 模型时，推荐导出“单文件 HTML5 3D 查看器 (.html)”，浏览器双击即看且无需安装客户端。",
    ),
    "stereo_3d": (
        "立体 3D 与旋转展示导出",
        "设置红青立体图、平行/交叉眼立体对的眼距视差、会聚零平面及深度反转，或设定 3D 旋转 MP4/GIF 的分辨率、帧数与运镜方式。",
        "• 推荐保持 2%–4% 适中眼距视差，既立体突出又不易重影晕眩。",
    ),
    "html_animation": (
        "交互式 HTML5 对焦浏览页导出",
        "将对齐后的焦栈切片与可选融合成品打包为一个可离线分享的交互网页，客户可直接用鼠标拖动体验整条景深对焦变化。",
        "• 默认 1280 px + WebP 格式兼顾高清晰度与极小体积，勾选“包含融合成品”可在网页内一键对比前后。",
    ),
    "raw_preview": (
        "相机 RAW 实时解片预览",
        "对所选 RAW 源帧实时运行 16 位线性解马赛克与影调计算，直观检验高光拉回与色彩表现后再应用到全栈。",
        "• 可通过下拉框切换检查堆栈中不同帧的高光和暗部，避免逐帧批量解片后才发现过曝。",
    ),
    "raw_wb_exposure": (
        "RAW 白平衡、曝光与高光恢复",
        "配置相机/自动/自定义白平衡及色温色调偏移、EV 曝光补偿、高光裁切重建模式和自动亮度。",
        "• 亮部花瓣或反光金属轻微过曝时，建议将高光恢复改为“重建”；正式堆栈建议关闭自动亮度以保证帧间亮度一致。",
    ),
    "raw_demosaic_color": (
        "解马赛克算法与 16 位输出色彩空间",
        "选择拜耳阵列解马赛克算法（如 AHD / DCB / DHT）及堆栈管线使用的输出色域（sRGB / AdobeRGB / ProPhoto 等）。",
        "• 宏微距高密纹理推荐使用 AHD 或 DCB，能有效减少细边缘伪色与拉链纹。",
    ),
    "batch_folders": (
        "批量处理堆栈队列",
        "管理待自动处理的多个堆栈文件夹，支持直接拖入多个目录或开启“单文件夹智能拆分”一次性处理连拍目录。",
        "• 使用单文件夹拆分前，可在下方选择自适应场景检测或按固定张数/时间/曝光拆分。",
    ),
    "batch_methods": (
        "批量 / 监视文件夹融合与配准设定",
        "统一指定批处理队列与自动监视文件夹新进堆栈所使用的融合算法与配准模式。",
        "• 无人值守大批量任务推荐使用 ECC 配准搭配 OpenFocus - B，兼容轻微呼吸与偏移。",
    ),
    "batch_grouping": (
        "大堆栈分组渲染",
        "按设定组大小与重叠帧数将超深堆栈分段融合再合并，降低大尺寸深栈的内存峰值压力。",
        "• 保持 1–2 帧重叠可确保跨组接缝处的清晰度平滑过渡。",
    ),
    "watch_settle": (
        "监视文件夹等待稳定延时",
        "检测到新增文件后先等待设定秒数确认写入完成再自动对齐融合，防止联机拍摄半途抢跑读到未写完的照片。",
        "• 若通过 Wi-Fi 或网络盘慢速传输大型 RAW 照片，可适当调大该等待时间。",
    ),
    "substacks": (
        "分段渲染（重叠子栈拆分）",
        "按每段张数或总段数将深焦栈均匀拆成多个重叠子栈分别渲染，可用于景深分层对比、大景深分段微调与自动二次合栈。",
        "• 适当保留 1–2 帧重叠可消除分段景深交界处断层；勾选“自动二次融合”可在各子栈完成后自动合成终稿。",
    ),
    "batch_converter": (
        "批量图像格式与分辨率转换器",
        "批量将多张图片或文件夹统一转换为 PNG / TIFF / JPG / WebP / BMP，并支持按 2K/4K/8K 长边或百分比等比缩放。",
        "• 可选输出到同级目录、子目录或自定义路径，安全保留原始母片不被覆盖。",
    ),
    "compare_methods": (
        "全部融合算法对比选择与参数调节",
        "勾选多种融合算法、统一调整配准及单算法参数，对同一组堆栈切片横向运行清晰度对比。",
        "• 高分辨率大图建议先用“ROI 局部对比”秒级试参，选出优胜方案后再验证全图。",
    ),
    "compare_view": (
        "候选对比网格与 1:1 细节裁切视窗",
        "左侧参考图与右侧所有候选算法同步联动 1:1 细节视角，方便零延时横评毛发锐度、光晕与暗部噪点。",
        "• 在左侧参考画布拖动局部框或切换“全图 / 细节”视图，可快速巡检多个关键部位。",
    ),
    "compare_adopt": (
        "全图验证与采纳结果 / 参数",
        "对满意的候选算法执行全分辨率验证、直接把它的算法与参数应用回主工作区，或将结果直接收入右侧“输出”列表。",
        "• 若配准一致且无需重算，点击“采纳输出”可直接复用已渲染画面，节省等待时间。",
    ),
    "mosaic_layout": (
        "大图拼接 / 马赛克序列与网格布局",
        "导入瓦片切片或已融合输出成品，按行列数、扫描方向、起始角落和重叠百分比一键网格排布或手动调整。",
        "• 对于显微镜电控台或微距导轨扫描拍摄的阵列图，先设定行/列数、蛇形路径和重叠率再点“应用网格”最快对齐。",
    ),
    "mosaic_align": (
        "拼接自动配准与单瓦片微调",
        "触发全局自动特征对齐、适配画布视角，或选中单个瓦片微调 X/Y 坐标、缩放、旋转角度与叠层透明度。",
        "• 短暂调低单个瓦片的“不透明度”，可直观对照下方相邻帧重叠区域的接缝贴合情况。",
    ),
    "mosaic_control_points": (
        "人工控制点标定与全局优化",
        "当重复周期纹理导致自动特征对齐困难时，手动点选两张相邻瓦片之间的对应同名特征点并执行全局优化。",
        "• 在重叠区的四个角落附近分散点选几对清晰控制点后点“优化”，观察残差值缩小即可明显改善拼接错位。",
    ),
    "mosaic_mask": (
        "拼接蒙版涂抹（排除 / 强制保留区域）",
        "在重叠拼接区用画笔标注“排除”或“包含”掩膜，剔除单帧边缘暗角、游走灰尘或局部虚影干扰。",
        "• 用排除画笔盖住某帧模糊或有异物的角落，接缝线查找器就会自动绕行并选用另一帧清晰部分。",
    ),
    "mosaic_export": (
        "接缝线、多频段融合、投影与 HDR 拼接选项",
        "配置最优接缝线算法、多频段融合层数、光度一致性校正、平面/柱面/球面投影、镜头畸变参数与输出位深/ICC。",
        "• 开启多频段融合与曝光光度平衡可有效消除网格瓦片之间的亮度阶梯与接缝。",
    ),
    "scalebar": (
        "物理标尺叠加（Scale Bar）",
        "基于镜头/显微 EXIF 元数据或手动 µm/px 标定值，在预览与导出成品角落绘制真实物理尺寸标尺。",
        "• 推荐颜色选择“自动”，系统会根据角落背景明暗自动切换黑白高对比文字与刻度线。",
    ),
    "watermark": (
        "平铺水印与动态 EXIF 宏标签",
        "为导出成品添加可调字号、透明度与位置的版权水印，并支持一键插入相机、日期、焦距等 EXIF 动态宏标签。",
        "• 使用下方 EXIF 标签快捷插入作者或拍摄参数后，右侧可实时预览排布与密铺效果。",
    ),
    "retouch_export": (
        "缺失数据 / 精修图层导出进度",
        "后台安全写出精修诊断图层或补全导出文件时的进度监视窗口。",
        "• 导出期间点击“取消”会安全终止写入且不残留半成品损坏文件。",
    ),
    "roi_options": (
        "ROI 感兴趣区域渲染输出方式",
        "选择仅单独输出裁剪后的框选区域，还是将清晰框选区域无缝贴回所选基准底图形成完整画幅。",
        "• 若只需在整体画面上验证或精修一小块关键细节，“贴回基准帧”能快速获得完整构图结果。",
    ),
    "font_size": (
        "界面字体大小调节",
        "调节桌面端全局界面字体大小（pt），并在右侧直接并排预览调整前后文字排版与可读性对比。",
        "• 高分辨率 4K 屏幕上若觉得面板标签偏小，适当提高字号后点击确定即可全局生效。",
    ),
}

_EXTRA_KEYS: dict[str, dict[str, str]] = {
    "en": {
        "ctx_help_beacon_label": "Help · Drag / H",
        "ctx_help_beacon_tooltip": "Click to pick any UI element, drag '?' onto any control, or hold H anywhere and click to view illustrated help.",
        "ctx_help_pick_banner": "Help Mode: click any panel, button, or setting to view illustrated help (or drag '?' / hold H anywhere) · Press Esc to exit",
        "ctx_help_footer_tip": "Tip: Drag the '?' pill in the top-right corner onto any area, or hold the H key anywhere (even inside sub-dialogs) and click any item to inspect it.",
        "ctx_help_btn_menu_more": "Queue, Batch & Compare Methods",
        "ctx_help_btn_menu_more_hint": "Enqueue current settings, open Render Queue, or run Render All Methods",
        "ctx_help_open_full_methods": "Available Algorithms in OpenFocus",
        "ctx_help_drag_chip": "? Drop onto control for help",
    },
    "zh": {
        "ctx_help_beacon_label": "帮助 · 拖/? · 按住H",
        "ctx_help_beacon_tooltip": "按住拖动到任意不懂的区域即可图文讲解；单击进入点选模式；在软件任意界面（含子窗口）按住 H 键并用鼠标点击也可随时呼出说明。",
        "ctx_help_pick_banner": "帮助模式已开启：点击任意功能区、按钮或参数即可查看图文说明（也可拖动右上角 ? 或按住 H 键点击）· 按 Esc 退出",
        "ctx_help_footer_tip": "提示：平时也可直接按住右上角“?”拖到任何区域，或在软件任意界面按住 H 键并点击目标控件直接呼出图文帮助。",
        "ctx_help_btn_menu_more": "队列与全部方法对比",
        "ctx_help_btn_menu_more_hint": "加入渲染队列、管理批量任务，或用所有可用融合算法对比渲染",
        "ctx_help_open_full_methods": "当前可选算法列表与适用场景",
        "ctx_help_drag_chip": "? 拖到对应功能区松手查看图文帮助",
    },
}


def _flatten_topics(lang: str, topics: dict[str, tuple[str, str, str]]) -> dict[str, str]:
    out = dict(_EXTRA_KEYS[lang])
    for topic, (title, body, tips) in topics.items():
        out[f"ctx_help_{topic}_title"] = title
        out[f"ctx_help_{topic}_body"] = body
        out[f"ctx_help_{topic}_tips"] = tips
    return out


CONTEXT_HELP_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": _flatten_topics("en", _EN_TOPICS),
    "zh": _flatten_topics("zh", _ZH_TOPICS),
}


def _context_help_pack(code: str) -> dict[str, str]:
    """Return localized context help strings with full English fallback."""
    en = CONTEXT_HELP_TRANSLATIONS["en"]
    pack = CONTEXT_HELP_TRANSLATIONS.get(code)
    if not pack:
        return dict(en)
    merged = dict(en)
    merged.update(pack)
    return merged

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
}

_EXTRA_KEYS: dict[str, dict[str, str]] = {
    "en": {
        "ctx_help_beacon_label": "Help (Drag / Hold H)",
        "ctx_help_beacon_tooltip": "Click to pick any UI element, drag '?' onto any control, or hold H anywhere and click to view illustrated help.",
        "ctx_help_pick_banner": "Help Mode: click any panel, button, or setting to view illustrated help (or drag '?' / hold H anywhere) · Press Esc to exit",
        "ctx_help_footer_tip": "Tip: Drag the '?' pill in the top-right corner onto any area, or hold the H key anywhere (even inside sub-dialogs) and click any item to inspect it.",
        "ctx_help_btn_render_settings_hint": "Configure quick preview draft, stack order reversal, exposure matching, base brightness correction, and output sharpening",
        "ctx_help_btn_reset_hint": "Restore registration, fusion method, core parameters, and render settings to defaults (keeps loaded sources and outputs)",
        "ctx_help_btn_menu_more": "Queue, Batch & Compare Methods",
        "ctx_help_btn_menu_more_hint": "Enqueue current settings, open Render Queue, or run Render All Methods",
        "ctx_help_open_full_methods": "Available Algorithms in OpenFocus",
        "ctx_help_drag_chip": "? Drop onto control for help",
    },
    "zh": {
        "ctx_help_beacon_label": "帮助（拖拽 / 按住 H）",
        "ctx_help_beacon_tooltip": "按住拖动到任意不懂的区域即可图文讲解；单击进入点选模式；在软件任意界面（含子窗口）按住 H 键并用鼠标点击也可随时呼出说明。",
        "ctx_help_pick_banner": "帮助模式已开启：点击任意功能区、按钮或参数即可查看图文说明（也可拖动右上角 ? 或按住 H 键点击）· 按 Esc 退出",
        "ctx_help_footer_tip": "提示：平时也可直接按住右上角“?”拖到任何区域，或在软件任意界面按住 H 键并点击目标控件直接呼出图文帮助。",
        "ctx_help_btn_render_settings_hint": "配置快速预览草稿、反转栈序、亮度匹配、基底亮度校正与输出锐化等后处理选项",
        "ctx_help_btn_reset_hint": "将配准方式、融合算法、核心参数与渲染设置恢复为推荐默认值（保留已导入源图与历史输出）",
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

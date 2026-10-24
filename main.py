import sys
import os

# Two OpenMP runtimes reach this process: Intel MKL (through numpy/scipy from
# the Anaconda distribution) and torch's own libiomp5md.dll. Whichever loads
# second aborts the process with "OMP: Error #15 ... already initialized" —
# observed as a hard crash the moment an ECC + AI render started, because that
# is the first code path that touches both. The variable must be set before
# numpy or torch is imported, so it lives here rather than in a helper module.
# Intel documents it as unsupported-but-safe for this exact case; the
# alternative (rebuilding numpy or torch against a single runtime) is not
# something a user install can do.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# In --noconsole (windowed) PyInstaller builds sys.stdout/stderr are None, and
# any print() then crashes with AttributeError ('NoneType' object has no
# attribute 'write'). Route them to the null device instead.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMessageBox,
    QProgressDialog,
    QDialog,
    QPushButton,
)
from PyQt6.QtCore import Qt, QThread, QUrl, QEvent, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtGui import QFont, QIcon, QDesktopServices, QDragEnterEvent, QDropEvent, QPixmap
from core import ImageStackLoader
from core.app import OpenFocusApplication, process_command_line_args
from core.workers import ROIAlignmentWorker
import cv2
from ui.styles import GLOBAL_DARK_STYLE
from locales import trans
from dialogs import (
    EnvironmentInfoDialog,
    ContactInfoDialog,
    BatchProcessingDialog,
    TileSettingsDialog,
    ThreadSettingsDialog,
    AIBatchSettingsDialog,
    ROIRenderOptionsDialog,
)
from dialogs.help import RenderMethodHelpDialog
from dialogs.settings import RegistrationSettingsDialog
from utils import (
    get_ui_font_family,
    get_monospace_font_family,
    get_default_font,
    show_message_box,
    show_warning_box,
    show_error_box,
    show_success_box,
    show_custom_message_box,
    style_message_box,
    resource_path,
)
from utils.image_utils import cv2_to_pixmap
from utils.settings_store import (
    load_window_settings,
    save_window_settings,
    get_saved_language,
    load_drag_export_format,
    get_drag_export_format,
    set_drag_export_format,
    save_window_layout,
    restore_window_layout,
    get_settings,
)
from ui.image_panels import create_source_panel, create_result_panel
from ui.menus import setup_menus
from ui.right_panel import bind_right_panel, create_right_panel
from controllers.render_manager import RenderManager
from controllers.output_manager import OutputManager
from controllers.source_manager import SourceManager
from controllers.label_manager import LabelManager
from controllers.export_manager import ExportManager
from controllers.transform_manager import TransformManager
from controllers.batch_manager import BatchManager
from controllers.update_manager import (
    UpdateManager,
    STATE_DOWNLOADING,
    STATE_FAILED,
    STATE_READY,
    STATE_STAGING,
    STATE_CANCELLING,
    STATE_CANCELLED,
)
from core import is_ai_fusion_available
from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    REG_DOWNSCALE_WIDTH, DEFAULT_THREAD_COUNT,
    AI_BATCH_SIZE,
)

# QThreads still running when the app shuts down. They are kept alive at
# module scope instead of being terminated: QThread.terminate() on a thread
# inside cv2/torch native code can corrupt the process, and dropping the last
# reference aborts it ("QThread: Destroyed while thread is still running").
_ORPHAN_THREADS: list = []


def _capture_label_state(host):
    """Label set as plain data for an undo snapshot (None when absent)."""
    manager = getattr(host, "label_manager", None)
    if manager is None or not hasattr(manager, "export_state"):
        return None
    try:
        return manager.export_state()
    except Exception:
        return None


def _restore_scalebar_state(host, saved) -> None:
    """Apply a snapshot scale-bar state, ignoring hosts without one."""
    if not isinstance(saved, dict) or not saved:
        return
    restore = getattr(host, "_restore_scalebar_state", None)
    if restore is None:
        return
    restore(saved)


def _restore_label_state(host, label_state) -> None:
    """Apply a snapshot label state, ignoring hosts without a manager."""
    if label_state is None:
        return
    manager = getattr(host, "label_manager", None)
    if manager is None or not hasattr(manager, "import_state"):
        return
    try:
        manager.import_state(label_state)
    except Exception:
        pass


def _shutdown_worker(worker, timeout_ms=1500):
    """Ask a QThread to stop, then wait; never terminate it.

    Cancellation is polled between pipeline stages, so a worker deep inside
    cv2/torch cannot be interrupted. Killing it with terminate() mid-native-
    call is undefined behaviour; park the reference at module scope instead
    (the process is exiting).
    """
    if worker is None:
        return
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:
            pass
    elif hasattr(worker, "is_cancelled"):
        worker.is_cancelled = True
    if worker.isRunning():
        worker.quit()  # no-op for run() overrides, harmless otherwise
        if not worker.wait(timeout_ms):
            _ORPHAN_THREADS.append(worker)


def _apply_font_family(qss: str, ui_font: str, mono_font: str) -> str:
    """Swap the stylesheet's font stacks for the resolved families.

    The old code replaced one literal string; when the generator changed its
    font stack the replacement silently stopped matching and the QSS kept its
    own family. Match every stack the generator can emit.
    """
    for stack in ('"Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif',
                  '"Segoe UI", "Microsoft YaHei"'):
        qss = qss.replace(stack, ui_font)
    for stack in ('Consolas, "Segoe UI", monospace',):
        qss = qss.replace(stack, mono_font)
    return qss


def update_manager_for(window) -> UpdateManager:
    """Return (creating on first use) the window's update manager.

    Module-level so the update entry points keep working on hosts that never
    ran OpenFocus.__init__ — the test suite binds those two methods onto a
    bare widget to exercise the whole download -> stage -> hand-off chain.
    """
    manager = getattr(window, 'update_manager', None)
    if manager is None:
        manager = UpdateManager(window)
        window.update_manager = manager
    return manager


class OpenFocus(QMainWindow):
    # Emitted from the background torch probe; auto-queued to the UI thread.
    _stackmff_probe_done = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        # Logging must exist however the class is constructed: the packaged
        # app calls setup_logging() from the __main__ block, but tests,
        # scripts and embeddings that import this module never reach it, and
        # then every action log silently vanished.
        from utils.logging_setup import setup_logging, log_session_start
        setup_logging()
        log_session_start()

        self.setWindowTitle("OpenFocus")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 设置窗口图标
        icon_path = resource_path("assets", "OpenFocus.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 数据初始化为空
        self.stack_images = []  # QPixmap列表
        self.current_img_index = -1
        self.image_filenames = []  # 文件名列表
        self.raw_images = []  # 存储原始的numpy数组图像，用于配准和融合
        self.base_images = []  # 存储首次加载的基准尺寸图像，用于恢复和resize计算
        # Scale the loader used to decode base_images, relative to the files on
        # disk; resize_all_images must never offer more detail than that.
        self.base_scale_factor = 1.0
        self.current_scale_factor = 1.0
        # Full path of each frame in image_filenames, when the origin is known
        self.image_source_paths = []
        self.fusion_result = None  # 存储最新的融合结果
        self.fusion_results = []  # 存储所有融合结果的历史记录
        self.registration_results = []  # 存储配准后的图像栈
        self.current_result_index = -1  # 当前显示的结果图像索引
        self.aligned_images = []  # 存储已对齐的图像栈，避免重复对齐
        self.is_images_aligned = False  # 标记图像是否已对齐
        self.last_alignment_options = None  # 存储上次使用的对齐选项
        self.current_kernel_mode = None
        
        # ROI模式相关属性
        self.roi_aligned_images = []  # 存储ROI模式下对齐的图像栈
        self.roi_alignment_worker = None  # ROI配准工作线程
        self.roi_mode_active = False  # ROI模式是否激活
        self.roi_aligned_raw_count = 0  # 记录对齐时原始图像的数量，用于判断是否可以复用

        # Tile settings defaults
        self.tile_enabled = True
        self.tile_block_size = TILE_BLOCK_SIZE
        self.tile_overlap = TILE_OVERLAP
        self.tile_threshold = TILE_THRESHOLD
        # Registration downscale default (用户可在 Settings -> Registration 中修改)
        self.reg_downscale_width = REG_DOWNSCALE_WIDTH
        # 全局线程数设置，默认4（可在 Settings 中修改）
        self.thread_count = DEFAULT_THREAD_COUNT
        # StackMFF V3 批量大小设置，默认2（可在 Settings 中修改）
        self.ai_batch_size = AI_BATCH_SIZE
        # GPU 加速开关（Settings 菜单可切换，QSettings 持久化）
        self.use_gpu = True
        # 基底校正（消除亮度漂移）：默认关闭，右面板开关控制
        self.base_frame_correct = False
        # 融合前曝光匹配：默认开启；整栈标定到末帧，所有算法共用
        self.exposure_match = True
        # 恢复持久化设置与最近文件（语言需在菜单构建前应用）
        load_window_settings(self)
        load_drag_export_format(self)
        saved_lang = get_saved_language()
        if saved_lang:
            trans.set_language(saved_lang)
        self.recent_files = getattr(self, 'recent_files', [])
        self.recent_projects = getattr(self, 'recent_projects', [])
        self._undo_stack = []
        self._redo_stack = []
        self._undo_guard = False
        self.excluded_frames = set()
        
        self.render_manager = RenderManager(self)
        self.output_manager = OutputManager(self)
        self.source_manager = SourceManager(self)
        self.label_manager = LabelManager(self)
        self.export_manager = ExportManager(self)
        self.transform_manager = TransformManager(self)
        self.batch_manager = BatchManager(self)
        self.update_manager = UpdateManager(self)

        # 初始化图像加载器
        self.image_loader = ImageStackLoader()
        
        # 启用拖放
        self.setAcceptDrops(True)
        
        # 当前显示的图像索引
        self.current_display_index = -1
        self.current_folder_path = ""

        # 主题应用必须在 init_ui 之后：light 模式要剥离控件的内联暗样式，
        # UI 未建时剥离是空操作，之后创建的控件会带着暗色内联样式覆盖
        # 应用级亮色 QSS（白底白字，整个右侧面板不可读）
        self.init_ui()
        self.apply_dark_theme()

        # Install global event filter to handle Space key for panning
        QApplication.instance().installEventFilter(self)

        self._mouse_in_source_preview = False
        self._mouse_in_result_preview = False
        
        # Connect language change signal
        trans.languageChanged.connect(self.update_ui_text)

        # Restore last session's window geometry and splitter positions
        restore_window_layout(self)

        # Optionally re-load the last opened stack a moment after launch
        if getattr(self, "restore_last_stack", False):
            from PyQt6.QtCore import QTimer as _QTimer
            _QTimer.singleShot(1500, lambda: self.source_manager.restore_last_stack())

        # Set initial text for drag hint if needed (it is set in image_panels or loaded later, 
        # but let's Ensure it matches current lang if empty)
        # Note: image_panels create_source_panel sets initial text. 
        # We might want to update it here or in update_ui_text called initially?
        # Let's call update_ui_text once to sync everything if needed, or just leave it.
        # Actually image_panels.py uses hardcoded text. 
        # I should probably update image_panels.py too, or just override text here.
        if hasattr(self, 'lbl_source_img'):
             self.lbl_source_img.setText(trans.t('drag_hint'))
        if hasattr(self, 'lbl_result_img'):
            self.lbl_result_img.setText(trans.t('result_hint'))

    def init_ui(self):
        setup_menus(self)
        self.rebuild_recent_menu()
        self.rebuild_recent_projects_menu()

        # 2. 主容器
        main_container = QWidget()
        self.setCentralWidget(main_container)
        # 垂直：顶部可选更新横幅 + 主分割器。横幅默认隐藏，隐藏时不占高度。
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === 核心布局：主分割器 ===
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ---------------------------------------------------------
        # A. 图像显示区 (左侧)
        # ---------------------------------------------------------
        image_display_container = QWidget()
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.setContentsMargins(0, 0, 0, 0) # 边缘贴合
        image_display_layout.setSpacing(0)

        # 双视图分割器 (左：源图像栈， 右：融合结果)
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- A1/A2. 图像视图面板 ---
        source_panel = create_source_panel()
        self._panel_widgets = [source_panel.widget]
        self.lbl_source_title = source_panel.title
        self.lbl_source_img = source_panel.image_label
        self.source_control_bar = source_panel.control_bar
        self.stack_slider = source_panel.slider
        self.lbl_stack_info = source_panel.info_label
        self.stack_slider.valueChanged.connect(self.update_source_view)

        # ROI Button
        self.btn_preview_roi = source_panel.roi_btn
        self.btn_preview_roi.toggled.connect(self.toggle_roi_mode)
        self.sharpness_curve = source_panel.sharpness_curve
        self.sharpness_curve.set_tooltip_base(trans.t('sharpness_curve_tooltip'))
        self.sharpness_curve.frame_clicked.connect(
            lambda idx: self.stack_slider.setValue(idx))
        # 注意：现在ROI是在右侧面板选择的，所以lbl_source_img的roiDeleted不再控制按钮状态
        # self.lbl_source_img.roiDeleted.connect(lambda: self.btn_preview_roi.setChecked(False))

        result_panel = create_result_panel()
        self._panel_widgets.append(result_panel.widget)
        self.lbl_result_title = result_panel.title
        self.lbl_result_img = result_panel.image_label
        self.result_control_bar = result_panel.control_bar
        self.result_slider = result_panel.slider
        # 连接右侧面板的ROI模式退出请求信号
        self.lbl_result_img.roiModeExitRequested.connect(self._on_roi_mode_exit_requested)
        self.lbl_result_info = result_panel.info_label
        self.result_slider.valueChanged.connect(self.update_result_view)

        # Wipe A/B compare mode (result panel second page)
        self.wipe_active = False
        self.view_stack = result_panel.view_stack
        self.wipe_widget = result_panel.wipe_widget
        self.wipe_bar = result_panel.wipe_bar
        self.chk_wipe_follow_left = result_panel.chk_wipe_follow_left
        self.slider_wipe_left = result_panel.slider_wipe_left
        self.lbl_wipe_left_index = result_panel.lbl_wipe_left_index
        self.chk_wipe_follow_right = result_panel.chk_wipe_follow_right
        self.slider_wipe_right = result_panel.slider_wipe_right
        self.lbl_wipe_right_index = result_panel.lbl_wipe_right_index
        self.btn_wipe = result_panel.btn_wipe
        self.btn_wipe.toggled.connect(self.set_wipe_mode)
        self.chk_wipe_follow_left.toggled.connect(self._on_wipe_follow_left_toggled)
        self.slider_wipe_left.valueChanged.connect(self._on_wipe_left_slider_changed)
        self.chk_wipe_follow_right.toggled.connect(self._on_wipe_follow_right_toggled)
        self.slider_wipe_right.valueChanged.connect(self._on_wipe_right_slider_changed)

        self.lbl_source_img.enterPreview.connect(self._on_enter_source_preview)
        self.lbl_source_img.leavePreview.connect(self._on_leave_source_preview)
        self.lbl_result_img.enterPreview.connect(self._on_enter_result_preview)
        self.lbl_result_img.leavePreview.connect(self._on_leave_result_preview)

        self.view_splitter.addWidget(source_panel.widget)
        self.view_splitter.addWidget(result_panel.widget)
        
        # 监听分割器移动事件，实时更新图像尺寸
        self.view_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        image_display_layout.addWidget(self.view_splitter)
        self.main_splitter.addWidget(image_display_container)

        # ---------------------------------------------------------
        # B. 右侧控制面板
        # ---------------------------------------------------------
        right_panel_components = create_right_panel()
        self.right_panel_components = right_panel_components
        self.right_splitter = right_panel_components.splitter
        self.btn_reset = right_panel_components.btn_reset
        self.btn_render = right_panel_components.btn_render
        self.btn_compare_all = right_panel_components.btn_compare
        self.btn_compare_all.clicked.connect(self.render_manager.start_compare_all)
        self.chk_quick_preview = right_panel_components.chk_quick_preview
        self.btn_method_help = right_panel_components.btn_method_help
        self.btn_reg_help = right_panel_components.btn_reg_help
        # 融合/对齐算法均为下拉单选；window.current_fusion_method() /
        # current_alignment_method() 是统一的读取入口
        self.combo_fusion = right_panel_components.combo_fusion
        self.combo_align = right_panel_components.combo_align
        self.chk_base_correct = right_panel_components.chk_base_correct
        self.chk_exposure_match = right_panel_components.chk_exposure_match
        self.chk_reverse_order = right_panel_components.chk_reverse_order
        # 帧序方向：False=asc（文件序，默认）/ True=desc（倒序栈）
        self.stack_reverse_order = False
        self.slider_smooth = right_panel_components.slider_smooth
        self.lbl_kernel = right_panel_components.lbl_kernel
        self.lbl_smooth_value = right_panel_components.smooth_value_label
        self.smooth_widget = right_panel_components.smooth_widget
        self.ai_mode_widget = right_panel_components.ai_mode_widget
        self.lbl_ai_mode = right_panel_components.lbl_ai_mode
        self.radio_ai_soft = right_panel_components.radio_ai_soft
        self.radio_ai_hard = right_panel_components.radio_ai_hard
        # AI 输出模式：soft=概率加权混合（默认）/ hard=逐像素取最可能帧
        self.ai_mode = "hard" if self.ai_mode == "hard" else "soft"
        self.radio_ai_hard.setChecked(self.ai_mode == "hard")
        self.radio_ai_soft.setChecked(self.ai_mode == "soft")
        self.depth_smooth_row = right_panel_components.depth_smooth_row
        self.lbl_depth_smooth = right_panel_components.lbl_depth_smooth
        self.spin_depth_smooth = right_panel_components.spin_depth_smooth
        # 先按当前方法显示它自己的 σ（A/B/C 各有默认与记忆值）
        self.spin_depth_smooth.setValue(
            float(self.smoothing_for_method(self.current_fusion_method())))
        self.spin_depth_smooth.valueChanged.connect(self.set_ofb_depth_smoothing)
        self.source_images_label = right_panel_components.source_images_label
        self.file_list = right_panel_components.file_list
        self.output_label = right_panel_components.output_label
        self.output_list = right_panel_components.output_list
        
        # Status labels
        self.lbl_status_loaded = right_panel_components.lbl_status_loaded
        self.lbl_status_resolution = right_panel_components.lbl_status_resolution
        self.lbl_status_gpu = right_panel_components.lbl_status_gpu
        self.lbl_status_memory = right_panel_components.lbl_status_memory

        self.main_splitter.addWidget(right_panel_components.widget)
        bind_right_panel(self, right_panel_components)

        # 两个亮度开关按当前算法取值（OpenFocus A/B/C 默认两者都开，
        # 其余算法沿用历史默认）。不能直接用全局属性：那会把上一次
        # 会话里别的算法的选择带进来。
        self._sync_brightness_checkboxes()
        self.chk_reverse_order.blockSignals(True)
        self.chk_reverse_order.setChecked(bool(
            getattr(self, "stack_reverse_order", False)))
        self.chk_reverse_order.blockSignals(False)

        # Probe torch availability in a background thread: importing torch
        # can block for seconds, and this only decides one checkbox state.
        from PyQt6.QtCore import QTimer
        self._stackmff_probe_done.connect(self._apply_stackmff_availability)
        self.update_manager.check_finished.connect(self._show_update_result)
        self.update_manager.state_changed.connect(self._on_update_state)
        self.update_manager.error.connect(
            lambda title, message: show_warning_box(self, title, message))
        self.update_manager.manual_download_needed.connect(self._open_releases_page)
        self.update_manager.restart_requested.connect(self._quit_for_update)
        self._build_update_banner()
        self._build_update_indicator()
        QTimer.singleShot(0, self._probe_stackmff_availability_async)

        # 确保分割器已经添加了子部件后再设置折叠属性
        # 使用QTimer来延迟设置折叠属性，确保所有子部件都已正确添加
        from PyQt6.QtCore import QTimer
        
        # Timer for system status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_dynamic_status)
        self.status_timer.start(2000) # Update every 2 seconds

        # Panel actions have no menu entry, so give them shortcuts here.
        # They must not clash with the menu shortcuts in ui/menus.py.
        for keys, handler in (
                ("Ctrl+R", lambda: self.render_manager.start_render()),
                ("Ctrl+Shift+R", lambda: self.render_manager.start_compare_all()),
                ("Ctrl+Shift+B", lambda: self.show_batch_processing_dialog()),
                ("Ctrl+M", lambda: self.show_depth_map_dialog()),
                ("Ctrl+E", lambda: self.btn_wipe.toggle()),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)

        def set_splitter_properties():
            try:
                if self.view_splitter.count() >= 2:
                    self.view_splitter.setCollapsible(0, False)
                    self.view_splitter.setCollapsible(1, False)
                if self.right_splitter.count() >= 3:
                    self.right_splitter.setCollapsible(0, False)
                    self.right_splitter.setCollapsible(1, False)
                    self.right_splitter.setCollapsible(2, False)
                if self.main_splitter.count() >= 2:
                    self.main_splitter.setCollapsible(0, False)
                    self.main_splitter.setCollapsible(1, False)
            except IndexError:
                pass

        QTimer.singleShot(100, set_splitter_properties)
        
        # 比例设置：左侧图像区域占据大部分空间，右侧控制面板可调整
        self.main_splitter.setStretchFactor(0, 3)  # 左侧图像区域伸展因子为3
        self.main_splitter.setStretchFactor(1, 1)  # 右侧控制面板伸展因子为1
        
        # 设置初始分割比例（可选）
        # 获取窗口宽度并设置初始比例为 75% : 25%
        total_width = self.width()
        self.main_splitter.setSizes([int(total_width * 0.75), int(total_width * 0.25)])

        main_layout.addWidget(self.update_banner)
        main_layout.addWidget(self.main_splitter, 1)

    def eventFilter(self, obj, event):
        """Global event filter to handle Space key and Delete key interactions."""
        # Handle Delete/Backspace for ROI
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
                # Check if mouse is over source image and ROI mode is active
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    # Check if widget is visible and under mouse
                    if self.lbl_source_img.isVisible() and self.lbl_source_img.underMouse():
                         # If ROI exists, delete it and consume event
                         if self.lbl_source_img.get_roi_rect() is not None:
                             self.lbl_source_img.set_roi_rect(None)
                             self.lbl_source_img.roiDeleted.emit()
                             return True # Event handled, do not propagate to list widget

        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            # Check if mouse is over source or result image
            handled = False
            
            # Using underMouse() is safer than tracking enter/leave events which can be unreliable
            # during rapid movement or with other widgets involved
            if hasattr(self, 'lbl_source_img') and self.lbl_source_img and self.lbl_source_img.isVisible():
                if self.lbl_source_img.underMouse():
                    self.lbl_source_img.set_space_pressed(True)
                    handled = True
            
            if hasattr(self, 'lbl_result_img') and self.lbl_result_img and self.lbl_result_img.isVisible():
                if self.lbl_result_img.underMouse():
                    self.lbl_result_img.set_space_pressed(True)
                    handled = True
            
            if handled:
                return True
                
        elif event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    self.lbl_source_img.set_space_pressed(False)
                
                if hasattr(self, 'lbl_result_img') and self.lbl_result_img:
                    self.lbl_result_img.set_space_pressed(False)
        
        return super().eventFilter(obj, event)

    # --- Status Display ---
    def update_loaded_status(self):
        """Update loaded images count and resolution info."""
        count = len(self.stack_images)
        if count > 0:
            # Calculate average size (approximate from QPixmap or numpy array if available)
            # self.raw_images stores numpy arrays
            total_size_mb = 0
            if self.raw_images:
                # height * width * channels * 1 byte (uint8) / 1024 / 1024
                # assuming 3 channels uint8
                try:
                    total_size_bytes = sum(img.nbytes for img in self.raw_images)
                    avg_size_mb = (total_size_bytes / count) / (1024 * 1024)
                    self.lbl_status_loaded.setText(trans.t('status_loaded_fmt').format(count, avg_size_mb))
                    
                    h, w = self.raw_images[0].shape[:2]
                    self.lbl_status_resolution.setText(trans.t('status_res_fmt').format(w, h))
                except Exception:
                    self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                    self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
            else:
                self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
        else:
            self.lbl_status_loaded.setText(trans.t('status_loaded').format(0))
            self.lbl_status_resolution.setText(trans.t('status_res').format('-'))

    def _update_dynamic_status(self):
        """Update GPU and Memory status."""
        # Memory
        try:
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            
            used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3)
            total_gb = stat.ullTotalPhys / (1024**3)
            self.lbl_status_memory.setText(trans.t('status_ram_fmt').format(used_gb, total_gb))
        except Exception:
            self.lbl_status_memory.setText(trans.t('status_ram').format('N/A'))

        # GPU. Rule the user asked for: when this machine has no usable
        # accelerator, say N/A — never "CPU", which sounds like the render
        # fell back mid-run. "CPU*" (classical methods are CPU-only by
        # design) is only meaningful when an accelerator actually exists.
        from core.multi_focus_fusion import (torch_device_kind,
                                             is_ai_fusion_available)
        try:
            if not getattr(self, 'use_gpu', True):
                kind = 'N/A'                      # user disabled it
            elif not is_ai_fusion_available():
                kind = 'N/A'                      # torch not installed
            else:
                kind = torch_device_kind()
                if kind is None:
                    # Resolve once in the background; until then assume none.
                    self._probe_device_kind_async()
                    kind = 'N/A'
                if kind not in ('CUDA', 'MPS'):
                    kind = 'N/A'
                elif not self._ai_method_selected():
                    # Accelerator present but the selected method is one of
                    # the CPU-only classical ones.
                    kind = 'CPU*'
            self.lbl_status_gpu.setText(trans.t('status_gpu').format(kind))
        except Exception:
            self.lbl_status_gpu.setText(trans.t('status_gpu').format('Err'))

    def _ai_method_selected(self) -> bool:
        try:
            return self.combo_fusion.currentData() == "stackmffv3"
        except Exception:
            return False

    # --- 逻辑控制 ---

    # --- Undo / redo for stack mutations (rotate, flip, resize, delete,
    #     drag-reorder and the per-frame exclusion set) ---

    UNDO_LIMIT = 20

    def _capture_stack_state(self) -> dict:
        """Snapshot stack membership for undo, without copying pixels.

        Every stack mutation (rotate, flip, resize, delete, drag-reorder,
        flat-field correction) replaces the array objects instead of writing
        into them, so list references are a valid snapshot and undo costs
        O(frames) memory instead of a full second copy of the stack.
        Invariant: never modify a frame array in place after push_undo().
        """
        return {
            "raw": list(self.raw_images),
            "filenames": list(self.image_filenames),
            "source_paths": list(getattr(self, "image_source_paths", [])),
            "base": list(getattr(self, "base_images", [])),
            "excluded": set(getattr(self, "excluded_frames", set())),
            "current": getattr(self, "current_img_index", 0),
            "scale": float(getattr(self, "current_scale_factor", 1.0) or 1.0),
            "base_scale": float(getattr(self, "base_scale_factor", 1.0) or 1.0),
            "source_px_um": getattr(self, "source_px_um", None),
            "source_transforms": list(getattr(self, "source_transforms", [])),
            "unrecoverable_source_changes": list(getattr(
                self, "unrecoverable_source_changes", []) or []),
            "scalebar": self._capture_scalebar_state(),
            "labels": _capture_label_state(self),
            "parameters": self._capture_parameter_state(),
        }

    def _capture_scalebar_state(self) -> dict:
        from utils.scalebar import capture_state
        return capture_state(self)

    def _capture_label_state(self):
        """Label set as plain data, for undo snapshots.

        Captions belong to the session state, so undoing a stack mutation
        must not leave captions that were added against the old frames.
        """
        manager = getattr(self, "label_manager", None)
        if manager is None:
            return None
        try:
            return manager.export_state()
        except Exception:
            return None

    def push_scalebar_undo(self, previous: dict) -> None:
        """Record a scale-bar configuration change as one undoable step."""
        if getattr(self, "_undo_guard", False):
            return
        state = self._capture_stack_state()
        state["scalebar"] = dict(previous or {})
        state["annotations_only"] = True
        self._mark_project_dirty()
        self._undo_stack.append(state)
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._refresh_undo_actions()

    def _restore_scalebar_state(self, saved) -> None:
        from utils.scalebar import restore_state
        restore_state(self, saved)

    def push_label_undo(self, previous) -> None:
        """Record a label-only change as one undoable step.

        The undo stack holds stack snapshots; a label edit captures the
        current stack unchanged plus the previous label set, so undo/redo
        walks captions without touching pixels.
        """
        if getattr(self, "_undo_guard", False):
            return
        state = self._capture_stack_state()
        if previous is not None:
            state["labels"] = previous
        state["annotations_only"] = True
        self._mark_project_dirty()
        self._undo_stack.append(state)
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._refresh_undo_actions()

    def _capture_parameter_state(self) -> dict:
        methods = ("guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv3",
                   "openfocus_a", "openfocus_b", "openfocus_c")
        return {
            "method": self.current_fusion_method(),
            "alignment": self.current_alignment_method(),
            "stack_reverse_order": self.stack_reverse_order,
            "dtcwt_levels": self.dtcwt_levels,
            "ai_mode": self.ai_mode,
            "kernels": {m: self.kernel_for_method(m) for m in methods},
            "smoothing": {m: self.smoothing_for_method(m) for m in methods},
            "brightness": {m: {key: self.brightness_option_for(key, m)
                                for key in ("exposure_match", "base_correct")}
                           for m in methods},
        }

    def push_parameter_undo(self, previous: dict) -> None:
        self._undo_stack.append({"parameters": previous, "parameters_only": True})
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._mark_project_dirty()
        self._refresh_undo_actions()

    def _restore_parameter_state(self, saved: dict) -> None:
        self.set_fusion_method(saved["method"])
        self.set_alignment_method(saved["alignment"])
        for method, value in saved["kernels"].items():
            if method in self._KERNEL_MODES:
                self._kernel_by_mode[self._KERNEL_MODES[method][0]] = value
        if saved["method"] in self._KERNEL_MODES:
            self.slider_smooth.setValue(saved["kernels"][saved["method"]])
        for method, value in saved["smoothing"].items():
            self.set_smoothing_for_method(method, value)
        for method, options in saved["brightness"].items():
            for option, enabled in options.items():
                self.set_brightness_option(option, method, enabled)
        self.chk_reverse_order.setChecked(saved["stack_reverse_order"])
        self.set_dtcwt_levels(saved["dtcwt_levels"])
        self.set_ai_mode(saved["ai_mode"])

    def invalidate_annotation_caches(self) -> None:
        """Drop rendered views that depend on labels or the scale bar."""
        cache = getattr(self, "_wipe_pix_cache", None)
        if cache is not None:
            cache.clear()
        if hasattr(self, "get_or_create_source_pixmap"):
            try:
                self.get_or_create_source_pixmap.cache_clear()
            except Exception:
                pass

    def push_undo(self) -> None:
        """Snapshot the stack before a mutation.

        Snapshots hold references (see _capture_stack_state), so the undo
        limit is about edit history, not memory.
        """
        if getattr(self, "_undo_guard", False):
            return
        # Any stack mutation makes the saved project stale.
        self._mark_project_dirty()
        self._undo_stack.append(self._capture_stack_state())
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._refresh_undo_actions()

    def _restore_stack_state(self, state: dict) -> None:
        if state.get("parameters_only"):
            self._restore_parameter_state(state["parameters"])
            self._mark_project_dirty()
            return
        if state.get("annotations_only"):
            _restore_label_state(self, state.get("labels"))
            _restore_scalebar_state(self, state.get("scalebar"))
            self.refresh_annotation_views()
            self._mark_project_dirty()
            return
        self.raw_images = list(state["raw"])
        self.image_filenames = list(state["filenames"])
        self.image_source_paths = list(state["source_paths"])
        self.base_images = list(state["base"])
        self.excluded_frames = set(state["excluded"])
        self.current_img_index = state["current"]
        # Resize is undoable too; without the scale factors the Resize dialog
        # reopened with the post-resize percentage while showing the old pixels.
        self.current_scale_factor = float(state.get("scale", 1.0) or 1.0)
        self.base_scale_factor = float(state.get("base_scale", 1.0) or 1.0)
        self.source_px_um = state.get("source_px_um")
        self.source_transforms = list(state.get("source_transforms", []))
        self.unrecoverable_source_changes = list(state.get(
            "unrecoverable_source_changes", []) or [])
        _restore_label_state(self, state.get("labels"))
        _restore_scalebar_state(self, state.get("scalebar"))
        # The restored stack invalidates every derived result, exactly like
        # rotate/flip do.
        self.transform_manager.invalidate_processing_results(
            clear_output_view=False, preserve_outputs=True)
        self.transform_manager.reload_image_stack(
            initial_index=max(0, min(state["current"],
                                     len(self.raw_images) - 1)))
        if hasattr(self, "file_list"):
            self.source_manager.decorate_source_items()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._undo_guard = True
        try:
            state = self._undo_stack.pop()
            current = self._capture_stack_state()
            current["annotations_only"] = state.get("annotations_only", False)
            current["parameters_only"] = state.get("parameters_only", False)
            self._redo_stack.append(current)
            self._restore_stack_state(state)
        finally:
            self._undo_guard = False
        self._refresh_undo_actions()
        self._show_status_message(trans.t("msg_undo_done"), 3000)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_guard = True
        try:
            state = self._redo_stack.pop()
            current = self._capture_stack_state()
            current["annotations_only"] = state.get("annotations_only", False)
            current["parameters_only"] = state.get("parameters_only", False)
            self._undo_stack.append(current)
            self._restore_stack_state(state)
        finally:
            self._undo_guard = False
        self._refresh_undo_actions()
        self._show_status_message(trans.t("msg_redo_done"), 3000)

    def _refresh_undo_actions(self) -> None:
        ui = getattr(self, "ui_objs", {})
        undo_action = ui.get("action_undo")
        redo_action = ui.get("action_redo")
        if undo_action is not None:
            undo_action.setEnabled(bool(self._undo_stack))
        if redo_action is not None:
            redo_action.setEnabled(bool(self._redo_stack))

    # --- Per-frame exclusion (checkboxes in the source list) ---

    def included_indices(self) -> list:
        """Indices of the frames that participate in registration/fusion."""
        excluded = getattr(self, "excluded_frames", set())
        return [i for i in range(len(self.raw_images)) if i not in excluded]

    def effective_stack(self):
        """(images, filenames, indices) of the frames that will be rendered.

        With nothing excluded this returns the live lists untouched, so every
        existing identity check keeps working.
        """
        idx = self.included_indices()
        all_names = list(self.image_filenames or [])
        if len(idx) == len(self.raw_images):
            return self.raw_images, all_names, idx
        images = [self.raw_images[i] for i in idx]
        names = [all_names[i] if i < len(all_names) else f"frame_{i + 1}"
                 for i in idx]
        return images, names, idx

    def reset_to_default(self):
        """重置到默认状态"""
        # 重置渲染方法 - 默认 AI（StackMFF V3）
        self.set_fusion_method("stackmffv3")

        # 重置配准选项 - 默认 ECC
        self.set_alignment_method("ecc")

        # 基底校正默认关闭；帧序默认文件序（不反转）
        self.chk_base_correct.setChecked(False)
        self.chk_reverse_order.setChecked(False)
        
        # 重置滑块值到默认值（含每个方法记住的值）
        self._kernel_by_mode = {}
        if getattr(self, "spin_depth_smooth", None) is not None:
            self.spin_depth_smooth.setValue(1.2)
        self.slider_smooth.setValue(31)
        
        # 更新滑块可用性
        self.update_slider_availability()

        # 重置图像显示缩放与放大镜状态
        self.lbl_source_img.reset_view()
        self.lbl_result_img.reset_view()
        
        # 恢复空状态提示 (如果当前没有加载图像)
        if hasattr(self, 'stack_images') and not self.stack_images:
            self.lbl_source_img.setText(trans.t('drag_hint'))
            self.lbl_source_img.setFont(QFont(get_default_font(), 16))
    


    def _configure_fusion_method_availability(self) -> None:
        """Disable fusion methods whose dependencies are not installed."""
        self._stackmff_available = bool(is_ai_fusion_available())
        self.refresh_method_availability()

    def _probe_device_kind_async(self) -> None:
        """Resolve CUDA/MPS in the background, only once something needs it.

        Called when the neural method is selected. Importing torch costs ~2.1 s
        and ~193 MB, which is why it is not done at startup any more.
        """
        import threading
        if getattr(self, "_device_probe_started", False):
            return
        self._device_probe_started = True

        def worker():
            from core.multi_focus_fusion import ensure_torch_device_kind
            ensure_torch_device_kind()

        threading.Thread(target=worker, daemon=True,
                         name="torch-device-probe").start()

    def _probe_stackmff_availability_async(self) -> None:
        """Probe torch availability off the UI thread, then apply the result."""
        import threading

        def worker():
            available = is_ai_fusion_available()
            self._stackmff_probe_done.emit(available)

        threading.Thread(target=worker, daemon=True, name="stackmff-probe").start()

        # The neural method is the default, so a default selection still needs
        # the accelerator resolved. Only pay for it when that is the case: a
        # user whose preference is a classical method never loads torch.
        try:
            if self.combo_fusion.currentData() == "stackmffv3":
                self._probe_device_kind_async()
        except Exception:
            pass

    def _apply_stackmff_availability(self, available: bool) -> None:
        self._stackmff_available = bool(available)
        self.refresh_method_availability()
        # The probe is what fills the cached device kind. Without this call the
        # status bar kept its "GPU: …" placeholder until some unrelated timer
        # tick happened to run, so a finished probe still looked unfinished.
        self._update_dynamic_status()

    def refresh_method_availability(self) -> None:
        """Re-apply the cached torch-probe result to the method controls.

        Other code paths (render restore, batch restore) blanket-enable the
        fusion combo, which silently re-armed StackMFF-V3 on machines without
        torch until the next restart."""
        available = getattr(self, '_stackmff_available', None)
        if available is None:
            return
        combo = self.combo_fusion
        i = combo.findData("stackmffv3")
        if i < 0:
            return
        # 下拉项级禁用：模型项 setEnabled(False) 让它在弹出列表里置灰
        item = combo.model().item(i)
        if item is None:
            return
        item.setEnabled(bool(available))
        combo.setItemData(
            i,
            trans.t("msg_stackmff_unavailable_text") if not available else "",
            Qt.ItemDataRole.ToolTipRole)
        if not available and combo.currentData() == "stackmffv3":
            # Fall back to the first ENABLED entry: AI is the default now, so
            # a torch-less machine would otherwise land on a disabled item.
            self.set_fusion_method(self._first_available_method())

    def _first_available_method(self) -> str:
        """First selectable fusion method in the dropdown (skips disabled)."""
        combo = self.combo_fusion
        for row in range(combo.count()):
            item = combo.model().item(row)
            if item is not None and item.isEnabled():
                code = combo.itemData(row)
                if code:
                    return code
        return "guided_filter"

    # --- Update check ---

    def check_for_updates(self, quiet: bool = False) -> None:
        """Ask GitHub for the latest release and report back asynchronously.

        The state machine lives in UpdateManager: quiet checks are
        rate-limited to one request per day and never open a dialog; a manual
        check reports "up to date" / network failures in a message box. A
        manual check also re-arms the banner for a version that was dismissed.
        """
        if not quiet:
            self._show_status_message(trans.t('update_checking'), 3000)
            self._set_banner_dismissed_tag("")
        self.update_manager.check(quiet=quiet)

    # --- Update execution (background download / restart) ---

    def _update_manager(self) -> UpdateManager:
        return update_manager_for(self)

    # --- Startup update banner ---

    def _build_update_banner(self) -> None:
        """Top-of-window reminder for a release the startup check found.

        The banner is the only update UI a user sees without opening a menu:
        the background check stays quiet, so without it a new release went
        unnoticed until the user happened to open Help. Dismissal is
        per-version (see utils.settings_store.UPDATE_BANNER_DISMISSED_KEY).
        """
        from widgets.update_banner import UpdateBanner
        self.update_banner = UpdateBanner(self)
        self.update_banner.update_requested.connect(self._on_banner_update_clicked)
        self.update_banner.whats_new_requested.connect(self._show_whats_new)
        self.update_banner.dismissed.connect(self._on_banner_dismissed)

    def _banner_dismissed_tag(self) -> str:
        try:
            from utils.settings_store import (UPDATE_BANNER_DISMISSED_KEY,
                                              get_settings)
            return str(get_settings().value(UPDATE_BANNER_DISMISSED_KEY, "") or "")
        except Exception:
            return ""

    def _set_banner_dismissed_tag(self, tag: str) -> None:
        try:
            from utils.settings_store import (UPDATE_BANNER_DISMISSED_KEY,
                                              get_settings)
            settings = get_settings()
            if tag:
                settings.setValue(UPDATE_BANNER_DISMISSED_KEY, tag)
            else:
                settings.remove(UPDATE_BANNER_DISMISSED_KEY)
        except Exception:
            pass

    def _on_banner_dismissed(self, tag: str) -> None:
        self._set_banner_dismissed_tag(tag or self.update_manager.tag)

    def _show_update_banner(self, tag: str, ready: bool = False) -> None:
        """Show the startup reminder unless this exact version was dismissed."""
        banner = getattr(self, "update_banner", None)
        if banner is None or not tag:
            return
        # Hiding a version's reminder is a decision about that version, not
        # about the download state: a READY build used to bypass it and pop
        # the banner back up.
        if tag == self._banner_dismissed_tag():
            return
        from constants import APP_VERSION
        banner.show_for(tag, APP_VERSION, ready=ready)

    def _on_banner_update_clicked(self) -> None:
        """Primary action: install when staged, otherwise start the update."""
        manager = self._update_manager()
        tag = getattr(self.update_banner, "tag", "") or manager.tag
        if manager.staged_tag():
            self.install_update_now()
            return
        if manager.zip_url:
            if manager.can_self_update():
                self._apply_update_and_restart(manager.zip_url, tag)
            else:
                self._download_and_launch_update(manager.setup_url, tag)
            return
        self._open_releases_page()

    def _show_whats_new(self) -> None:
        """'What's new': release notes when the check fetched them, else the
        releases page (a build with no notes has nothing to show inline)."""
        notes = getattr(self._update_manager(), "notes", "") or ""
        if not notes:
            self._open_releases_page()
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle(trans.t('update_available_title'))
        dialog.resize(620, 460)
        # Same theme rule as the other dialogs (light strips the dark inline
        # style, dark adds the control QSS); style_message_box only accepts a
        # QMessageBox and raises on a plain QDialog.
        # Styling comes from the generated application stylesheet.
        layout = QVBoxLayout(dialog)
        view = QTextBrowser(dialog)
        view.setPlainText(notes)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _build_update_indicator(self) -> None:
        """Status-bar widget: background download progress / "restart to update"."""
        from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

        bar = self.statusBar()
        container = QWidget(bar)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(6)
        self.lbl_update_status = QLabel("", container)
        self.update_progress = QProgressBar(container)
        self.update_progress.setFixedWidth(140)
        self.update_progress.setTextVisible(False)
        self.update_progress.setRange(0, 100)
        self.btn_update_action = QPushButton("", container)
        self.btn_update_action.setFlat(True)
        self.btn_update_action.clicked.connect(self._on_update_action_clicked)
        layout.addWidget(self.lbl_update_status)
        layout.addWidget(self.update_progress)
        layout.addWidget(self.btn_update_action)
        bar.addPermanentWidget(container)
        container.hide()
        # Always visible: showing/hiding the bar per message made the image
        # panels jump by one row on every render (and hidden-by-default bars
        # swallowed every direct statusBar().showMessage call).
        bar.messageChanged.connect(self._on_status_message_changed)
        bar.setSizeGripEnabled(False)
        self._update_indicator = container

    def _show_status_message(self, text: str, timeout_ms: int = 5000) -> None:
        bar = self.statusBar()
        bar.setVisible(True)
        bar.showMessage(text, timeout_ms)

    def _on_status_message_changed(self, text: str) -> None:
        if text:
            self.statusBar().setVisible(True)

    def _status_bar_busy(self) -> bool:
        """True while a permanent status widget (update indicator, render
        progress) is showing."""
        indicator = getattr(self, '_update_indicator', None)
        if indicator is not None and indicator.isVisible():
            return True
        progress = getattr(getattr(self, 'render_manager', None),
                           '_status_bar_progress', None)
        return progress is not None and progress.isVisible()

    def _refresh_update_indicator(self) -> None:
        """Render the manager's current state into the status bar + menu."""
        if not hasattr(self, '_update_indicator'):
            return
        manager = self.update_manager
        state, tag, percent = manager.state, manager.tag, manager.percent
        show = True
        self.btn_update_action.setEnabled(state != STATE_CANCELLING)
        if state == STATE_DOWNLOADING:
            self.lbl_update_status.setText(
                trans.t('update_bg_downloading').format(version=tag, percent=percent)
                if percent >= 0 else trans.t('update_downloading'))
            self.update_progress.setRange(0, 100 if percent >= 0 else 0)
            if percent >= 0:
                self.update_progress.setValue(percent)
            self.update_progress.show()
            self.btn_update_action.setText(trans.t('btn_cancel_download'))
            self.btn_update_action.show()
        elif state in (STATE_STAGING, STATE_CANCELLING):
            self.lbl_update_status.setText(
                trans.t('msg_update_cancel_stopping') if state == STATE_CANCELLING
                else trans.t('update_status_preparing'))
            self.update_progress.setRange(0, 0)  # busy indicator
            self.update_progress.show()
            self.btn_update_action.setText(trans.t('btn_cancel_download'))
            self.btn_update_action.show()
        elif state == STATE_READY:
            self.lbl_update_status.setText(
                trans.t('update_status_ready').format(version=tag))
            self.update_progress.hide()
            self.btn_update_action.setText(trans.t('btn_restart_now'))
            self.btn_update_action.show()
        elif state in (STATE_FAILED, STATE_CANCELLED):
            self.lbl_update_status.setText(
                trans.t('update_status_failed') if state == STATE_FAILED
                else trans.t('msg_render_cancelled'))
            self.update_progress.hide()
            self.btn_update_action.setText(trans.t('btn_retry'))
            self.btn_update_action.show()
        else:
            show = False
        self._update_indicator.setVisible(show)
        bar = self.statusBar()
        if show or not bar.currentMessage():
            bar.setVisible(show)

        action = getattr(self, 'ui_objs', {}).get('action_install_update')
        if action is not None:
            ready = bool(manager.staged_tag())
            action.setVisible(ready)
            if ready:
                action.setText(trans.t('menu_install_update').format(version=tag))

    def startup_update_check(self) -> None:
        """Startup entry point: staged build, then the quiet daily check.

        Runs on a timer after the window is up. `start_idle` restores an
        already-staged build without a network call; its state goes straight
        to READY, before any state_changed listener existed, so the banner is
        raised here explicitly.
        """
        self.update_manager.start_idle()
        staged = self.update_manager.staged_tag()
        if staged:
            self._show_update_banner(staged, ready=True)

    def _on_update_state(self, state: str, tag: str, percent: int) -> None:
        if state == STATE_READY and tag:
            self._show_status_message(
                trans.t('update_status_ready').format(version=tag), 8000)
            # The build finished downloading. The banner stays hidden if the
            # user hid this version; the status bar and Help menu still lead
            # to the restart, so the choice is not lost.
            self._show_update_banner(tag, ready=True)
        self._refresh_update_indicator()

    def _on_update_action_clicked(self) -> None:
        manager = self.update_manager
        if manager.state in (STATE_DOWNLOADING, STATE_STAGING):
            manager.cancel_download()
        elif manager.state in (STATE_FAILED, STATE_CANCELLED) and manager.zip_url:
            manager.download(manager.tag, manager.zip_url, manual=True)
        else:
            self.install_update_now()

    def install_update_now(self) -> None:
        """User pressed "update": the staged build is swapped in immediately."""
        manager = self._update_manager()
        if not manager.staged_tag():
            if manager.zip_url:
                manager.download(manager.tag, manager.zip_url, manual=True)
            return
        # Applying the update closes the app and stops its workers. The
        # manager runs the shared protection (tasks, unexported pixels,
        # unsaved project, snapshot) before it starts the swap script.
        manager.apply_and_restart()

    def _confirm_update_quit(self) -> bool:
        """Ask before a restart stops running tasks.

        The safe choice is the default: the old dialog defaulted to quitting
        and discarding the running work.
        """
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t('msg_warning'))
        box.setText(trans.t('msg_update_discards_work'))
        cancel_tasks = box.addButton(trans.t('btn_cancel_tasks_and_update'),
                                     QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton(trans.t('btn_continue'),
                             QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is not cancel_tasks:
            return False
        self._show_status_message(trans.t('msg_update_cancel_stopping'), 5000)
        self._cancel_running_work()
        return True

    def prepare_for_update_restart(self) -> bool:
        """Protect every kind of work before the swap script starts.

        Used by all restart entries. Returns False when the user decided to
        keep working, or when the session could not be saved; the staged build
        stays available in both cases.
        """
        editor = QApplication.activeModalWidget()
        if editor is not None and editor is not self:
            editor.raise_()
            editor.activateWindow()
            return False
        # 1) Running tasks: stop them cooperatively only after confirmation.
        if self._has_work_in_flight() and not self._confirm_update_quit():
            return False
        if self._has_work_in_flight():
            self._show_status_message(trans.t('msg_update_cancel_stopping'), 5000)
            return False
        # 2) Unexported result pixels: export, discard or cancel the restart.
        if hasattr(self, "output_manager") \
                and not self.output_manager.confirm_unexported_on_exit():
            return False
        # 3) Unsaved project state: save it, or let the user cancel.
        if getattr(self, "_project_dirty", False):
            if not self._confirm_discard_project_changes(
                    "msg_discard_unsaved_project_quit"):
                return False
        # 4) Persist preferences and a fresh session snapshot. Without a
        #    snapshot the relaunched build has nothing trustworthy to restore,
        #    so the update must not proceed.
        try:
            self.persist_settings()
        except OSError:
            show_warning_box(self, trans.t('update_check_failed_title'),
                             trans.t('msg_update_snapshot_failed'))
            return False
        from utils import recovery
        if not recovery.write_snapshot(self):
            show_warning_box(self, trans.t('update_check_failed_title'),
                             trans.t('msg_update_snapshot_failed'))
            return False
        return True

    def _cancel_running_work(self) -> None:
        """Cooperative stop of every long-running job (no terminate())."""
        render_manager = getattr(self, "render_manager", None)
        worker = getattr(render_manager, "worker", None) if render_manager else None
        if worker is not None and worker.isRunning():
            try:
                worker.is_cancelled = True
            except Exception:
                pass
        batch_manager = getattr(self, "batch_manager", None)
        if batch_manager is not None:
            try:
                batch_manager._cancel_running_batch()
            except Exception:
                pass
        roi_worker = getattr(self, "roi_alignment_worker", None)
        if roi_worker is not None:
            try:
                roi_worker.cancel()
            except Exception:
                pass

    def _open_releases_page(self) -> None:
        from utils.updater import RELEASES_PAGE_URL
        QDesktopServices.openUrl(QUrl(RELEASES_PAGE_URL))

    def _quit_for_update(self) -> None:
        """The swap script is running: close so it can replace the binaries.

        close() (not QApplication.quit()): quit() tears the app down without
        waiting for the window's closeEvent cleanup, and the swap script's
        robocopy can then still hit locked files — the silent failure that
        made the app "disappear" and never come back updated.
        """
        self._restart_for_update = True
        self.close()

    def set_auto_download(self, enabled: bool) -> None:
        """Settings menu: download updates in the background while idle."""
        self.update_manager.set_auto_download(enabled)
        self.persist_settings()

    def set_reverse_order(self, enabled: bool) -> None:
        """右面板开关：反转帧序（倒序栈，仅影响 OpenFocus A/B/C）。"""
        self.stack_reverse_order = bool(enabled)
        self.persist_settings()

    def set_dtcwt_levels(self, levels: int) -> None:
        """Remember the DTCWT decomposition level count (compare adoption)."""
        self.dtcwt_levels = max(1, min(6, int(levels)))
        self.persist_settings()

    def set_smoothing_for_method(self, method: str, value: float) -> None:
        """Set one method's smoothing sigma without switching the selection."""
        self.__dict__.setdefault("_smoothing_by_method", {})[
            str(method)] = float(value)
        if str(method) == self.current_fusion_method():
            self.set_ofb_depth_smoothing(float(value))
        self.persist_settings()

    def set_brightness_option(self, option: str, method: str,
                              enabled: bool) -> None:
        """Set exposure/base correction for a specific method."""
        self.__dict__.setdefault("_brightness_by_method", {})[
            (option, str(method))] = bool(enabled)
        if str(method) == self.current_fusion_method():
            self._set_brightness_option(option, enabled)
        self.persist_settings()

    def set_ai_mode(self, mode: str) -> None:
        """右面板单选：AI 输出模式（soft=概率加权，hard=逐像素取帧）。"""
        mode = mode if mode in ("soft", "hard") else "soft"
        self.ai_mode = mode
        from PyQt6.QtCore import QSignalBlocker
        for name, selected in (("radio_ai_soft", mode == "soft"),
                               ("radio_ai_hard", mode == "hard")):
            control = getattr(self, name, None)
            if control is not None:
                with QSignalBlocker(control):
                    control.setChecked(selected)
        self.persist_settings()

    # 两个亮度选项都按算法独立记忆：它们是"这次融合用哪条曝光策略"，
    # 而不是全局偏好。OpenFocus A/B/C 默认两者都开（它们没有内置标定，
    # 这正是它们与 Helicon 的 A/B/C 最接近的行为）；其余算法沿用历史
    # 默认（曝光匹配开、亮度漂移关）。
    _BRIGHTNESS_DEFAULTS = {
        "exposure_match": {"openfocus_a": True, "openfocus_b": True,
                           "openfocus_c": True},
        "base_correct": {"openfocus_a": True, "openfocus_b": True,
                         "openfocus_c": True},
    }

    def brightness_option_for(self, option: str, method: str = None) -> bool:
        """Value of 'exposure_match' / 'base_correct' for *method*."""
        method = method or self.current_fusion_method()
        memory = self.__dict__.setdefault("_brightness_by_method", {})
        key = (option, method)
        if key in memory:
            return bool(memory[key])
        defaults = self._BRIGHTNESS_DEFAULTS.get(option, {})
        if method in defaults:
            return bool(defaults[method])
        # legacy global default: exposure matching on, base correction off
        return option == "exposure_match"

    def _set_brightness_option(self, option: str, enabled: bool) -> None:
        method = self.current_fusion_method()
        self.__dict__.setdefault("_brightness_by_method", {})[
            (option, method)] = bool(enabled)
        if option == "exposure_match":
            self.exposure_match = bool(enabled)
        else:
            self.base_frame_correct = bool(enabled)
        self.persist_settings()

    def _sync_brightness_checkboxes(self, method: str = None) -> None:
        """Show the selected method's own values in the two checkboxes."""
        for widget_name, option in (
                ("chk_exposure_match", "exposure_match"),
                ("chk_base_correct", "base_correct")):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setChecked(self.brightness_option_for(option, method))
            widget.blockSignals(False)

    def set_base_correct(self, enabled: bool) -> None:
        """右面板开关：融合后的低频基底校正（按算法记忆）。"""
        self._set_brightness_option("base_correct", enabled)

    def set_exposure_match(self, enabled: bool) -> None:
        """右面板开关：融合前把整栈曝光标定到末帧（按算法记忆）。"""
        self._set_brightness_option("exposure_match", enabled)

    def _show_update_result(self, state: str, tag: str, page_url: str,
                            zip_url: str = "", setup_url: str = "") -> None:
        from PyQt6.QtWidgets import QMessageBox

        if state == "update":
            manager = self._update_manager()
            staged = bool(manager.staged_tag())
            # A background check must never open a modal box, but it does show
            # the in-window banner: without it the startup check was invisible
            # to anyone who does not watch the status bar.
            self._show_update_banner(tag, ready=staged)
            if not manager.manual_check:
                return
            box = QMessageBox(self)
            style_message_box(box)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle(trans.t('update_available_title'))
            can_one_click = (
                getattr(sys, "frozen", False)
                and sys.platform in ("win32", "darwin")
                and (bool(zip_url) or staged)
            )
            # 更新说明：把已安装版本到最新版之间每个 release 的正文都带上，
            # 放"详细信息"按钮里（用户特别要求逐版本展示，不只是提示有更新）
            notes = getattr(self.update_manager, "notes", "") or ""
            if notes:
                box.setDetailedText(notes)
            if staged:
                box.setText(trans.t('update_ready_text').format(version=tag)
                            + ("\n" + trans.t('update_notes_hint') if notes else ""))
                install_btn = box.addButton(
                    trans.t('btn_restart_now'), QMessageBox.ButtonRole.AcceptRole)
                open_btn = box.addButton(
                    trans.t('btn_open_downloads'), QMessageBox.ButtonRole.ActionRole)
                box.addButton(trans.t('btn_later'), QMessageBox.ButtonRole.RejectRole)
            elif can_one_click:
                box.setText(trans.t('update_available_oneclick_text').format(version=tag)
                            + ("\n" + trans.t('update_notes_hint') if notes else ""))
                install_btn = box.addButton(
                    trans.t('btn_update_restart'), QMessageBox.ButtonRole.AcceptRole)
                open_btn = box.addButton(
                    trans.t('btn_open_downloads'), QMessageBox.ButtonRole.ActionRole)
                box.addButton(trans.t('btn_later'), QMessageBox.ButtonRole.RejectRole)
            else:
                box.setText(trans.t('update_available_text').format(version=tag)
                            + ("\n" + trans.t('update_notes_hint') if notes else ""))
                download_btn = box.addButton(
                    trans.t('btn_download_install'), QMessageBox.ButtonRole.AcceptRole)
                open_btn = box.addButton(
                    trans.t('btn_open_downloads'), QMessageBox.ButtonRole.ActionRole)
                box.addButton(trans.t('btn_later'), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if staged and clicked is install_btn:
                self.install_update_now()
            elif can_one_click and clicked is install_btn:
                self._apply_update_and_restart(zip_url, tag)
            elif (not can_one_click) and clicked is download_btn:
                self._download_and_launch_update(setup_url, tag)
            elif clicked is open_btn:
                QDesktopServices.openUrl(QUrl(page_url))
        elif state == "latest":
            show_message_box(
                self,
                trans.t('update_up_to_date_title'),
                trans.t('update_up_to_date_text'),
                icon=QMessageBox.Icon.Information,
            )
        else:
            show_message_box(
                self,
                trans.t('update_check_failed_title'),
                trans.t('update_check_failed_text'),
                icon=QMessageBox.Icon.Warning,
            )

    # --- Project files (.ofproj) ---

    def open_project_dialog(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        from utils.project_file import validate_project_for_open, apply_project, PROJECT_SUFFIX
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        if not self._confirm_discard_project_changes(
                "msg_discard_unsaved_project"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, trans.t('menu_open_project'),
            get_last_dialog_dir(), f"OpenFocus Project (*{PROJECT_SUFFIX})")
        if not path:
            return
        ok, err, state = validate_project_for_open(self, path)
        if not ok:
            if err:
                show_warning_box(self, trans.t("msg_error"), err)
            return
        set_last_dialog_dir(path)
        def loaded():
            self.project_path = path
            self._clear_project_dirty()
            if state.get("sources_relocated"):
                self._mark_project_dirty()
            self.add_recent_project(path)
        apply_project(self, state, on_loaded=loaded)

    def save_project_dialog(self, save_as: bool = False) -> None:
        from PyQt6.QtWidgets import QFileDialog
        from utils.project_file import save_project, suggest_project_path, PROJECT_SUFFIX
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        import os as _os
        target = getattr(self, "project_path", None)
        if save_as or not target:
            suggested = suggest_project_path(self) or _os.path.join(
                get_last_dialog_dir() or _os.path.expanduser("~"),
                "OpenFocus" + PROJECT_SUFFIX)
            path, _ = QFileDialog.getSaveFileName(
                self, trans.t('menu_save_project'), suggested,
                f"OpenFocus Project (*{PROJECT_SUFFIX})")
            if not path:
                return
            if not path.endswith(PROJECT_SUFFIX):
                path += PROJECT_SUFFIX
            set_last_dialog_dir(path)
            target = path
        ok, err = save_project(self, target)
        if ok:
            self.project_path = target
            self._clear_project_dirty()
            # A project stores input references and restorable settings, not
            # result pixels: say so, and keep the count of results that still
            # need an image export visible in the same message.
            message = trans.t('msg_project_saved_config').format(path=target)
            try:
                pending = len(self.output_manager.protected_records())
            except Exception:
                pending = 0
            if pending:
                message += " " + trans.t(
                    'msg_project_unexported_note').format(n=pending)
            self._show_status_message(message, 8000)
        else:
            show_warning_box(self, trans.t("msg_error"),
                             trans.t('msg_project_save_failed').format(err=err))

    def _update_project_title(self) -> None:
        """Window title: project name plus an unsaved-changes marker.

        Every other editor marks a modified document; without it the user had
        no way to tell whether the current state was saved, and closing threw
        the state away silently.
        """
        import os as _os
        name = (_os.path.basename(self.project_path)
                if getattr(self, "project_path", None) else None)
        mark = "*" if getattr(self, "_project_dirty", False) else ""
        if name:
            self.setWindowTitle(f"OpenFocus — {name}{mark}")
        else:
            self.setWindowTitle(f"OpenFocus{mark}" if mark else "OpenFocus")

    def _mark_project_dirty(self) -> None:
        if not getattr(self, "_project_dirty", False):
            self._project_dirty = True
            self._update_project_title()

    def _clear_project_dirty(self) -> None:
        self._project_dirty = False
        self._update_project_title()

    def _confirm_discard_project_changes(self, action_key: str) -> bool:
        """Ask before an action that would drop unsaved project state."""
        if not getattr(self, "_project_dirty", False):
            return True
        box = QMessageBox(self)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t("msg_warning"))
        box.setText(trans.t(action_key))
        save_button = box.addButton(trans.t("menu_save_project"),
                                    QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton(trans.t("btn_discard_changes"),
                                QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(trans.t("btn_cancel"),
                               QMessageBox.ButtonRole.RejectRole)
        # Default to the safe choice: the old default discarded the work.
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_button:
            self.save_project_dialog(save_as=False)
            # Saving may have been cancelled or failed; only continue when the
            # project is actually clean.
            return not getattr(self, "_project_dirty", False)
        return clicked is discard

    def load_demo_stack(self) -> None:
        """Load the bundled demo stack (no downsample dialog)."""
        from utils import resource_path
        demo_dir = resource_path("assets", "demo_stack")
        if not os.path.isdir(demo_dir):
            show_warning_box(self, trans.t("msg_warning"), trans.t("msg_demo_missing"))
            return
        self.current_folder_path = demo_dir
        self.source_manager._start_load_worker(
            folder=demo_dir, scale=1.0, append=False,
            on_success=lambda: self.add_recent_file(demo_dir) if hasattr(self, "add_recent_file") else None,
        )

    def _apply_update_and_restart(self, zip_url: str, tag: str) -> None:
        """One-click self-update: download if needed, then swap and restart.

        The download/stage/apply state machine lives in UpdateManager; this
        keeps the entry point the UI (and the tests) already use.
        """
        from utils import self_update
        from utils.updater import RELEASES_PAGE_URL

        if not self_update.is_frozen():
            QDesktopServices.openUrl(QUrl(RELEASES_PAGE_URL))
            return
        update_manager_for(self).download_and_apply(tag, zip_url)

    def _stage_and_restart(self, zip_path: str, staging_root: str, tag: str) -> None:
        """Extract the downloaded zip and hand off to the apply script.

        Delegates to UpdateManager so the download path, the "already staged"
        path and the tests share one implementation.
        """
        update_manager_for(self).stage_and_handoff(zip_path, staging_root, tag)

    def _download_and_launch_update(self, url: str, tag: str) -> None:
        """Download the installer on a worker thread, then offer to run it.

        All UI updates happen through queued signals - never from the
        download thread directly.
        """
        running = getattr(self, "_update_download_worker", None)
        if running is not None and running.isRunning():
            show_warning_box(self, trans.t('update_check_failed_title'),
                             trans.t('msg_update_in_progress'))
            return

        import tempfile
        from utils import updater
        from utils.self_update import sanitize_tag

        if not url:
            QDesktopServices.openUrl(QUrl(updater.RELEASES_PAGE_URL))
            return

        dest = os.path.join(
            tempfile.gettempdir(),
            f"OpenFocus-{sanitize_tag(tag)}-{sanitize_tag(os.path.basename(url))}")

        class _DownloadWorker(QThread):
            progress = pyqtSignal(int, int)
            done = pyqtSignal(bool, str)

            def __init__(self):
                super().__init__()
                self._cancelled = False

            def cancel(self) -> None:
                self._cancelled = True

            def run(self) -> None:
                try:
                    updater.download_to_file(
                        url, dest,
                        on_progress=lambda d, tt: self.progress.emit(d, tt),
                        should_cancel=lambda: self._cancelled)
                    self.done.emit(True, dest)
                except Exception as exc:
                    self.done.emit(False, str(exc))

        # A visible, working Cancel button. Passing "" used to create a
        # QPushButton with no text (the grey empty box in the dialog) that
        # cancelled nothing and left the download running invisibly.
        self._update_progress = QProgressDialog(
            trans.t('update_downloading'), trans.t('btn_cancel_download'),
            0, 100, self)
        self._update_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._update_progress.setMinimumDuration(0)
        self._update_progress.setWindowTitle(trans.t('update_downloading'))

        def _on_download_progress(done: int, total: int) -> None:
            self._update_progress.setValue(int(done * 100 / max(1, total)))

        def _on_download_done(ok: bool, message: str) -> None:
            self._update_progress.close()
            if not ok:
                if message != "download cancelled":
                    show_warning_box(self, trans.t('update_check_failed_title'), message)
                return
            reply = QMessageBox.question(
                self, trans.t('update_downloaded_title'),
                trans.t('update_run_installer'),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if sys.platform == "win32":
                    os.startfile(message)  # noqa: S606 - user-confirmed installer
                else:
                    import subprocess
                    subprocess.Popen(["open", message])

        worker = _DownloadWorker()
        worker.progress.connect(_on_download_progress)
        worker.done.connect(_on_download_done)
        self._update_progress.canceled.connect(worker.cancel)
        self._update_download_worker = worker  # keep a reference
        worker.start()

    def remove_scale_bar(self, where: str) -> None:
        """Remove the bar from one canvas, or from everywhere.

        "Apply to" is a scope, so removing one canvas leaves the bar on the
        other; removing the last canvas (or choosing "everywhere") turns the
        bar off. The old handler only rewrote the scope, so "remove
        everywhere" set it back to `all` and the bar stayed visible.
        """
        previous = self._capture_scalebar_state()
        scope = getattr(self, "scale_bar_target", "all") or "all"
        if getattr(self, "scale_bar_enabled", False):
            if where == "all" or scope == where or (
                    where == "output" and scope == "registered"):
                self.scale_bar_enabled = False
            elif scope == "all":
                self.scale_bar_target = "output" if where == "source" else "source"
        if where in ("all", "output"):
            for row in range(self.output_list.count()):
                record = self.output_manager.record_for_row(row)
                if record is not None:
                    record.scale_bar_config = None
                    record.scale_bar_initialized = True
        if self._capture_scalebar_state() == previous:
            self.statusBar().showMessage(trans.t("scalebar_none_msg"), 6000)
            return
        self.push_scalebar_undo(previous)
        self.persist_settings()
        self.refresh_annotation_views()
        # Complete literals: the i18n scanner cannot resolve a computed key.
        message = {
            "source": trans.t("scalebar_removed_source"),
            "output": trans.t("scalebar_removed_output"),
            "all": trans.t("scalebar_removed_all"),
        }.get(where, trans.t("scalebar_none_msg"))
        self.statusBar().showMessage(message, 6000)

    def refresh_annotation_views(self) -> None:
        """Redraw both canvases after label or scale-bar state changes."""
        self.invalidate_annotation_caches()
        self.output_manager.refresh_annotation_export_state()
        try:
            if self.current_display_index is not None and self.stack_images:
                self.update_source_view(self.current_display_index)
        except Exception:
            pass
        try:
            item = self.output_list.currentItem()
            if item is not None:
                self.output_manager.display_output_image_in_result_view(item)
            elif self.fusion_result is not None:
                self.output_manager.show_fusion_result()
            elif self.registration_results:
                self.output_manager.show_registration_result(
                    getattr(self, "current_result_index", 0))
        except Exception:
            pass
        if getattr(self, "wipe_active", False):
            self._update_wipe_images()

    def show_scale_bar_dialog(self) -> None:
        """Configure the exported-image scale bar with a live preview."""
        from dialogs.scalebar import ScaleBarDialog
        ScaleBarDialog(self).exec()

    def show_depth_map_dialog(self) -> None:
        """Focus-position (pseudo-depth) map viewer and exporter."""
        from dialogs.depthmap import DepthMapDialog
        DepthMapDialog(self).exec()

    def show_retouch_dialog(self, item=None) -> None:
        """Retouch one output selected by identity, not by row order.

        `item` is the entry the user right-clicked; menu/shortcut calls pass
        None, in which case the visible selection decides.  With no selection
        a chooser lists every output and marks which ones have provenance.
        """
        from dialogs.retouch import RetouchDialog, RetouchResultChooser

        # QAction.triggered passes a `checked` bool; treat it as "no item".
        if item is not None and not hasattr(item, "data"):
            item = None

        if self.output_list.count() == 0:
            show_warning_box(self, trans.t("msg_no_result_title"),
                             trans.t("retouch_need_result"))
            return

        if item is None:
            selected = self.output_list.selectedItems()
            if len(selected) > 1:
                show_warning_box(self, trans.t("msg_warning"),
                                 trans.t("retouch_multi_selection"))
                return
            if len(selected) == 1:
                item = selected[0]

        if item is None:
            chooser = RetouchResultChooser(self)
            if not chooser.exec():
                return
            item = chooser.selected_item()
            if item is None:
                return

        self.output_list.clearSelection()
        item.setSelected(True)
        self.output_list.setCurrentItem(item)

        while True:
            record = self.output_manager.record_for_item(item)
            if record is None or not record.retouchable:
                show_warning_box(self, trans.t("msg_warning"),
                                 trans.t("retouch_no_source"))
                return

            try:
                dlg = RetouchDialog(
                    self, record=record,
                    apply_callback=self._apply_retouch_output)
            except Exception as exc:  # noqa: BLE001 - preparation failure
                show_message_box(
                    self, trans.t("msg_error"), trans.t("msg_proc_error"),
                    str(exc), QMessageBox.Icon.Critical)
                return
            dlg.exec()
            if not getattr(dlg, "change_requested", False):
                return

            chooser = RetouchResultChooser(self)
            if not chooser.exec():
                return
            new_item = chooser.selected_item()
            if new_item is None:
                return
            item = new_item
            self.output_list.clearSelection()
            item.setSelected(True)
            self.output_list.setCurrentItem(item)

    def _apply_retouch_output(self, base_record, image):
        """Publish the edited buffer; the dialog stays open on failure."""
        try:
            self.output_manager.add_retouched_output(base_record, image)
            self.statusBar().showMessage(
                trans.t("retouch_applied") + " · " + trans.t("retouch_unexported"),
                8000)
            try:
                from utils import recovery
                recovery.write_snapshot(self)
            except Exception:
                pass
            return True, ""
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            return False, str(exc)

    def show_advanced_settings_dialog(self) -> None:
        """One page for tiling, registration, AI and the convenience toggles."""
        from dialogs.advanced_settings import AdvancedSettingsDialog
        AdvancedSettingsDialog(self).exec()

    def show_compare_dialog(self) -> None:
        """Side-by-side comparison of several fusion methods on this stack."""
        if not self.raw_images or len(self.raw_images) < 2:
            show_warning_box(self, trans.t("msg_no_images_title"),
                             trans.t("msg_render_need_images_text"))
            return
        from dialogs.compare import CompareDialog
        CompareDialog(self).exec()

    def show_dust_map_dialog(self) -> None:
        """Flat-field (dust map) correction from a defocused white shot."""
        from dialogs.dustmap import DustMapDialog
        DustMapDialog(self).exec()

    def apply_flat_field_to_stack(self, gain, remove_index=None, prepared=None) -> bool:
        """Commit corrected source/base frames and optional removal together."""
        import time as _time
        from utils.dust_map import apply_flat_field_stack
        t0 = _time.time()
        # Apply to COPIES first: the previous order wrote raw_images and only
        # then tried base_images, so a failure on the second call left the
        # stack half-corrected with no way back.
        try:
            if prepared is None:
                corrected_raw = apply_flat_field_stack(self.raw_images, gain)
                corrected_base = (
                    apply_flat_field_stack(self.base_images, gain)
                    if getattr(self, "base_images", None) else None)
            else:
                corrected_raw, corrected_base = prepared
            if len(corrected_raw) != len(self.raw_images):
                return False
            if corrected_base is not None and len(corrected_base) != len(self.base_images):
                return False
            if remove_index is not None and not 0 <= remove_index < len(corrected_raw):
                return False
            corrected_raw = [image for i, image in enumerate(corrected_raw)
                             if i != remove_index]
            if corrected_base is not None:
                corrected_base = [image for i, image in enumerate(corrected_base)
                                  if i != remove_index]
        except (ValueError, MemoryError) as exc:
            # A frame whose size disagrees with the white field must not die
            # silently in a Qt slot (the excepthook only writes a crash file).
            import logging
            logging.getLogger("openfocus").warning("flat-field apply failed: %s", exc)
            show_warning_box(self, trans.t("msg_error"),
                             trans.t("dust_err_not_white"))
            return False
        self.push_undo()
        self.raw_images = corrected_raw
        if corrected_base is not None:
            self.base_images = corrected_base
        if remove_index is not None:
            self.image_filenames = [name for i, name in enumerate(self.image_filenames)
                                    if i != remove_index]
            self.image_source_paths = [path for i, path in enumerate(self.image_source_paths)
                                       if i != remove_index]
            self.excluded_frames = {i - (i > remove_index)
                                    for i in self.excluded_frames if i != remove_index}
        self.unrecoverable_source_changes = list(getattr(
            self, "unrecoverable_source_changes", []) or []) + [trans.t('dust_title')]
        if hasattr(self.get_or_create_source_pixmap, "cache_clear"):
            self.get_or_create_source_pixmap.cache_clear()
        self.transform_manager._invalidate_processing_results(
            clear_output_view=True, preserve_outputs=True)
        self.transform_manager.reload_image_stack(
            initial_index=getattr(self, "current_img_index", 0))
        self.statusBar().showMessage(
            trans.t('dust_applied').format(n=len(self.raw_images),
                                           t=_time.time() - t0), 5000)
        return True

    def show_quick_start(self) -> None:
        from dialogs.welcome import WelcomeDialog
        WelcomeDialog(self, main_window=self).exec()

    def open_logs_folder(self) -> None:
        """Open the folder that contains the application log files."""
        from PyQt6.QtGui import QDesktopServices
        from utils.logging_setup import logs_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_dir())))


    def get_or_create_source_pixmap(self, index: int):
        """Return the display pixmap for a frame, rendering it on first use.

        Full-resolution pixmaps are generated lazily: pre-rendering every
        frame costs seconds and hundreds of MB on large stacks while only
        one frame is visible at a time.
        """
        if 0 <= index < len(self.stack_images):
            if self.stack_images[index] is None and index < len(self.raw_images):
                made = self.image_loader.create_pixmaps(
                    [self.raw_images[index]], max_size=None)
                self.stack_images[index] = made[0]
                # 惰性生成但不设上限等于无界缓存：只保留当前帧±1，
                # 跳回远处帧时按需重建（这正是惰性机制的本意）
                for j in range(len(self.stack_images)):
                    if abs(j - index) > 1:
                        self.stack_images[j] = None
        return self.stack_images[index] if 0 <= index < len(self.stack_images) else None

    def update_source_view(self, index):
        if not self.stack_images:
            return
            
        if 0 <= index < len(self.stack_images):
            self.current_display_index = index
            
            try:
                # 获取原始图片并根据需要叠加标签
                original_pixmap = self.get_or_create_source_pixmap(index)
                from utils import scalebar
                display_pixmap = self.label_manager.apply_labels_to_source_pixmap(
                    original_pixmap, index,
                    scale_cfg=scalebar.export_cfg(self, target="source"))

                # 使用自定义label处理缩放和缩放重置
                self.lbl_source_img.set_display_pixmap(display_pixmap)
                
                # 更新文字
                self.lbl_stack_info.setText(f"{index + 1} / {len(self.stack_images)}")
                
                # 只有当列表选中项和当前slider不一致时才去设置列表，防止信号死循环
                if self.file_list.currentRow() != index:
                    self.file_list.setCurrentRow(index)

                self.sharpness_curve.set_current(index)

                # Keep the wipe view's left side following the current frame
                if self.wipe_active and self.chk_wipe_follow_left.isChecked():
                    self.slider_wipe_left.setValue(index)
                    self._update_wipe_images()
            except Exception as e:
                show_message_box(
                    self,
                    trans.t("msg_display_error_title"),
                    trans.t("msg_display_source_failed_text"),
                    f"Error: {str(e)}",
                    QMessageBox.Icon.Critical
                )
    
    
    def current_fusion_method(self) -> str:
        """当前选中的融合算法代码（下拉单选的唯一读取入口）。"""
        from ui.right_panel import FUSION_ITEMS, FUSION_DEFAULT
        code = self.combo_fusion.currentData()
        codes = {c for c, _k, _h in FUSION_ITEMS}
        return code if code in codes else FUSION_DEFAULT

    def current_alignment_method(self) -> str:
        """当前选中的对齐算法代码：none | homography | ecc | openfocus。"""
        code = self.combo_align.currentData()
        return code if code in ("none", "homography", "ecc", "openfocus") else "ecc"

    def set_fusion_method(self, code: str) -> None:
        i = self.combo_fusion.findData(code)
        if i >= 0:
            self.combo_fusion.setCurrentIndex(i)

    def set_alignment_method(self, code: str) -> None:
        i = self.combo_align.findData(code)
        if i >= 0:
            self.combo_align.setCurrentIndex(i)

    def handle_method_selection(self, *_args):
        """融合算法下拉切换（单选语义由下拉框保证，这里做联动刷新）。"""
        # update_slider_availability 已通过信号连接；保留此槽位给
        # 未来按方法变化的 UI 联动
        pass

    def handle_kernel_slider_change(self, value):
        """确保核大小始终为奇数，并更新显示标签"""
        adjusted = int(value)
        if adjusted % 2 == 0:
            if adjusted >= self.slider_smooth.maximum():
                adjusted -= 1
            else:
                adjusted += 1
            self.slider_smooth.blockSignals(True)
            self.slider_smooth.setValue(adjusted)
            self.slider_smooth.blockSignals(False)

        self.lbl_smooth_value.setText(str(adjusted))

    # 滑块含义按算法不同：模式 → (i18n 标签键, 默认值)
    _KERNEL_LABELS = {
        "guided": "label_kernel_guided",
        "dct": "label_kernel_dct",
        "gfg": "label_kernel_gfg",
        "ofab": "label_kernel_of",
    }

    # 方法 → (滑块模式, 范围上限, 默认 kernel)。OpenFocus A/B/C：滑块是方差
    # 窗口（kernel = 2r+1）；判别性探针（报告 §6.7/§6.8）定 r=5 → 11，范围 1–21。
    _KERNEL_MODES = {
        "guided_filter": ("guided", 51, 31),
        "dct": ("dct", 51, 7),
        "gfgfgf": ("gfg", 51, 7),
        "openfocus_a": ("ofab", 21, 11),
        "openfocus_b": ("ofab", 21, 11),
        "openfocus_c": ("ofab", 21, 11),
    }

    def kernel_for_method(self, method: str) -> int:
        """Kernel the user has set for *method* (its own remembered value,
        or its default) — the slider means a different thing per method, so
        another method's value must never leak in (the depth-map dialog
        used the guided-filter radius 31 as DCT's median window)."""
        spec = self._KERNEL_MODES.get(method)
        if spec is None:
            return 7
        mode, _hi, default = spec
        if mode == getattr(self, "current_kernel_mode", None):
            return int(self.slider_smooth.value())
        return int(getattr(self, "_kernel_by_mode", {}).get(mode, default))

    def update_slider_availability(self):
        """根据选中的融合方法更新滑块的可用性、默认值和含义标签。

        每个方法的滑块值单独记忆：切换方法再切回来不再丢失用户调好的值。"""
        method = self.current_fusion_method()
        memory = self.__dict__.setdefault("_kernel_by_mode", {})
        previous = getattr(self, "current_kernel_mode", None)
        spec = self._KERNEL_MODES.get(method)
        if previous and spec and previous != spec[0]:
            memory[previous] = int(self.slider_smooth.value())
        elif previous and spec is None:
            memory[previous] = int(self.slider_smooth.value())
        if spec is not None:
            mode, hi, default = spec
            self.smooth_widget.setEnabled(True)
            self.slider_smooth.setRange(1, hi)
            if previous != mode:
                self.slider_smooth.setValue(min(hi, memory.get(mode, default)))
            self.current_kernel_mode = mode
        else:
            # AI has no kernel: leaving the row visible showed "Kernel: 11"
            # next to a slider that changed nothing.
            self.smooth_widget.setEnabled(False)
            self.slider_smooth.setRange(1, 51)
            self.current_kernel_mode = None
        # 平滑行：B 平滑深度图；A/C 平滑混合（Helicon 的 A/C 只有
        # Smoothing 一个参数）。三种方法共用同一行，标签与取值各自独立。
        if getattr(self, "depth_smooth_row", None) is not None:
            is_a = method == "openfocus_a"
            is_b = method == "openfocus_b"
            is_c = method == "openfocus_c"
            self.depth_smooth_row.setVisible(is_a or is_b or is_c)
            blend = is_a or is_c
            label_key = ("label_blend_smoothing" if blend
                         else "label_depth_smoothing")
            hint_key = ("label_blend_smoothing_hint" if blend
                        else "label_depth_smoothing_hint")
            self.lbl_depth_smooth.setText(trans.t(label_key))
            self.lbl_depth_smooth.setToolTip(trans.t(hint_key))
            self.spin_depth_smooth.setToolTip(trans.t(hint_key))
            value = self.smoothing_for_method(method)
            self.spin_depth_smooth.blockSignals(True)
            self.spin_depth_smooth.setValue(float(value))
            self.spin_depth_smooth.blockSignals(False)
        # C 没有 Radius（Helicon 的 Method C 只有 Smoothing）：隐藏滑块
        # 而不是留一个改了也不影响输出的控件。
        # AI 输出模式只在选中神经网络时出现：其他算法没有概率图可分配。
        if getattr(self, "ai_mode_widget", None) is not None:
            self.ai_mode_widget.setVisible(method == "stackmffv3")
        if getattr(self, "smooth_widget", None) is not None:
            # 只在当前方法真的用得到 kernel 时显示：C 没有半径参数，AI
            # 也没有，留着只会显示一个改了不影响输出的数字。
            self.smooth_widget.setVisible(spec is not None
                                          and method != "openfocus_c")
        # Duck-typed hosts (tests bind update_slider_availability onto a stub)
        # may not provide the checkbox sync.
        sync = getattr(self, "_sync_brightness_checkboxes", None)
        if callable(sync):
            sync(method)
        # 滑块含义标签随算法切换（每个算法的 kernel 含义不同）
        label_key = self._KERNEL_LABELS.get(self.current_kernel_mode,
                                            "label_kernel")
        self.lbl_kernel.setText(trans.t(label_key))
    
    def update_result_view(self, index):
        """更新Output区域显示的图像（用于配准结果或ROI模式下的对齐图像）"""
        # 如果ROI模式激活，显示对齐后的图像栈
        if getattr(self, 'roi_mode_active', False) and self.roi_aligned_images:
            self._display_roi_aligned_image(index)
        else:
            self.output_manager.show_registration_result(index)

    # --- Wipe A/B compare ---

    def _bgr_to_pixmap(self, image) -> QPixmap:
        return cv2_to_pixmap(image)

    def _wipe_pixmap(self, kind: str, index: int, image):
        """Pixmap for one side of the wipe, cached across slider steps.

        Both halves used to be re-converted from numpy on every step of
        either slider: 12 MP -> uint8 -> RGB -> pixmap is ~27 ms per side,
        so dragging one slider re-rendered the other side for nothing.
        The cache holds the source array itself, so its identity cannot be
        recycled by a later allocation while an entry is alive.

        The label revision and each output's scale-bar configuration are
        part of the cache key, so annotation edits invalidate both sides.
        """
        if image is None:
            return None
        cache = getattr(self, "_wipe_pix_cache", None)
        if cache is None:
            cache = self._wipe_pix_cache = {}
        record = self.output_manager.record_for_image(image) if kind == "right" else None
        target = "output" if record is not None or kind == "right" and self.fusion_results else "registered" if kind == "right" else "input"
        baked = record is not None and record.annotations_baked
        labeled = not baked and self.label_manager.is_enabled_for_target(target)
        label_revision = int(getattr(self.label_manager, "revision", 0))
        from utils import scalebar
        scale_cfg = scalebar.export_cfg(
            self, target="source" if target == "input" else target, record=record)
        # Include the scaled-bar configuration in the key: without it a
        # cached pixmap kept showing a bar after it was removed (or kept
        # hiding one after it was added), until the frame changed.
        scale_key = (None if scale_cfg is None
                     else tuple(sorted(scale_cfg.items())))
        entry = cache.get(kind)
        if (entry is not None and entry[0] is image and entry[1] == index
                and entry[2] == labeled and entry[3] == scale_key
                and entry[4] == label_revision):
            return entry[5]
        display = image
        if labeled or scale_cfg:
            display = self.label_manager.display_bgr_image(
                target, image, index, scale_cfg=scale_cfg,
                output_id=record.output_id if record else None)
        pixmap = self._bgr_to_pixmap(display)
        if len(cache) > 4:
            cache.clear()
        cache[kind] = (image, index, labeled, scale_key, label_revision,
                       pixmap)
        return pixmap

    def set_wipe_mode(self, enabled: bool) -> None:
        """Toggle the A/B wipe compare page of the result panel."""
        if not enabled:
            self.wipe_active = False
            self.wipe_bar.setVisible(False)
            self.view_stack.setCurrentIndex(0)
            self.wipe_widget.clear()
            return

        if getattr(self, 'roi_mode_active', False):
            self.btn_wipe.setChecked(False)
            return

        if self.fusion_result is None and not self.fusion_results and not self.registration_results:
            show_warning_box(self, trans.t("msg_warning"), trans.t("wipe_need_result"))
            self.btn_wipe.setChecked(False)
            return

        self.wipe_active = True
        self._setup_wipe_sliders()
        self.wipe_bar.setVisible(True)
        self.view_stack.setCurrentIndex(1)
        self._update_wipe_images()

    def _setup_wipe_sliders(self) -> None:
        """Size the A/B frame sliders to the current stacks (follow checkboxes
        keep their state across renders)."""
        self.slider_wipe_left.blockSignals(True)
        self.slider_wipe_left.setRange(0, max(0, len(self.raw_images) - 1))
        if self.chk_wipe_follow_left.isChecked() and 0 <= self.current_display_index:
            self.slider_wipe_left.setValue(self.current_display_index)
        self.slider_wipe_left.setEnabled(not self.chk_wipe_follow_left.isChecked())
        self.slider_wipe_left.blockSignals(False)

        self.slider_wipe_right.blockSignals(True)
        # A registration-only render fills registration_results, not
        # fusion_results — size the slider to whatever B can actually show, or
        # both halves of the comparison go blank.
        self._wipe_right_count = len(self.fusion_results) or len(self.registration_results)
        self.slider_wipe_right.setRange(0, max(0, self._wipe_right_count - 1))
        self.slider_wipe_right.setEnabled(
            not self.chk_wipe_follow_right.isChecked() and self._wipe_right_count > 0)
        self.slider_wipe_right.blockSignals(False)

    def refresh_wipe_controls(self) -> None:
        """Re-range the wipe sliders after the stacks changed (new render,
        frames appended/deleted), keeping the user's positions where valid."""
        self._setup_wipe_sliders()
        self._update_wipe_images()

    def _on_wipe_follow_left_toggled(self, checked: bool) -> None:
        self.slider_wipe_left.setEnabled(not checked)
        if checked and 0 <= self.current_display_index:
            self.slider_wipe_left.setValue(self.current_display_index)
        if self.wipe_active:
            self._update_wipe_images()

    def _on_wipe_left_slider_changed(self, value: int) -> None:
        if not self.wipe_active:
            return
        self.lbl_wipe_left_index.setText(trans.t("wipe_frame_fmt").format(value + 1))
        self._update_wipe_images()

    def _on_wipe_follow_right_toggled(self, checked: bool) -> None:
        self.slider_wipe_right.setEnabled(
            not checked and getattr(self, "_wipe_right_count", 0) > 0)
        if self.wipe_active:
            self._update_wipe_images()

    def _on_wipe_right_slider_changed(self, value: int) -> None:
        if not self.wipe_active:
            return
        self.lbl_wipe_right_index.setText(trans.t("wipe_result_fmt").format(value + 1))
        self._update_wipe_images()

    def _wipe_left_index(self) -> int:
        if self.chk_wipe_follow_left.isChecked():
            return self.current_display_index
        return self.slider_wipe_left.value()

    def _wipe_left_image(self):
        index = self._wipe_left_index()
        if 0 <= index < len(self.raw_images):
            return self.raw_images[index]
        return None

    def _wipe_right_index(self) -> int:
        if self.chk_wipe_follow_right.isChecked():
            # Follow the result actually shown: the latest fused result is a
            # single image (frame 1), a registration preview carries its own
            # frame number.
            if self.fusion_result is not None:
                return 0
            if self.registration_results:
                return max(0, self.current_result_index)
            return 0
        return self.slider_wipe_right.value()

    def _wipe_right_image(self):
        """Raw B-side frame; labels and bit-depth conversion happen once, in
        _wipe_pixmap, so the result can be cached by identity."""
        if self.chk_wipe_follow_right.isChecked():
            # The most recent render
            if self.fusion_result is not None:
                return self.fusion_result
            if self.registration_results:
                return self.registration_results[
                    self.current_result_index if self.current_result_index >= 0 else 0]
            return None
        index = self.slider_wipe_right.value()
        if 0 <= index < len(self.fusion_results):
            return self.fusion_results[index]
        # registration-only session: B shows the aligned frame instead
        if 0 <= index < len(self.registration_results):
            return self.registration_results[index]
        return None

    def _wipe_right_shows_registration(self) -> bool:
        """True when B falls back to the registration results."""
        if self.fusion_result is not None or self.fusion_results:
            return False
        return bool(self.registration_results)

    def _update_wipe_images(self) -> None:
        if not self.wipe_active:
            return

        left_image = self._wipe_left_image()
        right_image = self._wipe_right_image()
        left_pix = self._wipe_pixmap("left", self._wipe_left_index(), left_image)
        right_pix = self._wipe_pixmap("right", self._wipe_right_index(), right_image)

        if self.chk_wipe_follow_left.isChecked():
            left_idx = self.current_display_index
        else:
            left_idx = self.slider_wipe_left.value()
        left_title = (
            trans.t("wipe_frame_fmt").format(left_idx + 1)
            if 0 <= left_idx < len(self.raw_images)
            else "-"
        )
        if self.chk_wipe_follow_right.isChecked():
            # B is the latest render; in a registration-only session that is an
            # aligned frame, so do not label it "Fusion Result".
            right_title = (trans.t("wipe_opt_registered")
                           if self._wipe_right_shows_registration()
                           else trans.t("wipe_opt_result"))
        else:
            right_title = trans.t("wipe_result_fmt").format(self.slider_wipe_right.value() + 1)
        self.wipe_widget.set_images(left_pix, right_pix, f"A: {left_title}", f"B: {right_title}")

        # Keep the index labels in sync with whatever is being shown
        self.lbl_wipe_left_index.setText(left_title)
        self.lbl_wipe_right_index.setText(right_title)

    
    # --- 文件加载功能 ---
    
    def open_folder_dialog(self):
        """打开文件夹选择对话框"""
        self.source_manager.prompt_and_load_stack()
    
    def open_video_dialog(self):
        """打开视频文件选择对话框"""
        self.source_manager.prompt_and_load_video()
    
    def show_feedback_dialog(self) -> None:
        """Report a bug or suggestion, with the session log attached."""
        from dialogs.feedback import FeedbackDialog
        FeedbackDialog(self).exec()

    def show_environment_info(self):
        """显示环境信息对话框"""
        dialog = EnvironmentInfoDialog(self)
        dialog.exec()
    
    def display_fusion_result(self):
        """显示融合结果到右侧预览区"""
        self.output_manager.show_fusion_result()
    
    # --- 拖放功能 ---
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖动进入事件"""
        if self.source_manager.can_accept_drag(event):
            event.acceptProposedAction()
            return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """拖放事件"""
        self.source_manager.handle_drop_event(event)
    
    def on_splitter_moved(self, pos, index):
        """分割器移动时重新缩放图像"""
        # The image labels re-fit themselves in resizeEvent from the pixmaps
        # they already hold; rebuilding those pixmaps from the numpy stack
        # here cost 55-165 ms per event while dragging the splitter.
        self._refresh_image_views()

    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        self._refresh_image_views()

    def _refresh_image_views(self) -> None:
        """Repaint the image panels after a size change.

        MagnifierLabel (and WipeCompareWidget) recompute their fit, zoom and
        pan from the pixmap they already hold whenever they are resized, so
        a repaint is all that is needed — re-running the numpy -> uint8 ->
        pixmap pipeline here re-uploaded a 12 MP frame per resize event
        (measured 165 ms/event while dragging the window edge).
        """
        for label in (getattr(self, "lbl_source_img", None),
                      getattr(self, "lbl_result_img", None)):
            if label is not None and label.isVisible():
                label.update()
        wipe = getattr(self, "wipe_widget", None)
        if wipe is not None and self.wipe_active:
            wipe.update()

    def toggle_roi_mode(self, enabled: bool):
        """Toggle ROI selection mode. 
        When enabled, first align images using ECC, then display aligned stack on result panel for ROI selection.
        """
        if enabled:
            images, names, indices = self.effective_stack()
            if len(images) < 2:
                show_warning_box(self, trans.t("msg_warning"), trans.t("roi_need_images"))
                self.btn_preview_roi.blockSignals(True)
                self.btn_preview_roi.setChecked(False)
                self.btn_preview_roi.blockSignals(False)
                return

            # ROI 模式占用结果面板，进入前退出 wipe 对比
            if getattr(self, 'wipe_active', False):
                self.btn_wipe.setChecked(False)
            
            # 检查是否可以复用已有的对齐图像
            can_reuse = (
                bool(self.roi_aligned_images) and
                getattr(self, "roi_included_indices", ()) == tuple(indices) and
                getattr(self, "roi_source_revision", -1) == getattr(self, "source_revision", 0)
            )
            
            if can_reuse:
                # 直接使用已有的对齐图像
                self.roi_mode_active = True
                self._display_roi_aligned_image(0)
                
                # 启用滑动条
                self.result_slider.setRange(0, len(self.roi_aligned_images) - 1)
                self.result_slider.setValue(0)
                self.result_slider.setEnabled(True)
                self.result_control_bar.setVisible(True)
                
                # 启用右侧面板的ROI选择模式
                if hasattr(self, 'lbl_result_img'):
                    self.lbl_result_img.roi_mode = True
                return
            
            # 需要重新对齐，禁用按钮，显示处理中状态
            self.btn_preview_roi.setEnabled(False)
            self.btn_preview_roi.setText(trans.t("roi_aligning"))
            QApplication.processEvents()
            
            # 启动ECC配准线程
            # Snapshot the list (mid-run frame deletes mutate the live list)
            # and capture the stack identity so a stale result — finished
            # after the user loaded/rotated/deleted — is discarded instead
            # of being published for the wrong stack.
            self._roi_align_snapshot = list(images)
            self._roi_align_context = (getattr(self, "source_revision", 0),
                                       tuple(indices), tuple(names))
            self._roi_frame_ids = tuple(self.label_manager.frame_ids("input")[i]
                                        for i in indices)
            self.roi_alignment_worker = ROIAlignmentWorker(
                self._roi_align_snapshot,
                reg_downscale_width=self.reg_downscale_width,
                thread_count=self.thread_count
            )
            self.roi_alignment_worker.finished_signal.connect(self._on_roi_alignment_finished)
            self.roi_alignment_worker.error_signal.connect(self._on_roi_alignment_error)
            self.roi_alignment_worker.start()
        else:
            # 退出ROI模式
            self.roi_mode_active = False
            if hasattr(self, 'lbl_result_img'):
                self.lbl_result_img.roi_mode = False
            # 恢复右侧面板显示
            if self.fusion_result is not None:
                self.output_manager.show_fusion_result()
            elif self.registration_results:
                self.output_manager.show_registration_result(self.current_result_index)
            else:
                self.lbl_result_img.clear()
                self.lbl_result_img.setText(trans.t("result_hint"))
            # 清理ROI对齐图像
            self.roi_aligned_images = []

    def _on_roi_alignment_finished(self, aligned_images, alignment_time):
        """ROI配准完成的回调"""
        # Discard the result if the stack changed while ECC ran.
        _images, names, indices = self.effective_stack()
        context_now = (getattr(self, "source_revision", 0), tuple(indices), tuple(names))
        if context_now != getattr(self, "_roi_align_context", None):
            self.btn_preview_roi.setEnabled(True)
            self.btn_preview_roi.setText(trans.t("btn_roi"))
            self._show_status_message(trans.t("msg_render_cancelled"), 4000)
            return

        self.roi_aligned_images = aligned_images
        self.roi_aligned_raw_count = len(self._roi_align_snapshot)  # 对齐时快照的帧数
        self.roi_source_revision = context_now[0]
        self.roi_included_indices = tuple(indices)
        self.roi_frame_ids = self._roi_frame_ids
        self.roi_filenames = list(names)
        self.roi_mode_active = True
        
        # 恢复按钮状态
        self.btn_preview_roi.setEnabled(True)
        self.btn_preview_roi.setText(trans.t("btn_roi"))
        
        # 在右侧结果面板显示对齐后的第一张图像
        if aligned_images and len(aligned_images) > 0:
            self._display_roi_aligned_image(0)
            
            # 启用滑动条，让用户可以浏览对齐后的图像栈
            self.result_slider.setRange(0, len(aligned_images) - 1)
            self.result_slider.setValue(0)
            self.result_slider.setEnabled(True)
            self.result_control_bar.setVisible(True)
            
            # 启用右侧面板的ROI选择模式
            if hasattr(self, 'lbl_result_img'):
                self.lbl_result_img.roi_mode = True
                
            # 显示提示信息
            show_message_box(
                self, 
                trans.t("roi_ready_title"),
                trans.t("roi_ready_msg"),
                trans.t("roi_ready_detail").format(alignment_time),
                QMessageBox.Icon.Information
            )
        
        self.roi_alignment_worker = None

    def _on_roi_alignment_error(self, error_message):
        """ROI配准错误的回调"""
        self.btn_preview_roi.setEnabled(True)
        self.btn_preview_roi.setText(trans.t("btn_roi"))
        self.btn_preview_roi.blockSignals(True)
        self.btn_preview_roi.setChecked(False)
        self.btn_preview_roi.blockSignals(False)
        
        show_error_box(
            self,
            trans.t("msg_error"),
            trans.t("roi_align_failed"),
            error_message
        )
        
        self.roi_alignment_worker = None

    def _display_roi_aligned_image(self, index: int):
        """在右侧结果面板显示ROI模式下对齐后的图像"""
        if not self.roi_aligned_images or index < 0 or index >= len(self.roi_aligned_images):
            return
        
        # 保存当前ROI区域，以便在切换图像时保持ROI选择
        current_roi = None
        if hasattr(self, 'lbl_result_img') and self.lbl_result_img.get_roi_rect() is not None:
            current_roi = self.lbl_result_img.get_roi_rect()
        
        try:
            image = self.roi_aligned_images[index]
            self.lbl_result_img.set_display_pixmap(cv2_to_pixmap(image))
            self.lbl_result_info.setText(f"{index + 1} / {len(self.roi_aligned_images)}")
            # ROI模式下保持控制栏可见，以便用户切换图像
            self.result_control_bar.setVisible(True)
            
            # 恢复之前的ROI选择区域
            if current_roi is not None:
                self.lbl_result_img.set_roi_rect(current_roi)
        except Exception as exc:
            show_error_box(self, trans.t("msg_error"), trans.t("roi_display_failed"), str(exc))

    def _on_roi_mode_exit_requested(self):
        """处理ROI模式退出请求（点击X按钮或右键点击非ROI区域）"""
        if self.roi_mode_active:
            # 取消ROI按钮的选中状态，这会触发toggle_roi_mode(False)
            self.btn_preview_roi.setChecked(False)

    def apply_dark_theme(self):
        """Apply the persisted theme at startup (light on a fresh install)."""
        from utils.settings_store import DEFAULT_THEME
        self.apply_theme(getattr(self, "ui_theme", DEFAULT_THEME))

    def apply_theme(self, theme: str) -> None:
        """Apply the 'dark' or 'light' theme immediately and persist it.

        Both themes are generated from the same design tokens, so this is a
        single code path: clear any per-widget leftovers, install the
        generated sheet on the application, then let widgets that paint by
        hand (image panels, banner, help buttons) refresh their own palette.

        The old implementation stripped every inline stylesheet in light mode
        and relied on a hand-written light sheet, which is why light mode kept
        losing the panels' own styling.
        """
        from ui.styles import get_theme_style, set_current_theme
        theme = "light" if theme == "light" else "dark"
        self.ui_theme = theme
        ui_font = get_ui_font_family()
        mono_font = get_monospace_font_family()

        # Drop stylesheets captured by the previous (light) implementation.
        for wdg, _ss in getattr(self, "_inline_styles_saved", None) or []:
            try:
                wdg.setStyleSheet("")
            except RuntimeError:
                pass
        self._inline_styles_saved = None
        self.setStyleSheet("")

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet("")
            app.setPalette(app.style().standardPalette())
            app.setStyleSheet(_apply_font_family(get_theme_style(theme),
                                                 ui_font, mono_font))

        # Re-colour themed inline widgets that the light pass above strips:
        # the update banner must survive that sweep with a readable palette.
        banner = getattr(self, "update_banner", None)
        if banner is not None:
            banner.apply_theme(theme)
        # Image panels carry their own QSS (title bar / canvas / control bar);
        # it must be rebuilt for the new palette.
        from ui.image_panels import apply_panel_theme
        for panel in getattr(self, "_panel_widgets", []):
            try:
                apply_panel_theme(panel, theme)
            except RuntimeError:
                pass
        # The active theme must be recorded BEFORE any inline style is
        # recomputed and before dialogs are constructed: help_button_style()
        # with no argument reads CURRENT_THEME, so refreshing the "?" chips
        # first re-applied the OLD theme's colours (the dark chip survived the
        # switch to light). Dialogs opened later read the same global.
        set_current_theme(theme)
        # Inline "?" button styles are captured at creation; re-apply them for
        # the new theme or the dark chip stays dark on the light panel.
        from ui.styles import help_button_style
        for widget in self.findChildren(QPushButton):
            try:
                if widget.objectName() == "helpButton":
                    widget.setStyleSheet(help_button_style(theme))
            except RuntimeError:
                pass
        # Numeric steppers paint their -/+ buttons from inline sheets built at
        # creation, so without this they keep the old theme's colours (a black
        # stepper on the light panel).
        from widgets.stepper import StepperSpinBox
        for widget in self.findChildren(StepperSpinBox):
            try:
                widget.reapply_theme()
            except RuntimeError:
                pass

        self._enable_dark_title_bar(dark=(theme != "light"))
        if hasattr(self, "ui_objs") and "menu_theme" in self.ui_objs:
            for act in self.ui_objs["menu_theme"].actions():
                # Sync by data property, not display text: the text changes
                # with the language and mis-matched after a switch.
                act.setChecked(act.data() == theme)

    def _enable_dark_title_bar(self, dark: bool = True):
        """Windows: paint the native title bar dark/light to match the theme."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = int(self.winId())  # forces native handle creation
            value = ctypes.c_int(1 if dark else 0)
            # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE on Win10 20H1+; 19 on older builds
            for attribute in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attribute, ctypes.byref(value), 4) == 0:
                    break
        except Exception:
            pass

    def show_contact_info(self):
        """显示联系信息"""
        # 创建联系信息对话框
        dialog = ContactInfoDialog(self)
        dialog.exec()
    
    def show_batch_processing_dialog(self, preload_folder_paths: list[str] = None, scale_factor: float = 1.0):
        """显示批处理设置对话框

        Args:
            preload_folder_paths: 可选，预加载的文件夹路径列表（用于拖入场景）
            scale_factor: 缩放因子，用于预加载时缩放图像
        """
        from dialogs import BatchProcessingDialog

        dialog = BatchProcessingDialog(self)

        if preload_folder_paths:
            if len(preload_folder_paths) == 1:
                dialog.preload_single_folder(preload_folder_paths[0], scale_factor)
            else:
                dialog.preload_multiple_folders(preload_folder_paths, scale_factor)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            folder_paths = dialog.folder_paths
            output_type, output_path = dialog.get_output_settings()
            processing_settings = dialog.get_processing_settings()
            import_mode = dialog.get_import_mode()
            split_method, split_param = dialog.get_split_settings()

            if import_mode == "single_folder":
                self.batch_manager.start_batch_processing(
                    folder_paths=[],
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                    split_method=split_method,
                    split_param=split_param,
                    single_folder_images_with_times=dialog.single_folder_images_with_times,
                )
            else:
                self.batch_manager.start_batch_processing(
                    folder_paths=folder_paths,
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                )

    def show_tile_settings(self):
        """显示 Tile 设置对话框"""
        dialog = TileSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 设置已由对话框写回到 self 属性；可在此处触发必要的刷新
            self.update_slider_availability()
            # 如果需要，可以在右侧面板或其它地方反映设置变化
            return True
        return False

    def show_registration_settings(self):
        """显示配准设置对话框，允许用户修改 downscale_width"""
        from dialogs import RegistrationSettingsDialog
        dialog = RegistrationSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 值已写回到 self.reg_downscale_width
            return True
        return False

    def show_thread_settings(self):
        """显示线程数设置对话框"""
        from dialogs import ThreadSettingsDialog
        dialog = ThreadSettingsDialog(self)
        dialog.exec()

    def show_ai_batch_settings(self):
        """显示 StackMFF V3 批量大小设置对话框"""
        dialog = AIBatchSettingsDialog(self)
        dialog.exec()

    def rotate_stack(self, rotation_code):
        """Delegate stack rotation to the transform manager."""
        self.transform_manager.rotate_stack(rotation_code)

    def reload_image_stack(self):
        """Ensure legacy callers refresh through the transform manager."""
        self.transform_manager.reload_image_stack()

    def flip_stack(self, flip_code):
        """Mirror the stack via the transform manager."""
        self.transform_manager.flip_stack(flip_code)

    def resize_all_images(self):
        """Resize images through the transform manager."""
        self.transform_manager.resize_all_images()

    def _on_enter_source_preview(self):
        self._mouse_in_source_preview = True

    def _on_leave_source_preview(self):
        self._mouse_in_source_preview = False

    def _on_enter_result_preview(self):
        self._mouse_in_result_preview = True

    def _on_leave_result_preview(self):
        self._mouse_in_result_preview = False

    def _is_mouse_in_preview(self) -> bool:
        return self._mouse_in_source_preview or self._mouse_in_result_preview

    def _has_work_in_flight(self) -> bool:
        """True while a render, batch job, ROI alignment, GIF export or
        stack load runs."""
        running = []
        render_manager = getattr(self, 'render_manager', None)
        if render_manager is not None and getattr(render_manager, 'worker', None):
            running.append(render_manager.worker.isRunning())
        batch_manager = getattr(self, 'batch_manager', None)
        if batch_manager is not None and getattr(batch_manager, '_thread', None):
            running.append(batch_manager._thread.isRunning())
        roi_worker = getattr(self, 'roi_alignment_worker', None)
        if roi_worker is not None:
            running.append(roi_worker.isRunning())
        export_manager = getattr(self, 'export_manager', None)
        gif_worker = getattr(export_manager, 'gif_worker', None) if export_manager else None
        if gif_worker is not None:
            running.append(gif_worker.isRunning())
        source_manager = getattr(self, 'source_manager', None)
        load_worker = getattr(source_manager, '_load_worker', None) if source_manager else None
        if load_worker is not None:
            running.append(load_worker.isRunning())
        # 深度图对话框关闭后仍在跑的 worker（AI 可达数分钟）也要算数，
        # 否则退出时模块级集合随解释器清理，QThread 运行中被析构 -> abort
        try:
            from dialogs.depthmap import _ORPHAN_WORKERS
            running.extend(w.isRunning() for w in _ORPHAN_WORKERS)
        except Exception:
            pass
        return any(running)

    def closeEvent(self, event):
        """Clean up background threads before closing the window."""
        # Nothing new may start once we are on the way out (update downloads
        # check this flag before they begin).
        self._shutting_down = True
        # A retouched output lives in memory until exported; exiting must not
        # silently throw those pixels away.
        if (not getattr(self, '_restart_for_update', False)
                and hasattr(self, "output_manager")
                and not self.output_manager.confirm_unexported_on_exit()):
            self._shutting_down = False
            event.ignore()
            return
        # Shutting down discards in-flight results without a word; ask first
        # (the self-update restart has already stopped the workers itself).
        # Unsaved project state: every editor asks before discarding it.
        # Skip during the update restart (the session is restored anyway).
        if (not getattr(self, '_restart_for_update', False)
                and getattr(self, "_project_dirty", False)):
            if not self._confirm_discard_project_changes(
                    "msg_discard_unsaved_project_quit"):
                self._shutting_down = False
                event.ignore()
                return

        if not getattr(self, '_restart_for_update', False) and self._has_work_in_flight():
            box = QMessageBox(self)
            style_message_box(box)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(trans.t('msg_warning'))
            box.setText(trans.t('msg_quit_during_render'))
            quit_btn = box.addButton(trans.t('btn_quit_anyway'),
                                     QMessageBox.ButtonRole.AcceptRole)
            box.addButton(trans.t('btn_continue'), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(quit_btn)
            box.exec()
            if box.clickedButton() is not quit_btn:
                # The user chose to keep working: un-arm the shutdown so
                # background update checks keep working afterwards.
                self._shutting_down = False
                event.ignore()
                return

        # Persist user preferences, layout and recent files for next time
        try:
            self.persist_settings()
            save_window_layout(self)
        except Exception:
            pass

        # Crash auto-recovery: keep the last snapshot, clear the live lock
        from utils import recovery
        recovery.write_snapshot(self)
        recovery.mark_clean_exit()

        # Stop ROI alignment worker if running
        if getattr(self, 'roi_alignment_worker', None) is not None:
            _shutdown_worker(self.roi_alignment_worker)
            self.roi_alignment_worker = None

        # Stop stack-load worker if running (closing mid-decode otherwise
        # destroys a running QThread at teardown and aborts the process)
        if hasattr(self, 'source_manager'):
            _shutdown_worker(getattr(self.source_manager, '_load_worker', None))
            self.source_manager._load_worker = None

        # Stop update download/stage workers if running. The background
        # update download is not "work in flight" — it is cancelled here
        # instead of nagging the user with the quit confirmation.
        manager = getattr(self, 'update_manager', None)
        if manager is not None:
            manager.shutdown()
        _shutdown_worker(getattr(self, '_update_download_worker', None))
        try:
            from dialogs.depthmap import _ORPHAN_WORKERS
            for orphan in list(_ORPHAN_WORKERS):
                orphan.cancel()
                _shutdown_worker(orphan)
        except Exception:
            pass

        # Stop render worker if running
        if hasattr(self, 'render_manager'):
            _shutdown_worker(self.render_manager.worker)
            self.render_manager.worker = None

        # Stop GIF saver worker if running (no cancel support; wait briefly)
        if hasattr(self, 'export_manager'):
            _shutdown_worker(getattr(self.export_manager, 'gif_worker', None))

        # Stop batch worker if running
        if hasattr(self, 'batch_manager') and self.batch_manager._thread is not None:
            if self.batch_manager._worker:
                self.batch_manager._worker.cancel()
            if self.batch_manager._thread.isRunning():
                self.batch_manager._thread.quit()
                self.batch_manager._thread.wait(5000)
            self.batch_manager._teardown_worker()

        # Stop status update timer
        if hasattr(self, 'status_timer') and self.status_timer.isActive():
            self.status_timer.stop()

        # Remove temp JPGs written for drag-out export
        if hasattr(self, 'output_list') and hasattr(self.output_list, 'cleanup_temp_files'):
            self.output_list.cleanup_temp_files()

        super().closeEvent(event)

    # --- Language ---
    
    def set_language(self, lang_code: str) -> None:
        """Switch application language (update_ui_text runs via the signal)."""
        trans.set_language(lang_code)
        self.persist_settings()

    # --- Settings persistence & recent files ---

    def persist_settings(self) -> None:
        """Write user preferences (thread count, tile params, GPU, language...)."""
        save_window_settings(self)

    def set_use_gpu(self, enabled: bool) -> None:
        """Toggle GPU acceleration (Settings menu); applies to future renders."""
        if enabled:
            # Nothing to accelerate without a device; the menu item is disabled
            # in that case, but a saved project or a scripted call could still
            # ask for it.
            from utils.platform_utils import has_compute_gpu
            if not has_compute_gpu():
                enabled = False
        self.use_gpu = bool(enabled)
        self._update_dynamic_status()
        self.persist_settings()

    def set_restore_last_stack(self, enabled: bool) -> None:
        """Toggle auto-restoring the last opened stack on startup."""
        self.restore_last_stack = bool(enabled)
        self.persist_settings()

    # 平滑默认值：B 深度图 σ、A 加权混合 σ、C 金字塔混合 σ
    _SMOOTHING_DEFAULTS = {
        "openfocus_a": 2.0,
        "openfocus_b": 1.2,
        "openfocus_c": 3.0,
    }

    def smoothing_for_method(self, method: str) -> float:
        """Smoothing σ the user has set for *method* (or its default).

        A and C blend their frames and B smooths a depth map, so one shared
        value would silently apply A's sigma to C or B's to A after a method
        switch.
        """
        memory = self.__dict__.setdefault("_smoothing_by_method", {})
        if method in memory:
            return float(memory[method])
        return float(self._SMOOTHING_DEFAULTS.get(method, 1.2))

    def set_ofb_depth_smoothing(self, value: float) -> None:
        """Smoothing spin box: per-method sigma (see smoothing_for_method)."""
        method = self.current_fusion_method()
        value = float(value)
        if method in ("openfocus_a", "openfocus_c"):
            self.of_blend_smoothing = value
        elif method == "openfocus_b":
            self.ofb_depth_smoothing = value
        else:
            return
        self.__dict__.setdefault("_smoothing_by_method", {})[method] = value
        self.persist_settings()

    def set_ask_load_scale(self, enabled: bool) -> None:
        """Toggle the loading-scale prompt shown before every import."""
        self.ask_load_scale = bool(enabled)
        self.persist_settings()

    def set_align_cache_enabled(self, enabled: bool) -> None:
        """Toggle the on-disk registration cache (Settings menu)."""
        self.align_cache_enabled = bool(enabled)
        self.persist_settings()

    def add_recent_file(self, path: str) -> None:
        """Record a successfully opened stack at the top of the recent list."""
        from utils.settings_store import add_recent_file as _store_add
        _store_add(self, path)

    def add_recent_project(self, path: str) -> None:
        """Record a successfully opened project at the top of the recent list."""
        from utils.settings_store import add_recent_project as _store_add
        _store_add(self, path)

    def rebuild_recent_projects_menu(self) -> None:
        """Fill the File > Open Recent Project submenu."""
        from PyQt6.QtGui import QAction
        menu = self.ui_objs.get('menu_recent_projects') if hasattr(self, 'ui_objs') else None
        if menu is None:
            return

        menu.clear()
        recent = list(getattr(self, 'recent_projects', []))
        if not recent:
            empty = QAction(trans.t('menu_recent_projects_empty'), menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        for path in recent:
            action = QAction(path, menu)
            action.triggered.connect(
                lambda checked=False, p=path: self._open_recent_project(p))
            menu.addAction(action)

        menu.addSeparator()
        clear_action = QAction(trans.t('menu_recent_projects_clear'), menu)
        clear_action.triggered.connect(self._clear_recent_projects)
        menu.addAction(clear_action)

    def _open_recent_project(self, path: str) -> None:
        from utils.project_file import validate_project_for_open, apply_project
        if not os.path.exists(path):
            show_warning_box(self, trans.t('msg_recent_missing_title'),
                             trans.t('msg_recent_project_missing'))
            if path in getattr(self, 'recent_projects', []):
                self.recent_projects.remove(path)
                self.persist_settings()
                self.rebuild_recent_projects_menu()
            return
        if not self._confirm_discard_project_changes(
                "msg_discard_unsaved_project"):
            return
        ok, err, state = validate_project_for_open(self, path)
        if not ok:
            if err:
                show_warning_box(self, trans.t('msg_error'), err)
            return
        def loaded():
            self.project_path = path
            self._clear_project_dirty()
            if state.get("sources_relocated"):
                self._mark_project_dirty()
            self.add_recent_project(path)
        apply_project(self, state, on_loaded=loaded)

    def _clear_recent_projects(self) -> None:
        self.recent_projects = []
        self.persist_settings()
        self.rebuild_recent_projects_menu()

    def rebuild_recent_menu(self) -> None:
        """Fill the File > Open Recent submenu from self.recent_files."""
        from PyQt6.QtGui import QAction
        menu = self.ui_objs.get('menu_recent') if hasattr(self, 'ui_objs') else None
        if menu is None:
            return

        menu.clear()
        recent = list(getattr(self, 'recent_files', []))
        if not recent:
            empty = QAction(trans.t('menu_recent_empty'), menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        for i, path in enumerate(recent):
            label = f"&{i + 1}  {path}" if i < 9 else path
            action = QAction(label, menu)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent_path(p))
            menu.addAction(action)

        menu.addSeparator()
        clear_action = QAction(trans.t('menu_recent_clear'), menu)
        clear_action.triggered.connect(self._clear_recent_files)
        menu.addAction(clear_action)

    def _open_recent_path(self, path: str) -> None:
        import os as _os
        if not _os.path.exists(path):
            show_warning_box(self, trans.t('msg_recent_missing_title'), trans.t('msg_recent_missing_text'))
            if path in self.recent_files:
                self.recent_files.remove(path)
                self.persist_settings()
                self.rebuild_recent_menu()
            return
        if _os.path.isdir(path):
            self.source_manager.load_image_stack(path)
        else:
            ext = _os.path.splitext(path)[1].lower()
            if ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS:
                self.source_manager.load_video_stack(path)
            else:
                self.source_manager.load_image_stack(_os.path.dirname(path))

    def _clear_recent_files(self) -> None:
        self.recent_files = []
        self.persist_settings()
        self.rebuild_recent_menu()

    def update_ui_text(self) -> None:
        """Update all UI strings based on current language."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        # Update Menus
        if hasattr(self, 'ui_objs'):
             for key, obj in self.ui_objs.items():
                # Language entries keep their endonym. The generic loop used
                # the ui_objs key ('action_lang_zh') as a translation key, so
                # in a Japanese UI the Chinese entry rendered as
                # "中文（简体）" — the same language under a different name
                # depending on the current UI language.
                if key.startswith('action_lang_') and isinstance(obj, QAction):
                    continue
                t_key = obj.property("trans_key")
                if not t_key:
                    t_key = key
                
                text = trans.t(t_key)
                if isinstance(obj, QMenu):
                    obj.setTitle(text)
                elif isinstance(obj, QAction):
                    obj.setText(text)
        
        # Update Right Panel
        c = self.right_panel_components
        c.method_group.setTitle(trans.t('group_fusion'))
        # 下拉框重翻译：按 (代码, i18n 键, 提示键) 重建条目文本
        from ui.right_panel import FUSION_ITEMS, ALIGN_ITEMS
        for i, (_code, key, hint_key) in enumerate(FUSION_ITEMS):
            if i < c.combo_fusion.count():
                c.combo_fusion.setItemText(i, trans.t(key))
                if hint_key:
                    c.combo_fusion.setItemData(
                        i, trans.t(hint_key), Qt.ItemDataRole.ToolTipRole)
        # set_panel_label elides to the panel width and keeps the full string
        # in the tooltip; plain setText here re-broke the row on language
        # switch after the panel had already fitted it.
        from ui.right_panel import set_panel_label
        if getattr(c, 'chk_base_correct', None) is not None:
            set_panel_label(c.chk_base_correct, trans.t('chk_base_correct'),
                            trans.t('chk_base_correct_hint'))
        if getattr(c, 'chk_exposure_match', None) is not None:
            set_panel_label(c.chk_exposure_match,
                            trans.t('chk_exposure_match'),
                            trans.t('chk_exposure_match_hint'))
        if getattr(c, 'chk_reverse_order', None) is not None:
            set_panel_label(c.chk_reverse_order,
                            trans.t('chk_reverse_order'),
                            trans.t('chk_reverse_order_hint'))

        c.registration_group.setTitle(trans.t('group_registration'))
        for i, (_code, key, hint_key) in enumerate(ALIGN_ITEMS):
            if i < c.combo_align.count():
                c.combo_align.setItemText(i, trans.t(key))
                if hint_key:
                    c.combo_align.setItemData(
                        i, trans.t(hint_key), Qt.ItemDataRole.ToolTipRole)
        
        # 滑块含义标签由 update_slider_availability 按当前算法写入
        self.update_slider_availability()
        c.btn_reset.setText(trans.t('btn_reset'))
        c.btn_select_all.setText(trans.t('btn_select_all'))
        c.btn_select_all.setToolTip(trans.t('btn_select_all_hint'))
        c.btn_select_none.setText(trans.t('btn_select_none'))
        c.btn_select_none.setToolTip(trans.t('btn_select_none_hint'))
        # Keep the cancel/progress affordance while a render runs — the
        # retranslation must not reset it to the idle "Render" label.
        busy = (getattr(self.render_manager, 'worker', None) is not None
                and self.render_manager.worker.isRunning())             or getattr(self.render_manager, '_compare_mode', False)
        if not busy:
            c.btn_render.setText(trans.t('btn_render'))
        c.btn_compare.setText(trans.t('btn_compare_all'))
        # ElidingButton elides for the current width itself; setting the hint
        # after setText keeps the shortcut description rather than the label.
        c.btn_compare.setToolTip(
            f"{trans.t('btn_compare_all')}\n{trans.t('btn_compare_all_hint')}")
        # Panel titles are QLabels built once at startup; without this they
        # kept the language the app was launched in ("Source Stack"/"Output"
        # stayed English after switching to Polish).
        if hasattr(self, 'lbl_source_title'):
            self.lbl_source_title.setText(' ' + trans.t('panel_source_stack'))
        if hasattr(self, 'lbl_result_title'):
            self.lbl_result_title.setText(trans.t('panel_output'))
        c.chk_quick_preview.setText(trans.t('chk_quick_preview'))
        c.chk_quick_preview.setToolTip(trans.t('chk_quick_preview_hint'))
        # The AI output-mode block was never retranslated: switching to any
        # language left "AI 输出方式 / 软融合 / 硬融合" in Chinese, because
        # these widgets only got their text once, at construction.
        c.lbl_ai_mode.setText(trans.t('label_ai_mode'))
        c.lbl_ai_mode.setToolTip(trans.t('label_ai_mode_hint'))
        set_panel_label(c.radio_ai_soft, trans.t('radio_ai_soft'),
                        trans.t('radio_ai_soft_hint'))
        set_panel_label(c.radio_ai_hard, trans.t('radio_ai_hard'),
                        trans.t('radio_ai_hard_hint'))
        
        # ROI Button
        if hasattr(self, 'btn_preview_roi'):
            self.btn_preview_roi.setText(trans.t('btn_roi'))

        # Lists labels
        count = self.file_list.count()
        try:
            c.source_images_label.setText(
                self.source_manager._source_count_text(count))
        except Exception:
            c.source_images_label.setText(
                trans.t('label_source_images').format(count))
        
        out_count = self.output_list.count()
        c.output_label.setText(trans.t('label_output').format(out_count))
        
        # Status
        self.update_loaded_status()
        self._update_dynamic_status()
        
        # Drag hint
        if not self.stack_images:
            if hasattr(self, 'lbl_source_img'):
                self.lbl_source_img.setText(trans.t('drag_hint'))
        if getattr(self, 'fusion_result', None) is None and not getattr(
                self, 'registration_results', None) and hasattr(self, 'lbl_result_img'):
            self.lbl_result_img.setText(trans.t('result_hint'))

        # Update language checked state. Derived from the registered actions
        # instead of a hardcoded list: a newly added pack kept its stale tick
        # because this loop only knew about four codes.
        for key, action in list(getattr(self, 'ui_objs', {}).items()):
            if key.startswith('action_lang_') and hasattr(action, 'setChecked'):
                action.setChecked(trans.current_lang == key[len('action_lang_'):])

        self.sharpness_curve.set_tooltip_base(trans.t('sharpness_curve_tooltip'))

        # Wipe controls are created outside ui_objs; retranslate them here
        if hasattr(self, 'btn_wipe'):
            self.btn_wipe.setText(trans.t('btn_wipe'))
            self.btn_wipe.setToolTip(trans.t('wipe_hint'))
            self.chk_wipe_follow_left.setText(trans.t('wipe_follow_source'))
            self.chk_wipe_follow_right.setText(trans.t('wipe_follow_latest'))
        if 'action_gpu_accel' in getattr(self, 'ui_objs', {}):
            self.ui_objs['action_gpu_accel'].setToolTip(trans.t('msg_gpu_accel_hint'))
        # The tuning knobs live in the Advanced settings page now, so there are
        # no menu actions to retranslate here.
        if getattr(self, 'lbl_depth_smooth', None) is not None:
            self.lbl_depth_smooth.setText(trans.t('label_depth_smoothing'))
            self.lbl_depth_smooth.setToolTip(trans.t('label_depth_smoothing_hint'))
            self.spin_depth_smooth.setToolTip(trans.t('label_depth_smoothing_hint'))
        # The install action carries a {version} placeholder, so the trans_key
        # sweep above cannot render it; the indicator does (and hides it when
        # no build is staged).
        self._refresh_update_indicator()
        banner = getattr(self, "update_banner", None)
        if banner is not None:
            banner.retranslate()
        self.rebuild_recent_menu()
        self.rebuild_recent_projects_menu()


if __name__ == "__main__":
    # Windowed (no-console) builds have no usable stdout/stderr - any
    # print() would raise "I/O operation on closed file" and kill renders.
    # Route them to a no-op writer; real diagnostics go to the file log.
    class _NullWriter:
        def write(self, s):
            return len(s)
        def flush(self):
            pass
        def isatty(self):
            return False

    if getattr(sys, "frozen", False) and (
            sys.stdout is None or getattr(sys.stdout, "closed", False)
            or sys.stderr is None or getattr(sys.stderr, "closed", False)):
        sys.stdout = _NullWriter()
        sys.stderr = _NullWriter()

    # File logging first so every later step leaves a trace in the log
    from utils.logging_setup import setup_logging, get_logger
    setup_logging()
    from utils.logging_setup import log_session_start, log_action
    log_session_start()
    log_action("startup", args=str(sys.argv[1:]) if len(sys.argv) > 1 else "-")
    _log = get_logger()

    # Diagnostics: capture every unhandled exception (slots abort silently
    # on PyQt6 otherwise) into a crash file next to the normal log.
    import faulthandler as _fh, tempfile as _tf
    _crash_path = os.path.join(_tf.gettempdir(), "openfocus_crash.txt")
    _crash_f = open(_crash_path, "a", encoding="utf-8")
    _fh.enable(_crash_f)

    def _dump_exc(kind, exc_info):
        try:
            _crash_f.write(kind + " " + repr(exc_info) + "\n")
            _crash_f.flush()
        except Exception:
            pass

    sys.excepthook = lambda t, v, tb: _dump_exc("EXCEPTHOOK", (t, v, tb))
    import threading as _th
    _th.excepthook = lambda a: _dump_exc("THREAD EXC", a)
    _log.info("crash diagnostics active: %s", _crash_path)

    # Headless CLI mode (--input/--output/--method ...); unknown options exit
    # with argparse usage. Plain positional paths still auto-load in the GUI.
    # --restore-session is ours (the updater passes it to the relaunched build)
    # and must be consumed before the CLI sees a leading dash.
    _argv = [a for a in sys.argv[1:] if a != "--restore-session"]
    _restore_flag = len(_argv) != len(sys.argv[1:])
    if any(arg.startswith("-") for arg in _argv):
        from core.cli import run_cli
        sys.exit(run_cli(_argv))

    app = OpenFocusApplication(sys.argv)

    # ---- Single-instance guard ----
    # A second launch forwards its file arguments to the running instance
    # (through a local socket) and exits, instead of opening a competing
    # instance whose settings writes would clobber the first one's.
    from PyQt6.QtNetwork import QLocalServer, QLocalSocket
    import json as _json
    _server_name = "OpenFocus-single-instance"

    probe = QLocalSocket()
    probe.connectToServer(_server_name)
    if probe.waitForConnected(500):
        paths = [os.path.abspath(a) for a in sys.argv[1:] if not a.startswith("-")]
        probe.write(_json.dumps(paths).encode("utf-8"))
        probe.flush()
        probe.waitForBytesWritten(1000)
        probe.disconnectFromServer()
        _log.info("forwarded %d path(s) to the running instance", len(paths))
        sys.exit(0)
    QLocalServer.removeServer(_server_name)
    _single_server = QLocalServer()
    _single_server.listen(_server_name)

    window = OpenFocus()
    app.set_main_window(window)
    _log.info("main window ready")

    def _on_second_instance(paths_json: str) -> None:
        try:
            paths = _json.loads(paths_json)
        except ValueError:
            paths = []
        if paths:
            process_command_line_args(window, paths)
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_new_connection() -> None:
        conn = _single_server.nextPendingConnection()
        if conn is None:
            return
        conn.readyRead.connect(
            lambda: _on_second_instance(bytes(conn.readAll()).decode("utf-8", "ignore")))

    _single_server.newConnection.connect(_on_new_connection)

    # ---- Update restart: restore the session the updater snapshotted ----
    # The swap script relaunches with --restore-session, so this runs before
    # (and instead of) the crash prompt: the previous exit was clean.
    from PyQt6.QtCore import QTimer
    from utils import recovery
    restore_hint = recovery.pending_restore()
    if restore_hint is None and _restore_flag:
        restore_hint = ""  # explicit request without a marker: use the snapshot
    recovered = False
    if restore_hint is not None:
        from utils.project_file import apply_project
        # Restore only the pre-restart snapshot. A saved project may predate
        # unsaved changes and must never stand in for a failed snapshot.
        state = recovery.load_snapshot_state()
        if state is not None:
            # Preserve the document association without changing snapshot data.
            window.project_path = restore_hint or None

            def _restored_ok():
                recovery.clear_pending_restore()
                window._show_status_message(
                    trans.t('update_restored_session'), 8000)

            def _restored_failed(reason: str):
                _log.warning("update restore failed: %s", reason)
                show_warning_box(window, trans.t('update_check_failed_title'),
                                 trans.t('update_restore_failed'))

            try:
                apply_project(window, state, on_loaded=_restored_ok,
                              on_error=_restored_failed)
                recovered = True
                _log.info("restoring the session after an update (%s)",
                          window.project_path)
            except Exception as exc:
                _restored_failed(str(exc))
        else:
            # Nothing to restore: do not consume the user's launch arguments
            # and do not claim a restore that did not happen.
            _log.warning("update restore requested but no valid snapshot exists")
            if not restore_hint:
                recovery.clear_pending_restore()

    # Report how the previous in-place update ended (a failed swap used to
    # look exactly like the user simply closing the app).
    from utils import self_update as _self_update
    for tag_res, ok_res, detail_res in _self_update.consume_results():
        if ok_res:
            _log.info("in-place update to %s succeeded", tag_res)
            QTimer.singleShot(900, lambda t=tag_res: window._show_status_message(
                trans.t('update_applied').format(version=t), 10000))
        else:
            _log.warning("in-place update to %s failed: %s", tag_res, detail_res)
            show_warning_box(window, trans.t('update_check_failed_title'),
                             trans.t('update_failed_after_restart'))

    # Crash auto-recovery: a leftover lock means the last session died
    # without a clean close. Offer the saved snapshot before anything else.
    pending = None if recovered else recovery.pending_recovery()
    if pending is not None:
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(window)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t('recovery_title'))
        box.setText(trans.t('recovery_text').format(
            count=recovery.snapshot_frame_count(pending)))
        restore_btn = box.addButton(trans.t('btn_restore'), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(trans.t('btn_start_fresh'), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restore_btn)
        box.exec()
        recovered = False
        if box.clickedButton() is restore_btn:
            from utils.project_file import apply_project
            window.project_path = recovery.session_path()
            apply_project(window, pending)
            recovered = True
            _log.info("recovered previous session from %s", recovery.session_path())
    recovery.init_lock()

    # Handle files passed via command-line (drag-to-EXE, desktop file, etc.)
    # 恢复会话与命令行参数互斥：接受恢复后再加载命令行文件夹会把同一栈
    # 追加成两倍帧数；恢复的工作区就是用户要的现场
    if len(_argv) > 0 and not recovered:
        process_command_line_args(window, _argv)

    window.show()
    _log.info("entering event loop")

    # Background updates: reuse an already staged build, then check quietly
    # (once a day, rate-limited inside the manager) and download any newer
    # release in the background so "update now" is instant.
    QTimer.singleShot(4000, window.startup_update_check)

    # First-run quick-start guide (once ever)
    from utils.settings_store import get_settings
    if str(get_settings().value("ui/first_run_done", "") or "") != "1":
        def _show_welcome():
            from dialogs.welcome import WelcomeDialog
            dlg = WelcomeDialog(window, main_window=window)
            dlg.exec()
            get_settings().setValue("ui/first_run_done", "1")
        QTimer.singleShot(1500, _show_welcome)

    sys.exit(app.exec())
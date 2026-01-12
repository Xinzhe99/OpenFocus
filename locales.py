
import datetime
from PyQt6.QtCore import QObject, pyqtSignal

class TranslationManager(QObject):
    languageChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_lang = 'en'
        
        # Auto-detect language based on timezone (UTC+8 -> China -> zh)
        try:
            offset = datetime.datetime.now().astimezone().utcoffset()
            if offset is not None and int(offset.total_seconds()) == 28800:
                self.current_lang = 'zh'
        except Exception:
            pass

        self.translations = {
            'en': {
                # Menu File
                'menu_file': 'File',
                'action_open_folder': 'Open Folder',
                'action_open_video': 'Open Video',
                'action_save': 'Save',
                'menu_save_stack': 'Save Stack',
                'menu_registered_stack': 'Registered Stack',
                'menu_input_stack': 'Input Stack',
                'action_save_folder': 'Save as Folder',
                'action_save_gif': 'Save as GIF',
                'action_clear_stack': 'Clear Stack',
                'action_exit': 'Exit',
                
                # Menu Edit
                'menu_edit': 'Edit',
                'menu_rotate': 'Rotate',
                'action_rotate_90_cw': '90° Clockwise',
                'action_rotate_90_ccw': '90° Counter-Clockwise',
                'action_rotate_180': '180°',
                'menu_flip': 'Flip',
                'action_flip_h': 'Horizontal Flip',
                'action_flip_v': 'Vertical Flip',
                'action_resize': 'Resize',
                'action_add_label': 'Add Label',
                'menu_del_label': 'Delete Label',
                'action_del_reg_label': 'Delete Registered Stack Labels',
                'action_del_input_label': 'Delete Input Stack Labels',
                
                # Menu Tools
                'menu_tools': 'Tools',
                'action_batch_process': 'Batch Processing',
                
                # Menu Settings
                'menu_settings': 'Settings',
                'menu_language': 'Language',
                'action_lang_en': 'English',
                'action_lang_zh': 'Chinese (Simplified)',
                'action_thread_settings': 'Thread Settings',
                'action_reg_settings': 'Registration Settings',
                'action_tile_settings': 'Tile Settings',
                'action_stackmffv4_batch_settings': 'StackMFF V4 Batch Size',
                
                # Menu Help
                'menu_help': 'Help',
                'action_env_info': 'Environment Info',
                'action_contact': 'Contact Us',
                
                # Right Panel
                'group_fusion': 'Fusion',
                'radio_guided_filter': 'Guided Filter',
                'radio_dct': 'DCT',
                'radio_dtcwt': 'DTCWT',
                'radio_gfg': 'GFG-FGF',
                'radio_stackmff': 'StackMFF-V4',
                'group_registration': 'Registration',
                'check_align_ecc': 'ECC',
                'check_align_homography': 'Homography',
                'label_kernel': 'Kernel:',
                'btn_reset': 'Reset Default',
                'btn_render': '▶ Start Render',
                'label_source_images': 'Source Images: {}',
                'label_output': 'Output: {}',
                'btn_roi': 'ROI',
                'btn_render_processing': '⏳ Processing...',
                
                # Status Panel
                'status_loaded': 'Loaded: {}',
                'status_gpu': 'GPU: {}',
                'status_res': 'Res: {}',
                'status_ram': 'RAM: {}',
                'status_loaded_fmt': 'Loaded: {} ({:.1f} MB/img)',
                'status_res_fmt': 'Res: {}x{}',
                'status_ram_fmt': 'RAM: {:.1f} GB / {:.0f} GB',
                
                # Main Window
                'drag_hint': 'Drag images here to add source files\nor use the image menu',
                'drag_roi_hint': 'Drag to select ROI',
                
                # Dialogs / Messages
                'msg_load_failed': 'Load Failed',
                'msg_load_error': 'Load Error',

                # ROI Dialog
                'dialog_roi_title': 'ROI Processing Options',
                'dialog_roi_msg': 'You have selected a Region of Interest (ROI).\nHow would you like to process the fusion?',
                'dialog_roi_group_output': 'Output Mode',
                'dialog_roi_opt_crop': 'Fuse ROI Only (Output Cropped Image)',
                'dialog_roi_opt_crop_tooltip': 'Result will be a small image containing only the selected region.',
                'dialog_roi_opt_paste': 'Fuse and Overlay on Base Frame',
                'dialog_roi_opt_paste_tooltip': 'Result will be the full base image with the fused ROI pasted on top.',
                'dialog_roi_group_base': 'Base Frame Selection',
                'dialog_roi_lbl_base': 'Select Base Frame:',
                
                # ROI Alignment (New)
                'roi_need_images': 'Please load at least 2 images before using ROI mode.',
                'roi_aligning': 'Aligning...',
                'roi_ready_title': 'ROI Mode Ready',
                'roi_ready_msg': 'Images have been aligned using ECC method.',
                'roi_ready_detail': 'Alignment completed in {:.2f}s.\n\nYou can now draw a ROI on the right panel.\nPress Render to fuse only the selected region.',
                'roi_align_failed': 'ROI alignment failed.',
                'roi_aligned': 'Aligned',
                'roi_display_failed': 'Failed to display aligned image.',
                'result_hint': 'Result will appear here',
                
                # Completion Dialog
                'dialog_completed_title': 'Processing Completed',
                'dialog_completed_msg': 'Processing completed successfully!',
                
                'info_align_method': 'Alignment Method: {}',
                'info_align_time': 'Alignment Time: {:.2f}s',
                'info_fusion_method': 'Fusion Method: {}',
                'info_fusion_time': 'Fusion Time: {:.2f}s',
                'info_proc_unit': 'Processing Unit: {}',
                'info_total_time': 'Total Time: {:.2f}s',
                'info_align_none_cached': 'Alignment Method: None (using cached results)',
                'info_align_none': 'Alignment Method: None',
                'info_fusion_none': 'Fusion Method: None',
                'val_enabled': 'Enabled',
                'val_none': 'None',
                'val_using_cached': 'using cached results',
                'btn_ok': 'OK',
                'btn_cancel': 'Cancel',
                'msg_success': 'Success',
                'msg_error': 'Error',
                'msg_proc_error': 'An error occurred during processing:',
                'msg_warning': 'Warning',
                
                # Help Dialogs
                'dialog_env_title': 'Environment Information',
                'env_subtitle': 'OpenFocus Environment Dependencies',
                'env_python': 'Python Version',
                'env_installed': 'Installed: Version {}',
                'env_not_installed': 'Not installed',
                'env_cuda_avail': 'CUDA available: {}',
                'env_cuda_ver': 'CUDA version: {}',
                'env_mps_avail': 'MPS available (Apple Silicon)',
                'env_gpu_accel': 'StackMFF-V4: GPU acceleration available',
                'env_no_gpu': 'Warning: No GPU acceleration (CUDA/MPS)',
                'env_cpu_mode': 'StackMFF-V4: Available (CPU mode - slower)',
                'env_stackmff_unavailable': 'StackMFF-V4 fusion not available',
                'env_dtcwt_unavailable': 'Not installed (DTCWT fusion unavailable)',
                'env_summary': 'Summary',
                'env_core_dep': 'Core Dependencies:',
                'env_core_desc': '- OpenCV, NumPy, PyQt6: Required for basic functionality',
                'env_gpu_opt': 'GPU Acceleration (Optional):',
                'env_gpu_desc': '- PyTorch: Enables StackMFF-V4 (CPU fallback available but slower)',
                'env_fusion_alg': 'Fusion Algorithms:',
                'env_fusion_desc': '- DTCWT library: Required for DTCWT fusion',
                
                'dialog_contact_title': 'Contact Us',
                'contact_info_title': 'Contact Information',
                'contact_email': 'Email',
                'contact_institution': 'Institution',
                'contact_zju': 'Zhejiang University',
                'contact_github': 'GitHub',
                'contact_welcome': 'We warmly welcome contributors who would like to add new fusion methods and help OpenFocus grow.',
                
                'btn_close': 'Close',

                # Batch Processing Dialog
                'batch_title': 'Batch Processing',
                'batch_import_mode': 'Import Mode',
                'batch_mode_multi': 'Multiple Folders (one stack per folder)',
                'batch_mode_single': 'Single Folder (auto-split into multiple stacks)',
                'batch_stack_folders': 'Image Stack Folders',
                'batch_path_placeholder': 'Type a path and press Enter to refresh',
                'batch_btn_add': 'Add Folders',
                'batch_btn_remove': 'Remove Selected',
                'batch_single_split_settings': 'Single Folder Split Settings',
                'batch_folder_none': 'Folder: (none)',
                'batch_split_method': 'Split Method:',
                'batch_split_fixed': 'Fixed Count',
                'batch_split_time': 'Time Threshold',
                'batch_images_per_stack': 'Images per Stack:',
                'batch_time_threshold': 'Time Threshold:',
                'batch_unit_images': 'images',
                'batch_unit_seconds': 'seconds',
                'batch_preview_default': 'Preview: 0 images → 0 stacks',
                'batch_preview_fmt_count': 'Preview: {} images → {} stacks ({} images each)',
                'batch_preview_fmt_time': 'Preview: {} images → {} stacks (threshold: {}s)',
                'batch_preview_no_folder': 'Preview: No folder selected',
                'batch_btn_select_split': 'Select Folder and Split',
                'batch_output_format': 'Output Format',
                'batch_format_label': 'Format:',
                'batch_quality_label': 'Quality:',
                'batch_output_location': 'Output Location',
                'batch_out_subfolder': 'Create subfolder in source folder',
                'batch_subfolder_name': 'Subfolder Name:',
                'batch_out_same': 'Same as source folder',
                'batch_out_custom': 'Specify output folder',
                'batch_btn_browse': 'Browse...',
                'batch_save_aligned': 'Save Aligned Image Stack',
                'batch_proc_options': 'Processing Options',
                'batch_lbl_fusion': 'Fusion Method: {}',
                'batch_lbl_reg': 'Registration Methods: {}',
                'batch_lbl_kernel': 'Kernel Size: {}',
                'batch_btn_start': 'Start Batch Processing',
                'batch_btn_cancel': 'Cancel',
                
                # Help Dialogs
                'help_render_title': 'Fusion Help',
                'help_registration_title': 'Registration Help',
                'help_tile_title': 'Tile Settings Help',

                'dialog_tile_title': 'Tile Settings',
                'dialog_tile_group': 'Tile Options',
                'dialog_tile_enabled_label': 'Enable Tiling:',
                'dialog_tile_enabled': 'Enabled',
                'dialog_tile_disabled': 'Disabled',
                'dialog_tile_block_size': 'Tile Block Size:',
                'dialog_tile_overlap': 'Tile Overlap:',
                'dialog_tile_threshold': 'Tile Threshold:',
                'dialog_tile_help_title': 'Tile Settings Help',
                'dialog_tile_help_text': '''<h3>Tile Settings Help</h3>
        <p>tile_enabled: Enable or disable tiled processing. When enabled, large images
        will be processed in smaller blocks to reduce memory usage.</p>

        <p>tile_block_size: Size (in pixels) of each square tile block. Typical values
        are 512–2048 depending on memory and speed tradeoffs.</p>

        <p>tile_overlap: Overlap (in pixels) between adjacent tiles used to avoid seams
        when combining results. A positive overlap helps smooth boundaries.</p>

        <p>tile_threshold: If the image's longest side is larger than this threshold,
        tiled processing will be considered. Smaller images are processed as a whole.</p>''',
                'dialog_reg_title': 'Registration Settings',
                'dialog_reg_group': 'Registration Options',
                'dialog_reg_downscale': 'Downscale Width:',
                'dialog_reg_help_title': 'Registration Settings Help',
                'dialog_reg_help_text': '''<h3>Downscale Width</h3>
        <p>downscale_width controls the pre-processing (downsampling) width when extracting
        registration features. Smaller values can speed up feature detection and reduce
        memory usage, but may sacrifice some geometric accuracy.</p>

        <p>Recommended values: Use <code>1024</code> for large images (>=2048px),
        <code>1600</code> for medium images. Only set higher values when maximum
        registration accuracy is needed and you have sufficient CPU/GPU resources.</p>

        <p>Lowering this value speeds up registration and reduces memory usage;
        increasing it can improve accuracy for very detailed images but increases runtime.</p>''',
                'dialog_thread_title': 'Thread Settings',
                'dialog_thread_group': 'Thread Count',
                'dialog_thread_label': 'Thread Count:',
                'dialog_thread_help_title': 'Thread Settings Help',
                'dialog_thread_help_text': '''<h3>Thread Count Settings</h3>
        <p>This setting controls the number of processing threads used by algorithms
        that support multi-threading. Typically set to match CPU core count (2-16).</p>

        <h4>Current multi-threading support:</h4>
        <ul>
        <li>GFG-FGF: Supports user-controlled thread count (default max: 8)</li>
        <li>Guided Filter Fusion (GFF): Supports user-controlled thread count (default max: 4)</li>
        <li>Image Registration: Feature extraction/transformation uses specified thread count</li>
        <li>DCT, DTCWT, StackMFF-V4: Currently do not use this setting (no thread control)</li>
        </ul>

        <p>Algorithms that don't use this value will safely ignore it. For best performance,
        avoid setting thread count higher than physical core count.</p>

        <p>Note: Installing <code>opencv-contrib-python</code> can accelerate certain
        operations (e.g., guided filtering).</p>''',

                # StackMFF V4 Batch settings dialog
                'dialog_stackmffv4_batch_title': 'StackMFF V4 Batch Settings',
                'dialog_stackmffv4_batch_group': 'Batch Processing',
                'dialog_stackmffv4_batch_label': 'Batch Size:',
                'dialog_stackmffv4_batch_help_title': 'StackMFF V4 Batch Settings Help',
                'dialog_stackmffv4_batch_help_text': '''<h3>StackMFF V4 Batch Settings</h3>
        <p>This setting controls the batch size when processing tiles with the StackMFF V4
        neural network fusion algorithm. Batch processing allows multiple tiles to be
        processed in parallel on the GPU, improving efficiency.</p>

        <h4>Batch Size:</h4>
        <ul>
        <li><b>Default: 2</b> - Balanced between memory usage and speed</li>
        <li><b>Higher values (4-8)</b> - Faster processing but requires more GPU memory</li>
        <li><b>Lower values (1)</b> - Lower memory usage, useful for limited GPU memory</li>
        </ul>

        <p><b>Note:</b> If you encounter out-of-memory errors on GPU, try reducing the batch size.
        This setting only affects StackMFF V4 when processing large images with tiling enabled.</p>

        <p>CPU mode is not significantly affected by this setting but will use it for
        consistency.</p>''',

                'dialog_duration_title': 'GIF Duration Settings',
                'dialog_duration_group': 'Frame Duration',
                'dialog_duration_label': 'Duration (ms):',

                # Add Label Dialog
                'add_label_title': 'Add Label Configuration',
                'label_target_stack': 'Target Stack:',
                'label_target_input': 'Input Image Stack',
                'label_target_registered': 'Registered Image Stack',
                'label_format': 'Format String:',
                'label_start_val': 'Starting Value:',
                'label_interval': 'Interval:',
                'label_x': 'X Location:',
                'label_y': 'Y Location:',
                'label_font_size': 'Font Size:',
                'label_font_family': 'Font Family:',
                'label_custom_text': 'Custom Text:',
                'label_range': 'Range:',
                'label_transparent_bg': 'Transparent Background',
                'label_bg_color': 'Background Color:',
                'label_choose_bg': 'Choose Background Color',
                'label_font_color': 'Font Color:',
                'label_choose_font_color': 'Choose Font Color',
                
                'btn_ok': 'OK',
                'btn_cancel': 'Cancel',

                'msg_config_saved_title': 'Configuration Saved',
                'msg_config_saved_text': 'Label configuration saved successfully!',
                'msg_config_saved_info': 'The labels are now visible on the selected stack and will be included when saving.',
                
                'msg_no_reg_labels': 'No Labels',
                'msg_reg_labels_disabled': 'Registered stack labels are not enabled.',
                'msg_reg_labels_removed': 'Labels removed from Registered Stack.',
                'msg_no_input_labels': 'No Labels',
                'msg_input_labels_disabled': 'Input stack labels are not enabled.',
                'msg_input_labels_removed': 'Labels removed from Input Stack.',
                
                 # Folder Import Dialog
                'import_folder_title': 'Import Folder',
                'import_folder_display': 'Folder: {}',
                'import_choice_label': 'How should this folder be imported?',
                'import_option_single': 'Single Image Stack (load as one stack)',
                'import_option_batch': 'Multiple Image Stacks (open batch processing)',
                
                # Downsample Dialog
                'ds_title': 'Downsample Settings',
                'ds_label': 'Set image loading scale (Downsampling):',
                'ds_hint': 'Use lower values for large images to save memory and speed up processing.',

            },
            'zh': {
                # Menu File
                'menu_file': '文件',
                'action_open_folder': '打开文件夹',
                'action_open_video': '打开视频',
                'action_save': '保存',
                'menu_save_stack': '保存堆栈',
                'menu_registered_stack': '已配准堆栈',
                'menu_input_stack': '输入堆栈',
                'action_save_folder': '保存为文件夹',
                'action_save_gif': '保存为 GIF',
                'action_clear_stack': '清空堆栈',
                'action_exit': '退出',
                
                # Menu Edit
                'menu_edit': '编辑',
                'menu_rotate': '旋转',
                'action_rotate_90_cw': '顺时针 90°',
                'action_rotate_90_ccw': '逆时针 90°',
                'action_rotate_180': '180°',
                'menu_flip': '翻转',
                'action_flip_h': '水平翻转',
                'action_flip_v': '垂直翻转',
                'action_resize': '调整大小',
                'action_add_label': '添加标签',
                'menu_del_label': '删除标签',
                'action_del_reg_label': '删除已配准堆栈标签',
                'action_del_input_label': '删除输入堆栈标签',
                
                # Menu Tools
                'menu_tools': '工具',
                'action_batch_process': '批处理',
                
                # Menu Settings
                'menu_settings': '设置',
                'menu_language': '语言',
                'action_lang_en': 'English',
                'action_lang_zh': '简体中文',
                'action_thread_settings': '线程设置',
                'action_reg_settings': '配准设置',
                'action_tile_settings': '分块设置',
                'action_stackmffv4_batch_settings': 'StackMFF V4 批量大小',
                
                # Menu Help
                'menu_help': '帮助',
                'action_env_info': '环境信息',
                'action_contact': '联系我们',
                
                # Right Panel
                'group_fusion': '融合算法',
                'radio_guided_filter': '引导滤波',
                'radio_dct': '余弦离散变换',
                'radio_dtcwt': '双数复小波变换',
                'radio_gfg': '引导滤波2',
                'radio_stackmff': 'StackMFF-V4',
                'group_registration': '图像配准',
                'check_align_ecc': 'ECC',
                'check_align_homography': 'Homography',
                'label_kernel': '核大小:',
                'btn_reset': '重置默认',
                'btn_render': '▶ 开始渲染',
                'label_source_images': '源图像: {}',
                'label_output': '输出: {}',
                'btn_roi': '选择感兴趣区域',
                'btn_render_processing': '⏳ 处理中...',
                
                # Status Panel
                'status_loaded': '已加载: {}',
                'status_gpu': 'GPU: {}',
                'status_res': '分辨率: {}',
                'status_ram': '内存: {}',
                'status_loaded_fmt': '已加载: {} ({:.1f} MB/张)',
                'status_res_fmt': '分辨率: {}x{}',
                'status_ram_fmt': '内存: {:.1f} GB / {:.0f} GB',
                
                # Main Window
                'drag_hint': '拖拽图像到此处添加源文件\n或使用图像菜单',
                'drag_roi_hint': '拖动鼠标选择感兴趣区域 (ROI)',

                # Dialogs / Messages
                'msg_load_failed': '加载失败',
                'msg_load_error': '加载错误',

                # ROI Dialog
                'dialog_roi_title': 'ROI 处理选项',
                'dialog_roi_msg': '您已选择了感兴趣区域 (ROI)。\n您希望如何处理融合结果？',
                'dialog_roi_group_output': '输出模式',
                'dialog_roi_opt_crop': '仅融合 ROI (输出裁剪图像)',
                'dialog_roi_opt_crop_tooltip': '结果将是一个仅包含所选区域的小图像。',
                'dialog_roi_opt_paste': '融合并覆盖在底图上',
                'dialog_roi_opt_paste_tooltip': '结果将是完整的底图，其中 ROI 区域被融合结果覆盖。',
                'dialog_roi_group_base': '底图选择',
                'dialog_roi_lbl_base': '选择底图:',
                
                # ROI Alignment (New)
                'roi_need_images': '请先加载至少2张图像才能使用ROI模式。',
                'roi_aligning': '对齐中...',
                'roi_ready_title': 'ROI模式就绪',
                'roi_ready_msg': '图像已使用ECC方法完成对齐。',
                'roi_ready_detail': '对齐耗时 {:.2f}s。\n\n您现在可以在右侧面板上绘制ROI区域。\n点击渲染按钮仅融合所选区域。',
                'roi_align_failed': 'ROI对齐失败。',
                'roi_aligned': '已对齐',
                'roi_display_failed': '显示对齐图像失败。',
                'result_hint': '结果将显示在这里',
                
                # Completion Dialog
                'dialog_completed_title': '处理完成',
                'dialog_completed_msg': '处理成功完成！',
                
                'info_align_method': '配准方法: {}',
                'info_align_time': '配准耗时: {:.2f}s',
                'info_fusion_method': '融合方法: {}',
                'info_fusion_time': '融合耗时: {:.2f}s',
                'info_proc_unit': '处理单元: {}',
                'info_total_time': '总耗时: {:.2f}s',
                'info_align_none_cached': '配准方法: 无 (使用缓存结果)',
                'info_align_none': '配准方法: 无',
                'info_fusion_none': '融合方法: 无',
                'val_enabled': '已启用',
                'val_none': '无',
                'val_using_cached': '使用缓存结果',
                'msg_success': '成功',
                'msg_error': '错误',
                'msg_proc_error': '处理过程中发生错误:',
                'msg_warning': '警告',
                
                # Help Dialogs
                'dialog_env_title': '环境信息',
                'env_subtitle': 'OpenFocus 环境依赖',
                'env_python': 'Python 版本',
                'env_installed': '已安装: 版本 {}',
                'env_not_installed': '未安装',
                'env_cuda_avail': 'CUDA 可用: {}',
                'env_cuda_ver': 'CUDA 版本: {}',
                'env_mps_avail': 'MPS 可用 (Apple Silicon)',
                'env_gpu_accel': 'StackMFF-V4: GPU 加速可用',
                'env_no_gpu': '警告: 无 GPU 加速 (CUDA/MPS)',
                'env_cpu_mode': 'StackMFF-V4: 可用 (CPU 模式 - 较慢)',
                'env_stackmff_unavailable': 'StackMFF-V4 融合不可用',
                'env_dtcwt_unavailable': '未安装 (DTCWT 融合不可用)',
                'env_summary': '总结',
                'env_core_dep': '核心依赖:',
                'env_core_desc': '- OpenCV, NumPy, PyQt6: 基本功能所需',
                'env_gpu_opt': 'GPU 加速 (可选):',
                'env_gpu_desc': '- PyTorch: 启用 StackMFF-V4 (提供 CPU 回退模式，但较慢)',
                'env_fusion_alg': '融合算法:',
                'env_fusion_desc': '- DTCWT 库: DTCWT 融合所需',
                
                'dialog_contact_title': '联系我们',
                'contact_info_title': '联系信息',
                'contact_email': '邮箱',
                'contact_institution': '机构',
                'contact_zju': '浙江大学',
                'contact_github': 'GitHub',
                'contact_welcome': '我们热烈欢迎贡献者添加新的融合方法，帮助 OpenFocus 成长。',
                
                'btn_close': '关闭',

                # Batch Processing Dialog
                'batch_title': '批处理',
                'batch_import_mode': '导入模式',
                'batch_mode_multi': '多个文件夹 (每个文件夹一组)',
                'batch_mode_single': '单个文件夹 (自动分割为多组)',
                'batch_stack_folders': '图像栈文件夹',
                'batch_path_placeholder': '输入路径并按回车以刷新',
                'batch_btn_add': '添加文件夹',
                'batch_btn_remove': '移除选中项',
                'batch_single_split_settings': '单文件夹分割设置',
                'batch_folder_none': '文件夹: (无)',
                'batch_split_method': '分割方式:',
                'batch_split_fixed': '固定数量',
                'batch_split_time': '时间阈值',
                'batch_images_per_stack': '每组图像数:',
                'batch_time_threshold': '时间阈值:',
                'batch_unit_images': '张',
                'batch_unit_seconds': '秒',
                'batch_preview_default': '预览: 0 张图像 → 0 组',
                'batch_preview_fmt_count': '预览: {} 张图像 → {} 组 (每组 {} 张)',
                'batch_preview_fmt_time': '预览: {} 张图像 → {} 组 (阈值: {}秒)',
                'batch_preview_no_folder': '预览: 未选择文件夹',
                'batch_btn_select_split': '选择文件夹并分割',
                'batch_output_format': '输出格式',
                'batch_format_label': '格式:',
                'batch_quality_label': '质量:',
                'batch_output_location': '输出位置',
                'batch_out_subfolder': '在源文件夹中创建子文件夹',
                'batch_subfolder_name': '子文件夹名称:',
                'batch_out_same': '与源文件夹相同',
                'batch_out_custom': '指定输出文件夹',
                'batch_btn_browse': '浏览...',
                'batch_save_aligned': '保存已配准图像栈',
                'batch_proc_options': '处理选项',
                'batch_lbl_fusion': '融合方法: {}',
                'batch_lbl_reg': '配准方法: {}',
                'batch_lbl_kernel': '核大小: {}',
                'batch_btn_start': '开始批处理',
                'batch_btn_cancel': '取消',
                
                # Help Dialogs
                'help_render_title': '融合帮助',

                # Add Label Dialog
                'add_label_title': '添加标签配置',
                'label_target_stack': '目标堆栈:',
                'label_target_input': '输入图像栈',
                'label_target_registered': '已配准图像栈',
                'label_format': '格式字符串:',
                'label_start_val': '起始值:',
                'label_interval': '步长:',
                'label_x': 'X 坐标:',
                'label_y': 'Y 坐标:',
                'label_font_size': '字体大小:',
                'label_font_family': '字体:',
                'label_custom_text': '自定义文本:',
                'label_range': '范围:',
                'label_transparent_bg': '背景透明',
                'label_bg_color': '背景颜色:',
                'label_choose_bg': '选择背景颜色',
                'label_font_color': '字体颜色:',
                'label_choose_font_color': '选择字体颜色',
                
                'btn_ok': '确定',
                'btn_cancel': '取消',

                'msg_config_saved_title': '配置已保存',
                'msg_config_saved_text': '标签配置保存成功！',
                'msg_config_saved_info': '标签现在可见于选定的堆栈，并将在保存时包含在内。',
                
                'msg_no_reg_labels': '无标签',
                'msg_reg_labels_disabled': '已配准堆栈标签未启用。',
                'msg_reg_labels_removed': '已从配准堆栈移除标签。',
                'msg_no_input_labels': '无标签',
                'msg_input_labels_disabled': '输入堆栈标签未启用。',
                'msg_input_labels_removed': '已从输入堆栈移除标签。',
                
                # Folder Import Dialog
                'import_folder_title': '导入文件夹',
                'import_folder_display': '文件夹: {}',
                'import_choice_label': '如何导入此文件夹？',
                'import_option_single': '单个图像栈 (作为一组加载)',
                'import_option_batch': '多个图像栈 (打开批处理)',
                
                # Downsample Dialog
                'ds_title': '下采样设置',
                'ds_label': '设置图像加载缩放比例 (下采样):',
                'ds_hint': '对大图像使用较低的值以节省内存并加快处理速度。',

                # Thread settings dialog
                'dialog_thread_title': '线程数设置',
                'dialog_thread_group': '线程数',
                'dialog_thread_label': '线程数 (Thread Count):',
                'dialog_thread_help_title': '线程设置帮助',
                'dialog_thread_help_text': '''<h3>线程数设置</h3>
        <p>此设置控制支持多线程的算法所使用的处理线程数。通常设置为与CPU核心数匹配（一般为 2–16）。</p>

        <h4>当前的多线程支持情况：</h4>
        <ul>
        <li>GFG-FGF：支持用户控制线程数（默认上限：8）</li>
        <li>引导滤波融合 (GFF)：支持用户控制线程数（默认上限：4）</li>
        <li>图像配准：特征提取/变换操作使用指定线程数</li>
        <li>DCT, DTCWT, StackMFF-V4：目前不使用此设置（无线程控制）</li>
        </ul>

        <p>不使用此值的算法会安全地忽略它。为获得最佳性能，请避免将线程数设置得高于物理核心数。</p>

        <p>注意：安装 <code>opencv-contrib-python</code> 可以加速某些操作（例如引导滤波）。</p>
        ''',

                # StackMFF V4 Batch settings dialog
                'dialog_stackmffv4_batch_title': 'StackMFF V4 批量设置',
                'dialog_stackmffv4_batch_group': '批量处理',
                'dialog_stackmffv4_batch_label': '批量大小 (Batch Size):',
                'dialog_stackmffv4_batch_help_title': 'StackMFF V4 批量设置帮助',
                'dialog_stackmffv4_batch_help_text': '''<h3>StackMFF V4 批量设置</h3>
        <p>此设置控制使用 StackMFF V4 神经网络融合算法处理分块时的批量大小。
        批量处理允许在 GPU 上并行处理多个分块，提高效率。</p>

        <h4>批量大小：</h4>
        <ul>
        <li><b>默认值: 2</b> - 在内存使用和速度之间取得平衡</li>
        <li><b>较高值 (4-8)</b> - 处理更快，但需要更多 GPU 内存</li>
        <li><b>较低值 (1)</b> - 内存使用较低，适用于 GPU 内存有限的情况</li>
        </ul>

        <p><b>注意：</b>如果在 GPU 上遇到内存不足的错误，请尝试减小批量大小。
        此设置仅在处理大图像并启用分块时影响 StackMFF V4。</p>

        <p>CPU 模式不会显著受此设置影响，但会保持一致性。</p>''',

                # Tile settings dialog
                'dialog_tile_title': '分块设置',
                'dialog_tile_group': '分块选项',
                'dialog_tile_enabled_label': '启用分块:',
                'dialog_tile_enabled': '启用',
                'dialog_tile_disabled': '禁用',
                'dialog_tile_block_size': '分块大小:',
                'dialog_tile_overlap': '重叠大小:',
                'dialog_tile_threshold': '启用阈值:',
                'dialog_tile_help_title': '分块设置帮助',
                'dialog_tile_help_text': '''<h3>分块设置帮助</h3>
        <p>启用分块 (tile_enabled)：启用或禁用分块处理。启用时，大图像将被分割成较小的块进行处理，以减少内存使用。</p>

        <p>分块大小 (tile_block_size)：每个方形块的大小（像素）。根据内存和速度的平衡，典型值为 512–2048。</p>

        <p>重叠大小 (tile_overlap)：相邻块之间的重叠区域（像素），用于避免合并结果时的接缝。增加重叠有助于平滑边界。</p>

        <p>启用阈值 (tile_threshold)：如果图像的最长边大于此阈值，将考虑使用分块处理。较小的图像将作为整体处理。</p>''',

                # Registration settings dialog
                'dialog_reg_title': '配准设置',
                'dialog_reg_group': '配准选项',
                'dialog_reg_downscale': '下采样宽度:',
                'dialog_reg_help_title': '配准设置帮助',
                'dialog_reg_help_text': '''<h3>下采样宽度</h3>
        <p>downscale_width 控制在提取配准特征时的预处理（下采样）宽度。较小的值可以加快特征检测速度并减少内存使用，但可能会牺牲一些几何精度。</p>

        <p>推荐值：对于大图像（>=2048px）使用 <code>1024</code>，对于中等图像使用 <code>1600</code>。仅在需要最大配准精度且有足够的CPU/GPU资源时才将其设置得更高。</p>

        <p>降低此值可加速配准并减少内存使用；增加此值可以提高非常精细图像的准确性，但会增加运行时间。</p>'''
            }
        }

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_lang = lang_code
            self.languageChanged.emit()

    def get(self, key, default=None):
        val = self.translations.get(self.current_lang, {}).get(key)
        if val is None:
            # Fallback to english
            val = self.translations.get('en', {}).get(key)
        return val if val is not None else (default if default is not None else key)
    
    def t(self, key):
        return self.get(key)

# Global instance
trans = TranslationManager()

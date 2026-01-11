# dialogs/help.py
"""Help and information dialogs."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextOption, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextBrowser,
    QPushButton,
)
from ui.styles import PRIMARY_BLUE
from locales import trans
from utils import resource_path


class HelpDialog(QDialog):
    """帮助信息对话框（从 main.py 抽离）"""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)

        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #1e1e1e;
            }}
            QTextBrowser {{
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                font-family: 'Segoe UI', 'Microsoft YaHei';
                font-size: 13px;
                selection-background-color: {PRIMARY_BLUE};
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
        """)

        layout = QVBoxLayout(self)

        # 创建可滚动的文本浏览器
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(content)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_browser)

        # 关闭按钮
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # 居中显示
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )


class RenderMethodHelpDialog(HelpDialog):
    """渲染方法帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Render Methods</h3>
        
    <p>Guided Filter<br/>
    Guided-filter fusion tuned for practical edge preservation. Ideal for simpler scenes or moderate focus variations, and you can fine-tune the kernel slider to balance sharpness and smoothness.</p>

    <p>DCT<br/>
    Frequency-domain fusion that evaluates block-wise DCT variance and keeps the sharpest contributor per region. It is fast, fully CPU-based, and works well when you need crisp edges without deploying neural models.</p>

    <p>DTCWT<br/>
    Dual-tree complex wavelet fusion that decomposes the stack across scales and orientations before recombining it. It is well suited to intricate, high-frequency content where retaining fine detail is critical.</p>

    <p>GFG-FGF<br/>
    GFG-FGF is a multi-focus image fusion algorithm based on a generalized four-neighborhood Gaussian gradient (GFG) operator combined with a fast guided filter (FGF). Feature extraction uses the GFG operator to capture high-frequency edge and gradient information. Information enhancement leverages the FGF together with the original image texture to smooth defocused regions while emphasizing focused areas. The fusion strategy constructs a pixel-wise decision map by selecting the maximum focus measure per pixel and then refines these decisions with FGF for edge-preserving smoothing, producing a weighted fusion that favors sharp, well-focused pixels.</p>

    <p>StackMFF-V4<br/>
    A neural network trained on everyday focus stacks. It generally produces the strongest results with minimal tuning. Because it is not fine-tuned for specialist domains (microphotography, microscopy, medical imaging, etc.), avoid it when domain shifts are expected. Runs fastest with GPU acceleration.</p>"""
        
        super().__init__(trans.t('help_render_title'), help_text, parent)


class RegistrationHelpDialog(HelpDialog):
    """配准方法帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Registration Methods</h3>
        
    <p>Align (Homography)<br/>
    Uses feature-based homography transformation to align images. Detects SIFT features between consecutive frames and computes perspective transformation matrices. Ideal for most focus stacks that need global geometric correction.</p>

    <p>Align (ECC)<br/>
    Enhanced Correlation Coefficient alignment refines alignment at the sub-pixel level. Works well for fine adjustments or whenever feature detection is unreliable.</p>

    <p>Both options are independent—enable either one individually or turn on both to apply homography alignment first and then refine with ECC.</p>"""
        
        super().__init__("Registration Help", help_text, parent)


class TileHelpDialog(HelpDialog):
    """Tile 参数帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Tile Settings Help</h3>
        <p>tile_enabled: Enable or disable tiled processing. When enabled, large images
        will be processed in smaller blocks to reduce memory usage.</p>

        <p>tile_block_size: Size (in pixels) of each square tile block. Typical values
        are 512–2048 depending on memory and speed tradeoffs.</p>

        <p>tile_overlap: Overlap (in pixels) between adjacent tiles used to avoid seams
        when combining results. A positive overlap helps smooth boundaries.</p>

        <p>tile_threshold: If the image's longest side is larger than this threshold,
        tiled processing will be considered. Smaller images are processed as a whole.</p>"""

        super().__init__("Tile Settings Help", help_text, parent)

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from styles import MESSAGE_BOX_STYLE


def show_message_box(
    parent,
    title,
    text,
    informative_text="",
    icon=QMessageBox.Icon.Information,
):
    """
    工具函数：统一创建深色主题的 QMessageBox，避免 main.py 里重复样板代码。
    """
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(MESSAGE_BOX_STYLE)
    msg_box.exec()


def show_warning_box(parent, title, text, informative_text=""):
    """
    工具函数：创建警告消息框
    """
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Warning)


def show_error_box(parent, title, text, informative_text=""):
    """
    工具函数：创建错误消息框
    """
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Critical)


def show_success_box(parent, title, text, informative_text=""):
    """
    工具函数：创建成功消息框
    """
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Information)


def show_custom_message_box(parent, title, text, informative_text="", icon=QMessageBox.Icon.Information, style_sheet=MESSAGE_BOX_STYLE):
    """
    工具函数：创建自定义样式的消息框
    """
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(style_sheet)
    msg_box.exec()


def pixmap_to_cv2(pixmap):
    """将QPixmap转换为OpenCV图像"""
    try:
        # 将QPixmap转换为QImage
        qimage = pixmap.toImage()

        # 统一转换为 RGBA8888，确保每像素4字节
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        # 正确设置缓冲区大小，再从缓冲区构造 numpy 数组
        ptr = qimage.bits()
        # 在 PyQt6 中必须先设置缓冲区大小，否则 np.frombuffer 读不到完整数据
        ptr.setsize(bytes_per_line * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))  # RGBA

        # 转换为 BGR（OpenCV 使用 BGR）
        bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        return bgr_image
    except Exception as e:
        print(f"Error converting QPixmap to OpenCV image: {str(e)}")
        return None


def cv2_to_pixmap(cv2_img):
    """将OpenCV图像转换为QPixmap"""
    try:
        # 转换为RGB格式（Qt使用）
        if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
        
        # 复制图像数据以确保内存连续性
        rgb_image = np.ascontiguousarray(rgb_image)
        
        # 转换为QImage
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 转换为QPixmap
        pixmap = QPixmap.fromImage(qimage)
        
        return pixmap
    except Exception as e:
        print(f"Error converting OpenCV image to QPixmap: {str(e)}")
        return QPixmap()
    

class LabelConfig:
    """标签配置类"""
    
    def __init__(self):
        self.target_stack = 1  # 0 for input stack, 1 for registered stack
        self.format = "{value}"
        self.starting_value = 1
        self.interval = 1
        self.x_location = 20
        self.y_location = 80
        self.font_size = 80
        self.font_family = "Arial"
        self.text = ""
        self.range = "All"
        self.transparent_bg = True
        self.bg_color = (0, 0, 0)  # BGR format
        self.font_color = (255, 255, 255)  # BGR format
    
    def update_config(self, config_dict):
        """更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LabelAdder:
    """标签添加器类"""
    
    def __init__(self):
        self.config = LabelConfig()
        # 字体映射
        self.font_mapping = {
            'Arial': cv2.FONT_HERSHEY_SIMPLEX,
            'Times New Roman': cv2.FONT_HERSHEY_SIMPLEX,
            'Courier New': cv2.FONT_HERSHEY_TRIPLEX,
            'Calibri': cv2.FONT_HERSHEY_SIMPLEX,
            'Verdana': cv2.FONT_HERSHEY_SIMPLEX,
            'Georgia': cv2.FONT_HERSHEY_SIMPLEX,
            'Helvetica': cv2.FONT_HERSHEY_SIMPLEX,
            'Comic Sans MS': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'Impact': cv2.FONT_HERSHEY_SIMPLEX,
            'Lucida Console': cv2.FONT_HERSHEY_TRIPLEX,
            'Tahoma': cv2.FONT_HERSHEY_SIMPLEX,
            'Trebuchet MS': cv2.FONT_HERSHEY_SIMPLEX,
            'Palatino': cv2.FONT_HERSHEY_SIMPLEX,
            'Garamond': cv2.FONT_HERSHEY_SIMPLEX,
            'Bookman': cv2.FONT_HERSHEY_SIMPLEX
        }
    
    def add_label_to_image(self, image, index):
        """
        在图像上添加标签
        
        Args:
            image: OpenCV图像 (BGR格式)
            index: 图像索引
            
        Returns:
            添加标签后的图像
        """
        try:
            # 解析配置
            format_str = self.config.format
            starting_value = self.config.starting_value
            interval = self.config.interval
            x_location = self.config.x_location
            y_location = self.config.y_location
            font_size = self.config.font_size
            font_family = self.config.font_family
            text = self.config.text
            transparent_bg = self.config.transparent_bg
            bg_color = self.config.bg_color
            font_color = self.config.font_color
            
            # 计算当前帧的值
            current_value = starting_value + index * interval
            
            # 替换格式字符串中的占位符
            if '{value}' in format_str:
                label_text = format_str.replace('{value}', str(current_value))
            else:
                label_text = text
            
            # 选择字体
            font = self.font_mapping.get(font_family, cv2.FONT_HERSHEY_SIMPLEX)
            
            font_scale = font_size / 30.0  # 调整字体大小比例
            thickness = max(1, int(font_size / 15))  # 调整线条粗细
            
            # 获取文本大小以便绘制背景
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # 根据用户设置决定是否绘制背景
            if not transparent_bg:
                # 绘制背景矩形
                cv2.rectangle(
                    image, 
                    (x_location, y_location - text_height - 10), 
                    (x_location + text_width + 10, y_location + baseline + 5), 
                    bg_color, 
                    -1
                )
            
            # 绘制文字
            cv2.putText(
                image, 
                label_text, 
                (x_location + 5, y_location), 
                font, 
                font_scale, 
                font_color, 
                thickness
            )
            
            return image
        except Exception as e:
            print(f"Error adding label to image: {str(e)}")
            return image
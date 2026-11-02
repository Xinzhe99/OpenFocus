"""Controls for optional sensor-space cleanup."""
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

from locales import trans

_INSPECTIONS = set()


class _Inspection(QThread):
    inspected = pyqtSignal(int, object, str)

    def __init__(self, serial, path):
        super().__init__()
        self.serial = serial
        self.path = path

    def run(self):
        try:
            from utils.sensor_cleanup import load_sensor_defect_mask
            mask = load_sensor_defect_mask(self.path)
            self.inspected.emit(self.serial, mask, '')
        except Exception as exc:
            self.inspected.emit(self.serial, None, str(exc))



class SensorCleanupControls(QWidget):
    def __init__(self, path='', denoise=False, parent=None, *, show_scope=True):
        super().__init__(parent)
        self._serial = 0
        self._inspected_path = None
        self._pending = False
        self.mask = None
        self.error = ''
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(trans.t('sensor_mask')))
        self.path = QLineEdit(path or '')
        self.path.setPlaceholderText(trans.t('sensor_mask_none'))
        row.addWidget(self.path, 1)
        browse = QPushButton(trans.t('dust_browse'))
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.extent = QLabel()
        self.extent.setWordWrap(True)
        layout.addWidget(self.extent)
        hint = QLabel(trans.t('sensor_mask_hint'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.denoise = QCheckBox(trans.t('sensor_denoise'))
        layout.addWidget(self.denoise)
        if show_scope:
            scope = QLabel(trans.t('sensor_settings_scope'))
            scope.setWordWrap(True)
            layout.addWidget(scope)
        self.denoise.setChecked(bool(denoise))
        self.path.editingFinished.connect(self._inspect)
        self._inspect()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, trans.t('sensor_mask'), '',
            'Images (*.png *.tif *.tiff *.bmp)')
        if path:
            self.path.setText(path)
            self._inspect()

    def _inspect(self):
        path = self.path.text().strip()
        if path == self._inspected_path:
            return
        self._serial += 1
        self._inspected_path = path
        self.mask = None
        self.error = ''
        self._pending = bool(path)
        if not path:
            self.extent.setText(trans.t('sensor_mask_none'))
            return
        self.extent.setText(trans.t('sensor_mask_checking'))
        job = _Inspection(self._serial, path)
        _INSPECTIONS.add(job)
        job.inspected.connect(self._inspection_finished)
        job.finished.connect(lambda: (_INSPECTIONS.discard(job), job.deleteLater()))
        job.start()

    def _inspection_finished(self, serial, mask, error):
        if serial != self._serial or self.path.text().strip() != self._inspected_path:
            return
        self._pending = False
        self.mask = mask
        self.error = error
        if error:
            self.extent.setText(trans.t('sensor_mask_invalid').format(error=error))
        elif mask is not None:
            self.extent.setText(trans.t('sensor_mask_extent').format(
                width=mask.width, height=mask.height, coverage=mask.coverage * 100))

    def validate(self, window=None):
        if self.path.text().strip() != self._inspected_path:
            self._inspect()
        if self._pending:
            return trans.t('sensor_mask_checking')
        if self.error:
            return trans.t('sensor_mask_invalid').format(error=self.error)
        if self.mask is not None and window is not None:
            from utils.path_identity import same_path
            path = self.path.text().strip()
            if any(same_path(path, source) for source in
                   (getattr(window, 'image_source_paths', None) or [])):
                return trans.t('sensor_mask_in_stack')
            images = getattr(window, 'raw_images', None) or []
            if images and (float(getattr(window, 'current_scale_factor', 1.0) or 1.0) != 1.0
                           or any(image.shape[:2] != (self.mask.height, self.mask.width)
                                  for image in images)):
                return trans.t('sensor_mask_size_error')
        return ''

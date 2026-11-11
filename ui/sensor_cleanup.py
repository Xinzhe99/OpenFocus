"""Controls for optional sensor-space cleanup."""
import weakref
from PyQt6 import sip
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

from locales import trans

_INSPECTIONS = set()
_RETIREMENTS = {}


class _Inspection(QThread):
    inspected = pyqtSignal(int, object, str)

    def __init__(self, serial, path):
        super().__init__()
        self.serial = serial
        self.path = path

    def run(self):
        from utils.sensor_cleanup import load_sensor_defect_mask, SensorCleanupCancelled
        try:
            mask = load_sensor_defect_mask(
                self.path, cancel_callback=self.isInterruptionRequested)
            self.inspected.emit(self.serial, mask, '')
        except SensorCleanupCancelled:
            pass
        except Exception as exc:
            self.inspected.emit(self.serial, None, str(exc))


class _InspectionRetirement(QObject):
    """Keep a worker alive until Qt reports its actual completion."""
    def __init__(self, owner, job):
        super().__init__()
        self.owner = weakref.ref(owner)
        self.job = job

    @pyqtSlot()
    def cancel(self):
        job = self.job
        if job is not None and not sip.isdeleted(job):
            job.requestInterruption()

    @pyqtSlot()
    def finished(self):
        job = self.job
        if job is not None:
            _retire(self.owner, job)



class SensorCleanupControls(QWidget):
    def __init__(self, path='', denoise=False, parent=None, *, show_scope=True):
        super().__init__(parent)
        self._serial = 0
        self._result_path = None
        self._result_mask = None
        self._result_error = ''
        self._worker = None
        self._queued = None
        self._stopped = False
        self.mask = None
        self.error = ''
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(trans.t('sensor_mask')))
        self.path = QLineEdit(path or '')
        self.path.setPlaceholderText(trans.t('sensor_mask_none'))
        mask_hint = trans.t('sensor_mask_hint')
        if show_scope:
            mask_hint += '\n\n' + trans.t('sensor_settings_scope')
        self.path.setToolTip(mask_hint)
        row.addWidget(self.path, 1)
        browse = QPushButton(trans.t('dust_browse'))
        browse.setToolTip(mask_hint)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.extent = QLabel()
        self.extent.setWordWrap(True)
        layout.addWidget(self.extent)
        self.denoise = QCheckBox(trans.t('sensor_denoise'))
        layout.addWidget(self.denoise)
        if show_scope:
            self.denoise.setToolTip(trans.t('sensor_settings_scope'))
        self.denoise.setChecked(bool(denoise))
        self.path.textChanged.connect(self._text_changed)
        self.path.editingFinished.connect(self._inspect)
        self._inspect()

    def _browse(self):
        path, _ = FileDialog.getOpenFileName(
            self, trans.t('sensor_mask'), '',
            'Images (*.png *.tif *.tiff *.bmp)')
        if path:
            self.path.setText(path)
            self._inspect()

    def _text_changed(self, _text):
        self.mask, self.error = None, ''
        self.extent.setText(trans.t('sensor_mask_checking') if self.path.text().strip()
                            else trans.t('sensor_mask_none'))
        if not self.path.text().strip():
            self._inspect()

    def _inspect(self):
        if self._stopped:
            return
        path = self.path.text().strip()
        if not path:
            self._serial += 1
            self._queued = None
            self._result_path = self._result_mask = None
            self._result_error = ''
            self.mask, self.error = None, ''
            if self._worker is not None:
                self._worker.requestInterruption()
            self.extent.setText(trans.t('sensor_mask_none'))
            return
        if self._result_path == path:
            if self._queued or (self._worker is not None and self._worker.path != path):
                self._serial += 1
                self._queued = None
                if self._worker is not None:
                    self._worker.requestInterruption()
            self._show_result()
            return
        if self._queued and self._queued[1] == path:
            return
        if (self._worker is not None and self._worker.path == path
                and not self._worker.isInterruptionRequested()):
            return
        self._serial += 1
        self.mask, self.error = None, ''
        self.extent.setText(trans.t('sensor_mask_checking'))
        self._queued = (self._serial, path)
        if self._worker is not None:
            self._worker.requestInterruption()
        else:
            self._start_queued()

    def _start_queued(self):
        if self._stopped or not self._queued:
            return
        serial, path = self._queued
        self._queued = None
        job = self._worker = _Inspection(serial, path)
        _INSPECTIONS.add(job)
        retirement = _InspectionRetirement(self, job)
        _RETIREMENTS[job] = retirement
        self.destroyed.connect(retirement.cancel)
        job.inspected.connect(self._inspection_finished)
        job.finished.connect(retirement.finished)
        job.start()

    def _inspection_finished(self, serial, mask, error):
        if self._stopped or serial != self._serial or self._worker is None:
            return
        self._result_path = self._worker.path
        self._result_mask, self._result_error = mask, error
        if self.path.text().strip() == self._result_path:
            self._show_result()

    def _show_result(self):
        self.mask, self.error = self._result_mask, self._result_error
        if self.error:
            self.extent.setText(trans.t('sensor_mask_invalid').format(error=self.error))
        elif self.mask is not None:
            self.extent.setText(trans.t('sensor_mask_extent').format(
                width=self.mask.width, height=self.mask.height, coverage=self.mask.coverage * 100))

    def stop_inspection(self):
        self._stopped = True
        self._serial += 1
        self._queued = None
        self._result_path = self._result_mask = None
        self._result_error = ''
        self.mask, self.error = None, ''
        if self._worker is not None:
            self._worker.requestInterruption()

    def closeEvent(self, event):
        self.stop_inspection()
        super().closeEvent(event)

    def validate(self, window=None):
        self._inspect()
        path = self.path.text().strip()
        if not path:
            return ''
        if self._result_path != path:
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



def _retire(owner, job):
    _INSPECTIONS.discard(job)
    retirement = _RETIREMENTS.pop(job, None)
    if retirement is not None:
        retirement.job = None
        retirement.deleteLater()
    controls = owner()
    if controls is not None and not sip.isdeleted(controls) and controls._worker is job:
        controls._worker = None
        window = controls.window()
        if not controls._stopped and not getattr(window, '_shutting_down', False):
            controls._start_queued()
    job.deleteLater()


def cancel_sensor_inspections(window):
    for controls in window.findChildren(SensorCleanupControls):
        if not sip.isdeleted(controls):
            controls.stop_inspection()
    for job in tuple(_INSPECTIONS):
        job.requestInterruption()
    return any(job.isRunning() for job in tuple(_INSPECTIONS))

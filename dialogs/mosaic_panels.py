"""GUI-thread snapshots of the mosaic renderer and per-source optics settings."""
from __future__ import annotations

import os

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from core.mosaic import BlendSpec, OutputSpec, RenderSpec
from core.mosaic_hdr import HDRSpec
from core.mosaic_photo import PhotometricSpec
from core.mosaic_projection import CameraSpec, LensSpec, ProjectionSpec
from core.mosaic_seam import SeamSpec
from locales import trans
from widgets.stepper import StepperSpinBox


class MosaicSettingsPanel(QWidget):
    """Own settings only; the parent snapshots these before dispatching a worker.

    Exposure value zero means use the current frame metadata. No source metadata
    is modified. Validation errors from the core specs propagate to the caller.
    """

    changed = pyqtSignal()
    overlaysChanged = pyqtSignal()
    sourceSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = ()
        self._states = {}
        self._named_states = {}
        self._selected = -1
        self._loading = False
        self._labels = []
        self._options = []
        self._checks = []
        self._groups = []
        layout = QVBoxLayout(self)
        source_form = QFormLayout()
        self.source_choice = QComboBox()
        self.source_choice.currentIndexChanged.connect(self._source_selected)
        self._row(source_form, 'mosaic_sources', self.source_choice)
        layout.addLayout(source_form)

        def group(key):
            box = QGroupBox()
            self._groups.append((box, key))
            layout.addWidget(box)
            return QFormLayout(box)

        seam = group('mosaic_seams')
        modes = [('mosaic_dp', 'dp')]
        if callable(getattr(getattr(cv2, 'detail', None), 'GraphCutSeamFinder', None)):
            modes.append(('mosaic_graphcut', 'graphcut'))
        modes.append(('mosaic_feather', 'feather'))
        self.seam_mode = self._combo(seam, 'mosaic_seam_mode', modes)
        self.precision = self._combo(seam, 'mosaic_precision', [(str(v), v) for v in (1, 2, 4, 8)], numeric=True)
        self.precision.setCurrentIndex(3)
        self.band = self._integer(seam, 'mosaic_band', 0, 2048, 64)
        self.cost = self._combo(seam, 'mosaic_cost', [('mosaic_gradient_cost', 'color_gradient'), ('mosaic_color_cost', 'color')])
        self.loop_refine = self._check(seam, 'mosaic_loop_refine')

        blend = group('mosaic_blending')
        self.blend_mode = self._combo(blend, 'mosaic_blending', [('mosaic_multiband', 'multiband'), ('mosaic_feather', 'feather'), ('mosaic_hard', 'hard')])
        self.levels = self._integer(blend, 'mosaic_levels', 1, 7, 5)
        photo = group('mosaic_photometric')
        self.photo_mode = self._combo(photo, 'mosaic_photometric', [('mosaic_off', 'off'), ('mosaic_exposure', 'exposure'), ('mosaic_color_balance', 'color'), ('mosaic_gain_field', 'gain_field')])
        self.linear = self._check(photo, 'mosaic_linear')

        output = group('mosaic_output_settings')
        self.depth = self._combo(output, 'mosaic_depth', [('mosaic_auto_value', None), ('mosaic_u8', 'uint8'), ('mosaic_u16', 'uint16')])
        self.icc = self._combo(output, 'mosaic_icc', [('mosaic_icc_auto', 'auto'), ('mosaic_icc_srgb', 'srgb'), ('mosaic_icc_first', 'keep_first'), ('mosaic_icc_drop', 'drop')])
        self.keep_exif = self._check(output, 'mosaic_keep_exif', True)
        self.alpha = self._combo(output, 'mosaic_alpha', [('mosaic_auto_value', 'auto'), ('mosaic_enabled', 'on'), ('mosaic_off', 'off')])
        self.output_scale = self._real(output, 'mosaic_output_scale', 0.01, 10, 1, 6)
        self.crop = tuple(self._real(output, key, 0, 1, default, 6) for key, default in (
            ('mosaic_crop_left', 0), ('mosaic_crop_top', 0), ('mosaic_crop_right', 1), ('mosaic_crop_bottom', 1)))
        self.icc.setToolTip(trans.t('mosaic_icc_hint'))

        projection = group('mosaic_projection')
        self.projection = self._combo(projection, 'mosaic_projection', [('mosaic_planar', None), ('mosaic_rectilinear', 'rectilinear'), ('mosaic_cylindrical', 'cylindrical'), ('mosaic_equirectangular', 'equirectangular')])
        self.projection_focal = self._real(projection, 'mosaic_projection_focal', 0.000001, 1000000000, 1000, 6)
        self.optimize_fov = self._check(projection, 'mosaic_optimize_fov')
        camera = group('mosaic_camera_settings')
        self.camera_hint = QLabel()
        self.camera_hint.setWordWrap(True)
        camera.addRow(self.camera_hint)
        self._camera = {}
        for name, key, low, high, default in (
            ('yaw', 'mosaic_yaw', -360, 360, 0),
            ('pitch', 'mosaic_pitch', -180, 180, 0),
            ('roll', 'mosaic_roll', -180, 180, 0),
            ('hfov', 'mosaic_fov', 0.000001, 179.999999, 60),
            ('a', 'mosaic_lens_a', -100, 100, 0),
            ('b', 'mosaic_lens_b', -100, 100, 0),
            ('c', 'mosaic_lens_c', -100, 100, 0),
            ('shift_x', 'mosaic_lens_x', -1000000000, 1000000000, 0),
            ('shift_y', 'mosaic_lens_y', -1000000000, 1000000000, 0),
        ):
            widget = self._real(camera, key, low, high, default, 6)
            widget.valueChanged.disconnect(self._changed)
            widget.valueChanged.connect(self._camera_changed)
            self._camera[name] = widget
        self._vignette = tuple(self._real(camera, 'mosaic_vignette_k' + str(i), -1000, 1000, 0, 6) for i in (1, 2, 3))
        for widget in self._vignette:
            widget.valueChanged.disconnect(self._changed)
            widget.valueChanged.connect(self._camera_changed)
        self.exposure_seconds = self._real(camera, 'mosaic_exposure_seconds', 0, 1000000000, 0, 9)
        self.exposure_seconds.valueChanged.disconnect(self._changed)
        self.exposure_seconds.valueChanged.connect(self._store_exposure)

        hdr = group('mosaic_hdr')
        self.hdr_mode = self._combo(hdr, 'mosaic_hdr', [('mosaic_off', 'off'), ('mosaic_exposure_fusion', 'exposure_fusion'), ('mosaic_debevec', 'debevec')])
        self.bracket_size = self._integer(hdr, 'mosaic_bracket_size', 1, 10000, 1)
        self.exposure_sigma = self._real(hdr, 'mosaic_exposure_sigma', 0.000001, 100, 0.2, 6)
        self.exposure_target = self._real(hdr, 'mosaic_exposure_target', 0, 1, 0.5, 6)
        self.tone_key = self._real(hdr, 'mosaic_tone_key', 0.000001, 10000, 0.18, 6)
        self.tone_white = self._real(hdr, 'mosaic_tone_white', 0.000001, 1000000, 4, 6)
        self.hdr_hint = QLabel()
        self.hdr_hint.setWordWrap(True)
        hdr.addRow(self.hdr_hint)
        overlays = group('mosaic_preview')
        self.show_seams = self._check(overlays, 'mosaic_show_seams')
        self.show_numbers = self._check(overlays, 'mosaic_show_numbers')
        self.show_masks = self._check(overlays, 'mosaic_show_masks')
        for widget in (self.show_seams, self.show_numbers, self.show_masks):
            widget.toggled.disconnect(self._changed)
            widget.toggled.connect(lambda _checked: self.overlaysChanged.emit())
        layout.addStretch()
        self.projection.currentIndexChanged.connect(self._applicability)
        self.seam_mode.currentIndexChanged.connect(self._applicability)
        self.blend_mode.currentIndexChanged.connect(self._applicability)
        self.hdr_mode.currentIndexChanged.connect(self._applicability)
        trans.languageChanged.connect(self._retranslate)
        self._retranslate()
        self._applicability()

    def _row(self, form, key, widget):
        label = QLabel()
        label.setBuddy(widget)
        self._labels.append((label, key))
        form.addRow(label, widget)
        return widget

    def _combo(self, form, key, options, numeric=False):
        widget = QComboBox()
        for label, value in options:
            widget.addItem(label if numeric else trans.t(label), value)
        if not numeric:
            self._options.append((widget, tuple(label for label, _ in options)))
        widget.currentIndexChanged.connect(self._changed)
        return self._row(form, key, widget)

    def _integer(self, form, key, low, high, default):
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(default)
        widget.valueChanged.connect(self._changed)
        return self._row(form, key, widget)

    def _real(self, form, key, low, high, default, decimals):
        widget = StepperSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(default)
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self._changed)
        return self._row(form, key, widget)

    def _check(self, form, key, checked=False):
        widget = QCheckBox()
        widget.setChecked(checked)
        self._checks.append((widget, key))
        widget.toggled.connect(self._changed)
        form.addRow(widget)
        return widget

    def _retranslate(self):
        for label, key in self._labels:
            label.setText(trans.t(key))
        for box, key in self._groups:
            box.setTitle(trans.t(key))
        for widget, key in self._checks:
            widget.setText(trans.t(key))
        for combo, keys in self._options:
            for index, key in enumerate(keys):
                combo.setItemText(index, trans.t(key))
        self.camera_hint.setText(trans.t('mosaic_camera_required'))
        self.hdr_hint.setText(trans.t('mosaic_hdr_hint'))
        self.exposure_seconds.setSpecialValueText(trans.t('mosaic_auto_value'))
        self.icc.setToolTip(trans.t('mosaic_icc_hint'))

    def _gui_thread(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError('Mosaic settings must be accessed on the GUI thread')

    def _changed(self, *_):
        if not self._loading:
            self.changed.emit()

    @staticmethod
    def _identity(frame):
        if frame.source_path:
            return ('path', os.path.normcase(os.path.abspath(frame.source_path)), id(frame.image))
        return ('image', id(frame.image))

    def _state(self, frame):
        key = self._identity(frame)
        if key not in self._states:
            self._states[key] = dict(yaw=0.0, pitch=0.0, roll=0.0, hfov=60.0,
                                     a=0.0, b=0.0, c=0.0, shift_x=0.0, shift_y=0.0,
                                     vignette=(0.0, 0.0, 0.0), exposure=None)
        return self._states[key]

    def _store_selected(self, *_):
        if self._loading or not 0 <= self._selected < len(self._frames):
            return
        state = self._state(self._frames[self._selected])
        state.update({name: widget.value() for name, widget in self._camera.items()})
        state['vignette'] = tuple(widget.value() for widget in self._vignette)

    def _camera_changed(self, *_):
        self._store_selected()
        self._changed()

    def _store_exposure(self, value):
        if not self._loading and 0 <= self._selected < len(self._frames):
            self._state(self._frames[self._selected])['exposure'] = value if value > 0 else None
            self.changed.emit()

    def set_frames(self, frames):
        self._gui_thread()
        self._store_selected()
        previous = self._identity(self._frames[self._selected]) if 0 <= self._selected < len(self._frames) else None
        self._frames = tuple(frames)
        self.source_choice.blockSignals(True)
        try:
            self.source_choice.clear()
            for index, frame in enumerate(self._frames):
                self.source_choice.addItem(f'{index + 1}: {frame.name}')
        finally:
            self.source_choice.blockSignals(False)
        name_counts = {}
        for frame in self._frames:
            name_counts[frame.name] = name_counts.get(frame.name, 0) + 1
        for frame in self._frames:
            key = self._identity(frame)
            previous_state = self._named_states.get(frame.name)
            if (key not in self._states and not frame.source_path and frame.name
                    and name_counts[frame.name] == 1 and previous_state is not None):
                self._states[key] = dict(previous_state)
            state = self._state(frame)
            self._named_states[frame.name] = state if name_counts[frame.name] == 1 else None
        selected = next((i for i, frame in enumerate(self._frames) if self._identity(frame) == previous), 0 if self._frames else -1)
        self._selected = -1
        self.select_frame(selected)

    def _source_selected(self, index):
        self.select_frame(index)
        self.sourceSelected.emit(index)

    def select_frame(self, index):
        self._gui_thread()
        self._store_selected()
        self._selected = index if 0 <= index < len(self._frames) else -1
        self._loading = True
        try:
            self.source_choice.blockSignals(True)
            try:
                self.source_choice.setCurrentIndex(self._selected)
            finally:
                self.source_choice.blockSignals(False)
            if self._selected >= 0:
                state = self._state(self._frames[self._selected])
                for name, widget in self._camera.items():
                    widget.setValue(state[name])
                for widget, value in zip(self._vignette, state['vignette']):
                    widget.setValue(value)
                # Zero retains the live metadata fallback instead of copying it.
                self.exposure_seconds.setValue(state['exposure'] or 0)
        finally:
            self._loading = False
        self._applicability()

    def _applicability(self, *_):
        panorama = self.projection.currentData() is not None
        self.projection_focal.setEnabled(panorama)
        self.optimize_fov.setEnabled(panorama)
        self.camera_hint.setVisible(not panorama)
        for widget in self._camera.values():
            widget.setEnabled(panorama and self._selected >= 0)
        for widget in (*self._vignette, self.exposure_seconds):
            widget.setEnabled(self._selected >= 0)
        content = self.seam_mode.currentData() != 'feather'
        for widget in (self.precision, self.cost, self.loop_refine):
            widget.setEnabled(content)
        self.levels.setEnabled(self.blend_mode.currentData() == 'multiband' or self.hdr_mode.currentData() == 'exposure_fusion')
        mode = self.hdr_mode.currentData()
        self.bracket_size.setEnabled(mode != 'off')
        self.exposure_sigma.setEnabled(mode == 'exposure_fusion')
        self.exposure_target.setEnabled(mode == 'exposure_fusion')
        self.tone_key.setEnabled(mode == 'debevec')
        self.tone_white.setEnabled(mode == 'debevec')

    def camera_specs(self):
        self._gui_thread()
        self._store_selected()
        result = []
        for frame in self._frames:
            state = self._state(frame)
            height, width = frame.image.shape[:2]
            lens = LensSpec(**{key: state[key] for key in ('a', 'b', 'c', 'shift_x', 'shift_y')})
            result.append(CameraSpec(width=width, height=height, lens=lens,
                                     **{key: state[key] for key in ('yaw', 'pitch', 'roll', 'hfov')}))
        return tuple(result)

    def set_cameras(self, cameras):
        self._gui_thread()
        cameras = tuple(cameras)
        if len(cameras) != len(self._frames):
            raise ValueError('One camera is required for every source frame')
        self._store_selected()
        for frame, camera in zip(self._frames, cameras):
            state = self._state(frame)
            state.update({key: getattr(camera, key) for key in ('yaw', 'pitch', 'roll', 'hfov')})
            state.update({key: getattr(camera.lens, key) for key in ('a', 'b', 'c', 'shift_x', 'shift_y')})
        selected = self._selected
        self._selected = -1
        self.select_frame(selected)
        self.changed.emit()

    def overlay_options(self):
        self._gui_thread()
        return dict(show_seams=self.show_seams.isChecked(),
                    show_numbers=self.show_numbers.isChecked(),
                    show_masks=self.show_masks.isChecked())

    def render_spec(self, frames):
        self._gui_thread()
        self._store_selected()
        frames = tuple(frames)
        coefficients = tuple(self._state(frame)['vignette'] for frame in frames)
        if not any(any(row) for row in coefficients):
            coefficients = ()
        mode = self.hdr_mode.currentData()
        times = ()
        if mode != 'off':
            # Missing times remain invalid and are reported by HDRSpec for
            # Debevec. Exposure fusion does not require calibrated timings.
            values = tuple(self._state(frame)['exposure'] if self._state(frame)['exposure'] is not None
                           else getattr(frame, 'exposure_time', None) for frame in frames)
            if mode == 'debevec' or all(value is not None for value in values):
                times = tuple(float(value) if value is not None else 0.0 for value in values)
        bracket = self.bracket_size.value() if mode != 'off' else 1
        groups = ()
        if mode != 'off':
            if bracket < 2 or not frames or len(frames) % bracket:
                raise ValueError(trans.t('mosaic_hdr_hint'))
            groups = tuple(tuple(range(start, start + bracket))
                           for start in range(0, len(frames), bracket))
        projection = self.projection.currentData()
        cameras = ()
        if projection is not None:
            if tuple(self._identity(frame) for frame in frames) != tuple(self._identity(frame) for frame in self._frames):
                raise ValueError('Settings frames must match the render source order')
            cameras = self.camera_specs()
        return RenderSpec(
            seam=SeamSpec(mode=self.seam_mode.currentData(), precision=self.precision.currentData(), band=self.band.value(), cost=self.cost.currentData(), loop_refine=self.loop_refine.isChecked()),
            blend=BlendSpec(mode=self.blend_mode.currentData(), levels=self.levels.value()),
            photometric=PhotometricSpec(mode=self.photo_mode.currentData(), linear=self.linear.isChecked(), vignette_coefficients=coefficients),
            output=OutputSpec(dtype=self.depth.currentData(), icc_policy=self.icc.currentData(), preserve_exif=self.keep_exif.isChecked(), scale=self.output_scale.value(), crop=tuple(widget.value() for widget in self.crop), alpha=self.alpha.currentData()),
            projection=ProjectionSpec(kind=projection, focal=self.projection_focal.value()) if projection is not None else None,
            cameras=cameras,
            hdr=HDRSpec(mode=mode, exposure_times=times, exposure_sigma=self.exposure_sigma.value(), exposure_target=self.exposure_target.value(), levels=self.levels.value(), tone_key=self.tone_key.value(), tone_white=self.tone_white.value()),
            bracket_size=bracket, bracket_groups=groups,
        )

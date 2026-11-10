"""Lazy, bounded source-list tooltip metadata reads."""
import html
import re
import os

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QEvent, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QToolTip

from locales import trans


_FIELDS = ((271, 'source_hover_make'), (272, 'source_hover_model'),
           (36867, 'source_hover_date'), (33434, 'source_hover_exposure'),
           (33437, 'source_hover_aperture'), (34855, 'source_hover_iso'),
           (37386, 'source_hover_focal'))


def _read_details(path, page, fingerprint):
    """Read headers only; do not return metadata for a replaced source."""
    from PIL import Image
    from core.image_loader import is_raw_source, _read_raw_exif_bytes
    from utils.image_utils import read_image_exif

    result = {'fields': []}
    try:
        stat = os.stat(path)
        stamp = (stat.st_size, stat.st_mtime_ns)
        if fingerprint is not None and tuple(fingerprint) != stamp:
            return result
        if page is not None and page < 0:
            import tifffile
            from core.image_loader import tiff_source_pages
            with tifffile.TiffFile(path) as tif:
                page = next(index for ordinal, (index, _) in enumerate(tiff_source_pages(tif), 1)
                            if ordinal == -page)
        if is_raw_source(path):
            raw = _read_raw_exif_bytes(path)
            exif = Image.Exif()
            if raw:
                exif.load(raw)
        else:
            with Image.open(path) as image:
                if page is not None and image.format == 'TIFF':
                    image.seek(page)
                width, height = image.size
                exif = read_image_exif(image)
                orientation = exif.get(274, 1)
                if orientation in (5, 6, 7, 8):
                    width, height = height, width
                result['size'] = (width, height)
                # Resolve the nested IFD before closing the image stream.
                nested = dict(exif.get_ifd(0x8769)) if 0x8769 in exif else {}
                values = {tag: exif.get(tag, nested.get(tag)) for tag, _ in _FIELDS}
                exif = values
        if not isinstance(exif, dict):
            nested = exif.get_ifd(0x8769) if 0x8769 in exif else {}
            exif = {tag: exif.get(tag, nested.get(tag)) for tag, _ in _FIELDS}
        for tag, key in _FIELDS:
            value = exif.get(tag)
            if value is None or isinstance(value, (bytes, bytearray, tuple, list, dict)):
                continue
            text = str(value).strip()
            try:
                if tag == 33434:
                    from fractions import Fraction
                    text = f'{Fraction(float(value)).limit_denominator(100000)} s'
                elif tag == 33437:
                    text = f'f/{float(value):g}'
                elif tag == 37386:
                    text = f'{float(value):g} mm'
            except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                pass
            if text and len(text) <= 160 and all(c.isprintable() for c in text):
                result['fields'].append((key, text))
        stat = os.stat(path)
        if (stat.st_size, stat.st_mtime_ns) != stamp:
            return {'fields': []}
    except Exception:  # Missing, replaced, malformed or unsupported sources.
        return {'fields': []}
    return result


class _Signals(QObject):
    done = pyqtSignal(object, object)


class _Read(QRunnable):
    def __init__(self, request, signals):
        super().__init__()
        self.request = request
        self.signals = signals

    def run(self):
        path, page, fingerprint = self.request[3:6]
        details = _read_details(path, page, fingerprint)
        self.signals.done.emit(self.request, details)


class SourceHover(QObject):
    def __init__(self, manager, view):
        super().__init__(view)
        self.manager = manager
        self.view = view
        self.generation = 0
        self.serial = 0
        self.running = False
        self.pending = None
        self.signals = _Signals()
        self.signals.done.connect(self._finished)
        view.viewport().installEventFilter(self)

    def invalidate(self):
        self.generation += 1
        self.serial += 1
        self.pending = None
        QToolTip.hideText()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ToolTip:
            item = self.view.itemAt(event.pos())
            if item is None:
                self.serial += 1
                self.pending = None
                QToolTip.hideText()
                return True
            self.show_row(self.view.row(item), event.globalPos())
            return True
        if event.type() in (QEvent.Type.Leave, QEvent.Type.MouseMove):
            self.serial += 1
            self.pending = None
        return False

    def show_row(self, row, position):
        window = self.manager.window
        frames = getattr(window, 'raw_images', ())
        names = getattr(window, 'image_filenames', ())
        if row < 0 or row >= len(frames) or row >= len(names):
            return
        frame = frames[row]
        paths = getattr(window, 'image_source_paths', ()) or ()
        path = getattr(frame, 'path', None) or (paths[row] if row < len(paths) else '')
        name = str(names[row])
        page = getattr(frame, 'page', None)
        if page is None and path and str(path).lower().endswith(('.tif', '.tiff')):
            match = re.search(r'_page_(\d+)\.tiff?$', name, re.IGNORECASE)
            if match:
                page = -int(match.group(1))
        native = getattr(frame, 'native_shape', ())
        shape = getattr(frame, 'shape', ())
        self.serial += 1
        request = (self.generation, self.serial, row, path, page,
                   getattr(frame, 'fingerprint', None), name, shape, native, position, frame)
        QToolTip.showText(position, self._format(request, {}), self.view)
        if not path:
            return
        if self.running:
            self.pending = request
        else:
            self._start(request)

    def _start(self, request):
        self.running = True
        QThreadPool.globalInstance().start(_Read(request, self.signals))

    def _finished(self, request, details):
        self.running = False
        generation, serial, row = request[:3]
        frames = getattr(self.manager.window, 'raw_images', ())
        position = self.view.viewport().mapFromGlobal(QCursor.pos())
        item = self.view.itemAt(position)
        if (generation == self.generation and serial == self.serial
                and row < len(frames) and frames[row] is request[10]
                and item is not None and self.view.row(item) == row):
            QToolTip.showText(request[9], self._format(request, details), self.view)
        pending, self.pending = self.pending, None
        if pending is not None and pending[0] == self.generation and pending[1] == self.serial:
            self._start(pending)

    @staticmethod
    def _format(request, details):
        _, _, _, path, page, _, name, shape, native, _, _ = request
        lines = [html.escape(name)]
        if path:
            lines.append(html.escape(str(path)))
        if page is not None and str(path).lower().endswith(('.tif', '.tiff')):
            page_number = -page if page < 0 else page + 1
            lines.append(f"{trans.t('source_hover_page')}: {page_number}")
        if len(shape) >= 2:
            lines.append(f"{trans.t('source_hover_working')}: {shape[1]} × {shape[0]}")
        size = (native[1], native[0]) if len(native) >= 2 else details.get('size')
        if size:
            lines.append(f"{trans.t('source_hover_original')}: {size[0]} × {size[1]}")
        lines.extend(f"{trans.t(key)}: {html.escape(value)}" for key, value in details.get('fields', ()))
        return '<br>'.join(lines)

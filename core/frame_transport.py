"""Transfer lazy sources and mapped pixels without pickling image buffers."""
from dataclasses import fields, is_dataclass
from functools import partial
from importlib import import_module
from types import FunctionType

import numpy as np

from core import frame_storage as storage
from core.source_frames import PreviewPixels


def _locate(module, name):
    value = import_module(module)
    for part in name.split('.'):
        value = getattr(value, part)
    return value


def _reference(value):
    if '<locals>' in value.__qualname__ or '<lambda>' in value.__qualname__:
        raise TypeError(f'Cannot transfer local render callable: {value.__qualname__}')
    return value.__module__, value.__qualname__


class FrameEncoder:
    """Keep borrowed/created file owners alive until a transfer is acknowledged."""

    def __init__(self, should_cancel=None):
        self.should_cancel = should_cancel
        self.paths = set()
        self._memo = {}
        self._keep = []

    @property
    def owned_paths(self):
        return {path for path in self.paths if storage.owns_temp_file(path)}

    def _field(self, name, value):
        if callable(value):
            if name in ('should_cancel', 'cancel_callback'):
                return ('cancel',)
            if name == 'is_current':
                return ('current',)
        return self.encode(value)

    def encode(self, value):
        if self.should_cancel is not None and self.should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        identity = id(value)
        if identity in self._memo:
            return ('ref', self._memo[identity])
        index = len(self._memo)
        self._memo[identity] = index
        self._keep.append(value)
        if isinstance(value, np.ndarray):
            description = storage.describe_frame(value)
            if description is None:
                mapped = storage.store_frame(value, prefer_ram=False)
                self._keep.append(mapped)
                description = storage.describe_frame(mapped)
            self.paths.add(description['path'])
            kind, data = 'frame', description
        elif isinstance(value, PreviewPixels):
            self.paths.add(value.path)
            kind, data = 'preview', (value.path, value.shape, value.dtype.str)
        elif isinstance(value, np.dtype):
            kind, data = 'dtype', value.str
        elif isinstance(value, partial):
            kind, data = 'partial', (
                self.encode(value.func), self.encode(value.args),
                [(key, self._field(key, item)) for key, item in (value.keywords or {}).items()])
        elif isinstance(value, FunctionType):
            kind, data = 'function', _reference(value)
        elif is_dataclass(value) and not isinstance(value, type):
            kind, data = 'record', (
                _reference(type(value)),
                [(field.name, self._field(field.name, getattr(value, field.name)))
                 for field in fields(value) if field.init])
        elif isinstance(value, dict):
            kind, data = 'dict', [(self.encode(key), self._field(key, item))
                                  for key, item in value.items()]
        elif isinstance(value, (list, tuple)):
            kind, data = ('tuple' if isinstance(value, tuple) else 'list'), [
                self.encode(item) for item in value]
        else:
            from PyQt6.QtCore import QRect, QRectF
            if isinstance(value, (QRect, QRectF)):
                kind = 'rect' if isinstance(value, QRect) else 'rectf'
                data = (value.x(), value.y(), value.width(), value.height())
            else:
                raise TypeError(f'Cannot transfer render value: {type(value).__qualname__}')
        return ('value', index, kind, data)


class FrameDecoder:
    """Adopt only sender-owned files; borrow original source and cache mappings."""

    def __init__(self, *, owned_paths=(), owned_directories=(), should_cancel=None):
        self.owned_paths = set(owned_paths)
        self.owned_directories = set(owned_directories)
        self.adopted_paths = set()
        self.should_cancel = should_cancel
        self._memo = {}

    def _owns(self, path):
        if path in self.owned_paths:
            self.adopted_paths.add(path)
            return True
        return storage.owns_temp_file(path)

    def _directory(self, path):
        import os
        directory = os.path.dirname(path)
        return directory if directory in self.owned_directories else None

    def decode(self, value):
        if not isinstance(value, tuple):
            return value
        tag = value[0]
        if tag == 'cancel':
            return self.should_cancel
        if tag == 'current':
            return lambda: self.should_cancel is None or not self.should_cancel()
        if tag == 'ref':
            return self._memo[value[1]]
        _, index, kind, data = value
        if kind == 'frame':
            description = dict(data, cleanup_directory=self._directory(data['path']))
            result = storage.open_frame(description, owned=self._owns(data['path']))
        elif kind == 'preview':
            path, shape, dtype = data
            result = PreviewPixels.__new__(PreviewPixels)
            result.path, result.shape, result.dtype = path, tuple(shape), np.dtype(dtype)
            if self._owns(path):
                storage.track_temp_file(result, path, cleanup_directory=self._directory(path))
        elif kind == 'dtype':
            result = np.dtype(data)
        elif kind == 'partial':
            function, args, kwargs = data
            result = partial(self.decode(function), *self.decode(args),
                             **{key: self.decode(item) for key, item in kwargs})
        elif kind == 'function':
            result = _locate(*data)
        elif kind == 'record':
            reference, values = data
            result = _locate(*reference)(**{key: self.decode(item) for key, item in values})
        elif kind == 'dict':
            result = {}
            self._memo[index] = result
            result.update((self.decode(key), self.decode(item)) for key, item in data)
        elif kind in ('list', 'tuple'):
            result = []
            self._memo[index] = result
            result.extend(self.decode(item) for item in data)
            if kind == 'tuple':
                result = tuple(result)
        elif kind in ('rect', 'rectf'):
            from PyQt6.QtCore import QRect, QRectF
            result = (QRect if kind == 'rect' else QRectF)(*data)
        else:
            raise ValueError(f'Unknown render transfer type: {kind}')
        self._memo[index] = result
        return result

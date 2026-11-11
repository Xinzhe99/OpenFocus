"""Collect lazy utilities and the shared frozen runtime resources."""
import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from importlib.metadata import packages_distributions
from utils import _EXPORTS

_root = Path(__file__).resolve().parents[1]
_manifest = json.loads((_root / 'packaging/runtime_resources.json').read_text(encoding='utf-8'))
datas = []
for _pattern in _manifest['include']:
    _files = sorted(path for path in _root.glob(_pattern) if path.is_file())
    if not _files:
        raise RuntimeError(f'Runtime resource pattern has no files: {_pattern}')
    datas.extend((str(path), str(path.relative_to(_root).parent)) for path in _files)

# Native extensions and provider libraries are needed even when their Python
# imports are discovered through a lazy algorithm entry point.
binaries = collect_dynamic_libs('rawpy') + collect_dynamic_libs('onnxruntime')
for _package in ('rawpy', 'onnxruntime'):
    for _source, _destination in collect_data_files(_package):
        _parts = {part.lower() for part in Path(_destination).parts}
        _name = Path(_source).name.lower()
        if (_parts & {'test', 'tests', 'testing', 'tools'} and
                not _name.startswith(('license', 'licence', 'notice', 'copying'))):
            continue
        datas.append((_source, _destination))
_distributions = packages_distributions()
for _package in ('rawpy', 'onnxruntime'):
    for _distribution in _distributions.get(_package, []):
        datas += copy_metadata(_distribution)

hiddenimports = sorted(set(_EXPORTS.values()) | {
    'utils.layered_tiff', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'PyQt6.QtNetwork', 'PyQt6.QtSvg', 'rawpy', 'onnxruntime',
})

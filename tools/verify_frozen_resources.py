"""Verify a PyInstaller resource root against the source runtime manifest.

Pass the directory corresponding to sys._MEIPASS, such as
``dist/OpenFocus/_internal`` on Windows or the app's resource directory on macOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('packaging/runtime_resources.json')


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_resources(bundle: Path, source: Path = ROOT) -> list[str]:
    """Return every missing, changed or excluded resource in a frozen bundle."""
    manifest_path = source / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    resources = set()
    for pattern in manifest['include']:
        matches = {path.relative_to(source) for path in source.glob(pattern)
                   if path.is_file()}
        if not matches:
            errors.append(f'no source files match {pattern}')
        resources.update(matches)
    for relative in sorted(resources):
        target = bundle / relative
        if not target.is_file():
            errors.append(f'missing {relative.as_posix()}')
        elif _digest(source / relative) != _digest(target):
            errors.append(f'hash mismatch {relative.as_posix()}')

    exceptions = set(manifest.get('exclude_exceptions', []))
    for pattern in manifest['exclude']:
        for path in bundle.glob(pattern):
            if path.is_file() and path.relative_to(bundle).as_posix() not in exceptions:
                errors.append(f'excluded path present: {path.relative_to(bundle).as_posix()}')
            elif path.is_dir() and pattern.endswith('/**'):
                for child in path.rglob('*'):
                    if child.is_file() and child.relative_to(bundle).as_posix() not in exceptions:
                        errors.append(f'excluded path present: {child.relative_to(bundle).as_posix()}')
    return sorted(set(errors))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='frozen resource root (sys._MEIPASS)')
    args = parser.parse_args(argv)
    errors = verify_resources(args.bundle)
    if errors:
        for error in errors:
            print(error)
        return 1
    print('Frozen runtime resources match the source manifest.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

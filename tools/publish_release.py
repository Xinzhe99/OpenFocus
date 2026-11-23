"""Publish a release only after validating its draft and immutable identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

try:
    from .release_archives import (DELTA_PLATFORMS, GITHUB_ASSET_LIMIT, delta_asset_identity,
                                   delta_version, expected_name, linux_required, list_downloads,
                                   load_manifest, reassemble, sha256_file, verify_stage)
    from .release_metadata import resolve_identity
except ImportError:
    from release_archives import (DELTA_PLATFORMS, GITHUB_ASSET_LIMIT, delta_asset_identity,
                                  delta_version, expected_name, linux_required, list_downloads,
                                  load_manifest, reassemble, sha256_file, verify_stage)
    from release_metadata import resolve_identity


def command(*args, check=True):
    return subprocess.run(args, encoding='utf-8', stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def api(path):
    return json.loads(command('gh', 'api', path).stdout)



def release_pages(repo):
    return json.loads(command('gh', 'api', '--paginate', '--slurp',
                              f'repos/{repo}/releases?per_page=100').stdout)


def release_assets(repo, release_id):
    pages = json.loads(command('gh', 'api', '--paginate', '--slurp',
                               f'repos/{repo}/releases/{release_id}/assets?per_page=100').stdout)
    return [asset for page in pages for asset in page]

def tag_commit(repo, tag):
    obj = api(f'repos/{repo}/git/ref/tags/{tag}')['object']
    for _ in range(8):
        if obj['type'] == 'commit':
            return obj['sha']
        if obj['type'] != 'tag':
            raise ValueError('tag does not resolve to a commit')
        obj = api(f"repos/{repo}/git/tags/{obj['sha']}")['object']
    raise ValueError('tag nesting limit exceeded')


def get_release(repo, tag):
    matches = [release for page in release_pages(repo) for release in page
               if release.get('tag_name') == tag]
    if len(matches) > 1:
        raise ValueError('multiple releases have the same tag')
    if not matches:
        return None
    release = matches[0]
    release['assets'] = release_assets(repo, release['id'])
    return release


def record(path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'missing or empty asset: {path.name}')
    return {'name': path.name, 'size': path.stat().st_size, 'sha256': sha256_file(path)}


def _validate_delta_assets(directory, tag, full_sizes):
    delta_paths = sorted(directory.glob('*.ofdelta.zip'))
    if not delta_paths:
        return []
    from utils.release_delta import PATCH_REQUIRED_FROM, legacy_patch_name, read_manifest

    legacy_names = []
    for path in delta_paths:
        base, platform = delta_asset_identity(path.name, tag)
        if not 0 < path.stat().st_size < min(full_sizes.get(platform, 0), GITHUB_ASSET_LIMIT):
            raise ValueError(f'delta is not smaller than full asset or exceeds limit: {path.name}')
        manifest = read_manifest(path)
        if (manifest['from'], manifest['to'], manifest['platform']) != (base, tag, platform):
            raise ValueError(f'delta manifest identity differs: {path.name}')
        if base != PATCH_REQUIRED_FROM:
            continue
        legacy_name = legacy_patch_name(tag, platform)
        legacy = directory / legacy_name
        if not legacy.exists():
            continue
        if not legacy.is_file() or not 0 < legacy.stat().st_size < min(full_sizes[platform], GITHUB_ASSET_LIMIT):
            raise ValueError(f'empty or oversized legacy patch: {legacy_name}')
        helper = 'OpenFocus-Patch.exe' if platform == 'windows-x64' else 'OpenFocus-Patch'
        with zipfile.ZipFile(legacy) as archive:
            if sorted(archive.namelist()) != sorted([helper, path.name, 'README.txt']):
                raise ValueError(f'legacy patch archive contents differ: {legacy_name}')
            metadata = archive.getinfo(helper)
            if (metadata.file_size <= 0 or archive.getinfo('README.txt').file_size <= 0 or
                    (platform != 'windows-x64' and not (metadata.external_attr >> 16) & 0o111) or
                    archive.getinfo(path.name).file_size != path.stat().st_size):
                raise ValueError(f'legacy patch archive entries differ: {legacy_name}')
            with archive.open(path.name) as embedded:
                digest = hashlib.sha256()
                for chunk in iter(lambda: embedded.read(1024 * 1024), b''):
                    digest.update(chunk)
                if digest.hexdigest() != sha256_file(path):
                    raise ValueError(f'legacy delta differs from standalone patch: {legacy_name}')
        legacy_names.append(legacy_name)
    return [path.name for path in delta_paths] + legacy_names


def local_assets(directory, tag):
    manifest = directory / f'OpenFocus-{tag}-windows-assets.json'
    data = load_manifest(manifest)
    if data['tag'] != tag:
        raise ValueError('Windows manifest tag mismatch')
    verify_stage(manifest, directory)
    names = list_downloads(manifest) + [manifest.name, 'join-openfocus-parts.ps1',
        f'OpenFocus-{tag}-macos.dmg', f'OpenFocus-{tag}-macos-arm64.zip']
    linux = expected_name(tag, 'linux-desktop')
    if linux_required(tag) or (directory / linux).exists():
        names.append(linux)
    full_sizes = {'windows-x64': data['assets']['windows-zip']['size']}
    for platform in ('macos-arm64', 'linux-x86_64'):
        path = directory / expected_name(tag, DELTA_PLATFORMS[platform])
        if path.is_file():
            full_sizes[platform] = path.stat().st_size
    names.extend(_validate_delta_assets(directory, tag, full_sizes))
    if len(names) != len(set(names)):
        raise ValueError('duplicate expected asset names')
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != set(names):
        raise ValueError(f'unexpected or missing staged assets: {actual ^ set(names)}')
    return [record(directory / name) for name in sorted(names)]


def remote_digest(repo, tag, asset, temp):
    digest = asset.get('digest')
    if digest:
        if not digest.startswith('sha256:') or len(digest) != 71:
            raise ValueError(f"unsupported digest for {asset['name']}")
        return digest[7:].lower()
    command('gh', 'release', 'download', tag, '--repo', repo, '--pattern',
            asset['name'], '--dir', str(temp), '--clobber')
    downloaded = temp / asset['name']
    if not downloaded.is_file():
        raise ValueError(f"could not download {asset['name']} for verification")
    return sha256_file(downloaded)


def verify_remote(repo, tag, release, records):
    assets = release.get('assets', [])
    by_name = {item['name']: item for item in assets}
    expected = {item['name']: item for item in records}
    if len(by_name) != len(assets) or set(by_name) != set(expected):
        raise ValueError('remote asset set differs from expected release')
    with tempfile.TemporaryDirectory() as temp:
        for name, item in expected.items():
            remote = by_name[name]
            if remote.get('size') != item['size'] or remote_digest(repo, tag, remote, Path(temp)) != item['sha256']:
                raise ValueError(f'remote asset mismatch: {name}')


def download_identity(repo, tag, directory):
    name = f'OpenFocus-{tag}-build.json'
    command('gh', 'release', 'download', tag, '--repo', repo,
            '--pattern', name, '--dir', str(directory), '--clobber')
    return json.loads((directory / name).read_text(encoding='utf-8'))


def select_delta_bases(releases, label, legacy_base):
    if '-preview.' in label:
        return [legacy_base]
    target = delta_version(label)
    if target is None:
        raise ValueError(f'invalid delta target release: {label}')
    stable, beta = [], []
    for release in releases:
        tag = release.get('tag_name', '')
        version = delta_version(tag)
        if version is None or version >= target or release.get('draft') is not False:
            continue
        if release.get('prerelease') is False and version[3]:
            stable.append((version, tag))
        elif release.get('prerelease') is True and not version[3] and version[:3] == target[:3]:
            beta.append((version, tag))
    return [max(group)[1] for group in (stable, beta) if group]


def _download_verified_base(repo, tag, asset, directory):
    name = asset['name']
    size, digest = asset.get('size'), asset.get('digest')
    if type(size) is not int or size <= 0 or not isinstance(digest, str) or not digest.startswith('sha256:'):
        raise ValueError(f'published base asset lacks a trusted size or SHA-256: {name}')
    command('gh', 'release', 'download', tag, '--repo', repo, '--pattern', name, '--dir', str(directory))
    path = directory / name
    if path.stat().st_size != size or digest != 'sha256:' + sha256_file(path):
        raise ValueError(f'published base asset differs from GitHub size or SHA-256: {name}')
    return path


def _base_archive(repo, tag, platform, directory):
    release = api(f'repos/{repo}/releases/tags/{tag}')
    if (release.get('tag_name') != tag or release.get('draft') is not False or
            release.get('prerelease') is not ('-beta.' in tag)):
        raise ValueError(f'base release is not published in its expected channel: {tag}')
    records = release_assets(repo, release['id'])
    assets = {item['name']: item for item in records}
    if len(assets) != len(records):
        raise ValueError(f'duplicate names in base release: {tag}')
    key = DELTA_PLATFORMS[platform]
    name = expected_name(tag, key)
    manifest_name = f'OpenFocus-{tag}-windows-assets.json'
    part_names = {asset_name for asset_name in assets if asset_name.startswith(name + '.part')}
    if platform == 'windows-x64' and manifest_name in assets:
        manifest_path = _download_verified_base(repo, tag, assets[manifest_name], directory)
        manifest = load_manifest(manifest_path)
        if manifest['tag'] != tag:
            raise ValueError(f'base Windows manifest tag differs: {tag}')
        info = manifest['assets'][key]
        parts = info['parts']
        present = part_names | ({name} if name in assets else set())
        if present != {item['name'] for item in parts or [info]}:
            raise ValueError(f'published base Windows layout differs from manifest: {tag}')
        for item in parts or [info]:
            asset = assets[item['name']]
            if asset.get('size') != item['size'] or asset.get('digest') != 'sha256:' + item['sha256']:
                raise ValueError(f'published base Windows digest differs from manifest: {item["name"]}')
            _download_verified_base(repo, tag, asset, directory)
        if parts:
            joined = reassemble(manifest_path, directory, directory / 'joined', keys=(key,))[0]
            for item in parts:
                (directory / item['name']).unlink()
            return joined
        return directory / name
    if part_names:
        raise ValueError(f'published split assets have no manifest: {tag}')
    if name not in assets:
        if platform != 'linux-x86_64' or linux_required(tag):
            raise ValueError(f'published base release is missing {name}')
        return None
    return _download_verified_base(repo, tag, assets[name], directory)


def _smoke_delta(tree, label, platform):
    executable = {'windows-x64': 'OpenFocus.exe', 'macos-arm64': 'Contents/MacOS/OpenFocus',
                  'linux-x86_64': 'OpenFocus'}[platform]
    smoke = [sys.executable, 'tools/frozen_cli_smoke.py', '--executable', str(tree / executable)]
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    if platform == 'windows-x64':
        smoke += ['--resources', str(tree / f'_openfocus_runtime_{label}')]
    elif platform == 'linux-x86_64':
        env['QT_QPA_PLATFORM'] = 'xcb'
        smoke = ['xvfb-run', '-a', *smoke, '--resources', str(tree / '_internal')]
    subprocess.run(smoke, env=env, check=True)


def build_release_deltas(repo, label, platform, output):
    from utils.release_delta import PATCH_REQUIRED_FROM, delta_name
    from utils.self_update import extract_linux_tar, extract_zip

    target = Path(expected_name(label, DELTA_PLATFORMS[platform]))
    if not target.is_file():
        raise ValueError(f'missing final archive: {target}')
    releases = (release for page in release_pages(repo) for release in page) if '-preview.' not in label else ()
    bases = select_delta_bases(releases, label, PATCH_REQUIRED_FROM)
    output.mkdir(parents=True, exist_ok=True)
    generator = [sys.executable, 'tools/build_update_delta.py']
    launched = False
    for base in bases:
        with tempfile.TemporaryDirectory(prefix='openfocus-delta-', dir=os.getenv('RUNNER_TEMP')) as temp:
            work = Path(temp)
            archive = _base_archive(repo, base, platform, work)
            if archive is None:
                print(f'{platform}: no published portable archive for {base}', flush=True)
                continue
            legacy = base == PATCH_REQUIRED_FROM
            tree_name = 'OpenFocus.app' if platform == 'macos-arm64' else 'OpenFocus'
            tree = work / tree_name
            patch = output / delta_name(label, base, platform)
            args = [*generator, 'build', '--base', str(archive), '--target', str(target),
                    '--from-tag', base, '--to-tag', label, '--platform', platform, '--output', str(output)]
            if not legacy and not launched:
                args += ['--verify-tree', str(tree)]
            subprocess.run(args, check=True)
            if not patch.is_file():
                print(f'{platform}: no smaller publishable delta from {base}', flush=True)
                continue
            if not legacy:
                if not launched:
                    _smoke_delta(tree, label, platform)
                    launched = True
                continue
            digest = work / 'OpenFocus-Patch.sha256'
            digest.write_text(sha256_file(patch) + '\n', encoding='ascii')
            subprocess.run([sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', '--onefile',
                            '--name', 'OpenFocus-Patch', '--paths', '.', '--distpath', str(work / 'helper'),
                            '--workpath', str(work / 'build'), '--specpath', str(work),
                            '--add-data', f'{digest}:.', 'tools/apply_release_patch.py'], check=True)
            helper = work / 'helper' / ('OpenFocus-Patch.exe' if platform == 'windows-x64' else 'OpenFocus-Patch')
            installation = work / 'installation'
            if platform == 'linux-x86_64':
                extract_linux_tar(archive, installation)
            else:
                extract_zip(archive, installation)
            tree = installation / tree_name
            archive.unlink()
            scratch = work / 'scratch'
            scratch.mkdir()
            env = dict(os.environ, TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
            subprocess.run([str(helper), '--install-dir', str(tree), '--patch', str(patch), '--no-restart'],
                           env=env, check=True)
            if platform == 'macos-arm64':
                subprocess.run(['codesign', '--verify', '--deep', '--strict', str(tree)], check=True)
            _smoke_delta(tree, label, platform)
            launched = True
            subprocess.run([*generator, 'bridge', '--patch', str(patch), '--helper', str(helper),
                            '--output', str(output), '--limit', str(target.stat().st_size)], check=True)


def publish(directory, notes, repo, tag, version, sha, run_id, attempt):
    channel = resolve_identity('refs/tags/' + tag, tag, '', version)['channel']
    prerelease = channel == 'beta'
    prerelease_flag = '--prerelease=true' if prerelease else '--prerelease=false'
    records = local_assets(directory, tag)
    identity_name = f'OpenFocus-{tag}-build.json'
    if tag_commit(repo, tag) != sha:
        raise ValueError('release tag moved from the build commit')
    release = get_release(repo, tag)
    if release and not release['draft']:
        if release.get('prerelease') is not prerelease or release.get('tag_name') != tag:
            raise ValueError('existing release has the wrong prerelease status')
        with tempfile.TemporaryDirectory() as temp:
            identity = download_identity(repo, tag, Path(temp))
            if (identity.get('schema_version') != 1 or identity.get('tag') != tag or
                    identity.get('version') != version or identity.get('sha') != sha or
                    not isinstance(identity.get('run_id'), int) or identity['run_id'] <= 0 or
                    not isinstance(identity.get('run_attempt'), int) or identity['run_attempt'] <= 0 or
                    identity.get('assets') != records):
                raise ValueError('published release identity differs; use a new version tag')
            verify_remote(repo, tag, release, records + [record(Path(temp) / identity_name)])
        return
    identity = {'schema_version': 1, 'tag': tag, 'version': version, 'sha': sha,
                'run_id': int(run_id), 'run_attempt': int(attempt), 'assets': records}
    identity_path = directory / identity_name
    identity_path.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    all_records = records + [record(identity_path)]
    if release is None:
        command('gh', 'release', 'create', tag, '--repo', repo, '--draft', '--verify-tag',
                *(['--prerelease', '--latest=false'] if prerelease else []),
                '--title', f'OpenFocus {tag}', '--notes-file', str(notes))
        for attempt in range(31):
            release = get_release(repo, tag)
            if release is not None:
                break
            if attempt < 30:
                time.sleep(2)
    if not release or not release['draft'] or release['tag_name'] != tag:
        raise ValueError('release is no longer the expected draft')
    # Only draft assets owned by this release may be removed or replaced.
    for asset in release.get('assets', []):
        name = asset['name']
        if not (name.startswith(f'OpenFocus-{tag}-') or name == 'join-openfocus-parts.ps1'):
            raise ValueError(f'unexpected asset in draft: {name}')
    for asset in release.get('assets', []):
        command('gh', 'release', 'delete-asset', tag, asset['name'], '--repo', repo, '-y')
    for item in all_records:
        command('gh', 'release', 'upload', tag, str(directory / item['name']), '--repo', repo)
    command('gh', 'release', 'edit', tag, '--repo', repo, '--title', f'OpenFocus {tag}',
            '--notes-file', str(notes), '--draft=true', prerelease_flag,
            *(['--latest=false'] if prerelease else []))
    release = get_release(repo, tag)
    if not release or not release['draft'] or release['tag_name'] != tag:
        raise ValueError('release changed before verification')
    if (release.get('prerelease') is not prerelease or release.get('name') != f'OpenFocus {tag}' or
            release.get('body') != notes.read_text(encoding='utf-8')):
        raise ValueError('draft title, notes or prerelease status did not match')
    verify_remote(repo, tag, release, all_records)
    if tag_commit(repo, tag) != sha:
        raise ValueError('release tag moved before publication')
    command('gh', 'release', 'edit', tag, '--repo', repo, '--draft=false', prerelease_flag,
            *(['--latest=false'] if prerelease else []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-deltas', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--platform', choices=DELTA_PLATFORMS)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    args = parser.parse_args()
    if args.build_deltas:
        if not args.label or not args.platform:
            parser.error('--build-deltas requires --label and --platform')
        build_release_deltas(os.environ['GITHUB_REPOSITORY'], args.label, args.platform, args.directory)
        return
    if args.notes is None:
        parser.error('publishing requires --notes')
    publish(args.directory, args.notes, os.environ['GITHUB_REPOSITORY'],
            os.environ['RELEASE_TAG'], os.environ['RELEASE_VERSION'], os.environ['RELEASE_SHA'],
            os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT'])


if __name__ == '__main__':
    main()

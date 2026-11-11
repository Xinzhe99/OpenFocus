"""Publish a release only after validating its draft and immutable identity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

try:
    from .release_archives import list_downloads, load_manifest, sha256_file, verify_stage
    from .release_metadata import resolve_identity
except ImportError:
    from release_archives import list_downloads, load_manifest, sha256_file, verify_stage
    from release_metadata import resolve_identity


def command(*args, check=True):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def api(path):
    return json.loads(command('gh', 'api', path).stdout)


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
    pages = json.loads(command('gh', 'api', '--paginate', '--slurp',
                               f'repos/{repo}/releases?per_page=100').stdout)
    matches = [release for page in pages for release in page
               if release.get('tag_name') == tag]
    if len(matches) > 1:
        raise ValueError('multiple releases have the same tag')
    if not matches:
        return None
    release = matches[0]
    pages = json.loads(command('gh', 'api', '--paginate', '--slurp',
        f"repos/{repo}/releases/{release['id']}/assets?per_page=100").stdout)
    release['assets'] = [asset for page in pages for asset in page]
    return release


def record(path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'missing or empty asset: {path.name}')
    return {'name': path.name, 'size': path.stat().st_size, 'sha256': sha256_file(path)}


def local_assets(directory, tag):
    manifest = directory / f'OpenFocus-{tag}-windows-assets.json'
    data = load_manifest(manifest)
    if data['tag'] != tag:
        raise ValueError('Windows manifest tag mismatch')
    verify_stage(manifest, directory)
    names = list_downloads(manifest) + [manifest.name, 'join-openfocus-parts.ps1',
        f'OpenFocus-{tag}-macos.dmg', f'OpenFocus-{tag}-macos-arm64.zip']
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
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    args = parser.parse_args()
    publish(args.directory, args.notes, os.environ['GITHUB_REPOSITORY'],
            os.environ['RELEASE_TAG'], os.environ['RELEASE_VERSION'], os.environ['RELEASE_SHA'],
            os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT'])


if __name__ == '__main__':
    main()

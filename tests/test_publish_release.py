"""Release publication is gated on immutable identity and verified draft assets."""
import json
from types import SimpleNamespace

import pytest

from tools import publish_release as publisher


class FakeGitHub:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.release = None
        self.content = {}
        self.calls = []
        self.commands = []
        self.sha = 'a' * 40
        self.fail_upload = None
        self.move_after_verify = False

    def command(self, *args, **kwargs):
        op = args[2]
        self.calls.append(op)
        self.commands.append(args)
        if op == 'create':
            self.release = {'draft': True, 'prerelease': '--prerelease' in args,
                            'tag_name': args[3], 'assets': []}
        elif op == 'upload':
            path = publisher.Path(args[4])
            if path.name == self.fail_upload:
                raise RuntimeError('upload failed')
            self.content[path.name] = path.read_bytes()
            self.refresh()
        elif op == 'delete-asset':
            del self.content[args[4]]
            self.refresh()
        elif op == 'download':
            name = args[args.index('--pattern') + 1]
            dest = args[args.index('--dir') + 1]
            (publisher.Path(dest) / name).write_bytes(self.content[name])
        elif op == 'edit' and '--draft=false' in args:
            self.release['prerelease'] = '--prerelease=true' in args
            self.release['draft'] = False
        elif op == 'edit':
            self.release['prerelease'] = '--prerelease=true' in args
            self.release['name'] = args[args.index('--title') + 1]
            self.release['body'] = publisher.Path(args[args.index('--notes-file') + 1]).read_text(encoding='utf-8')
        return SimpleNamespace(stdout='', stderr='', returncode=0)

    def refresh(self):
        self.release['assets'] = [
            {'name': name, 'size': len(data), 'digest': None}
            for name, data in self.content.items()]


def setup(monkeypatch, tmp_path):
    directory = tmp_path / 'release'
    directory.mkdir()
    (directory / 'notes').write_text('Release notes\n', encoding='utf-8')
    names = ('OpenFocus-v1.0.0-one', 'OpenFocus-v1.0.0-two')
    for name in names:
        (directory / name).write_bytes(name.encode())
    records = [publisher.record(directory / name) for name in names]
    gh = FakeGitHub(tmp_path)
    monkeypatch.setattr(publisher, 'local_assets', lambda *_: records)
    monkeypatch.setattr(publisher, 'command', gh.command)
    monkeypatch.setattr(publisher, 'get_release', lambda *_: gh.release)
    monkeypatch.setattr(publisher, 'tag_commit', lambda *_: gh.sha)
    return directory, gh


def run(directory):
    publisher.publish(directory, directory / 'notes', 'owner/repo', 'v1.0.0',
                      '1.0.0', 'a' * 40, 7, 1)


def test_draft_verified_before_publish_and_public_retry_is_read_only(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    run(directory)
    assert gh.release['draft'] is False
    assert gh.calls[-1] == 'edit'
    assert len(gh.release['assets']) == 3
    gh.calls.clear()
    run(directory)
    assert set(gh.calls) == {'download'}


def test_new_draft_can_appear_after_creation(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    invisible = [True, True]

    def get_release(*_):
        if gh.release is not None and invisible:
            invisible.pop()
            return None
        return gh.release

    monkeypatch.setattr(publisher, 'get_release', get_release)
    monkeypatch.setattr(publisher.time, 'sleep', lambda _: None)
    run(directory)
    assert not invisible
    assert gh.release['draft'] is False


def test_beta_remains_prerelease_and_retry_is_read_only(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    tag = 'v1.0.0-beta.1'
    publisher.publish(directory, directory / 'notes', 'owner/repo', tag,
                      '1.0.0', 'a' * 40, 7, 1)
    assert gh.release['draft'] is False
    assert gh.release['prerelease'] is True
    publication_commands = [args for args in gh.commands if args[2] in ('create', 'edit')]
    assert all('--latest=false' in args for args in publication_commands)
    assert '--prerelease' in publication_commands[0]
    assert all('--prerelease=true' in args for args in publication_commands[1:])
    gh.calls.clear()
    publisher.publish(directory, directory / 'notes', 'owner/repo', tag,
                      '1.0.0', 'a' * 40, 7, 1)
    assert set(gh.calls) == {'download'}
    gh.release['prerelease'] = False
    with pytest.raises(ValueError, match='prerelease status'):
        publisher.publish(directory, directory / 'notes', 'owner/repo', tag,
                          '1.0.0', 'a' * 40, 7, 1)


def test_failed_upload_keeps_draft(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    gh.fail_upload = 'OpenFocus-v1.0.0-two'
    with pytest.raises(RuntimeError, match='upload failed'):
        run(directory)
    assert gh.release['draft'] is True
    assert 'edit' not in gh.calls
    gh.fail_upload = None
    run(directory)
    assert gh.release['draft'] is False
    assert len(gh.release['assets']) == 3


def test_published_changed_bytes_are_never_modified(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    run(directory)
    gh.content['OpenFocus-v1.0.0-one'] = b'bad'
    gh.refresh()
    gh.calls.clear()
    with pytest.raises(ValueError, match='remote asset mismatch'):
        run(directory)
    assert set(gh.calls) == {'download'}


def test_tag_move_before_release_rejects_publication(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    gh.sha = 'b' * 40
    with pytest.raises(ValueError, match='tag moved'):
        run(directory)
    assert gh.calls == []


def test_tag_move_after_draft_verification_blocks_publication(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    original = publisher.verify_remote
    def move_tag(*args):
        original(*args)
        gh.sha = 'b' * 40
    monkeypatch.setattr(publisher, 'verify_remote', move_tag)
    with pytest.raises(ValueError, match='tag moved'):
        run(directory)
    assert gh.release['draft'] is True


def test_public_release_without_identity_is_immutable(monkeypatch, tmp_path):
    directory, gh = setup(monkeypatch, tmp_path)
    gh.release = {'draft': False, 'prerelease': False,
                  'tag_name': 'v1.0.0', 'assets': []}
    with pytest.raises(KeyError):
        run(directory)
    assert gh.calls == ['download']


@pytest.mark.parametrize(('tag', 'required'), [
    ('v1.76.0', False), ('v1.77.0-beta.7', False), ('v1.77.0', True),
])
def test_release_inventory_includes_linux_and_preserves_legacy(tmp_path, tag, required):
    from tools.release_archives import stage_windows

    zip_file = tmp_path / f'OpenFocus-{tag}-windows-x64.zip'
    setup_file = tmp_path / f'OpenFocus-{tag}-setup.exe'
    recovery = tmp_path / 'join-openfocus-parts.ps1'
    for path in (zip_file, setup_file, recovery):
        path.write_bytes(b'asset')
    directory = tmp_path / 'release'
    stage_windows(tag, zip_file, setup_file, directory, recovery)
    for suffix in ('macos.dmg', 'macos-arm64.zip'):
        (directory / f'OpenFocus-{tag}-{suffix}').write_bytes(b'macos')

    if required:
        with pytest.raises(ValueError, match='missing staged assets'):
            publisher.local_assets(directory, tag)
    else:
        assert not any(item['name'].endswith('-linux-x86_64.tar.gz')
                       for item in publisher.local_assets(directory, tag))

    linux = directory / f'OpenFocus-{tag}-linux-x86_64.tar.gz'
    linux.write_bytes(b'linux desktop')
    assert publisher.record(linux) in publisher.local_assets(directory, tag)
    linux.write_bytes(b'')
    with pytest.raises(ValueError, match='missing or empty asset'):
        publisher.local_assets(directory, tag)


def test_direct_bases_exclude_same_newer_draft_and_unrelated_beta():
    def published(tag, prerelease=False, draft=False):
        return dict(tag_name=tag, prerelease=prerelease, draft=draft)

    releases = [published('v2.0.2'), published('v2.0.0'), published('v1.99.0'),
                published('v2.0.1', draft=True), published('v2.0.1-beta.2', True),
                published('v2.0.1-beta.3', True), published('v2.0.1-beta.10', True),
                published('v2.0.0-beta.20', True), published('v2.0.1-beta.11')]
    assert publisher.select_delta_bases(releases, 'v2.0.1-beta.3', 'v2.0.0') == [
        'v2.0.0', 'v2.0.1-beta.2']
    assert publisher.select_delta_bases(releases, 'v2.0.1', 'v2.0.0') == [
        'v2.0.0', 'v2.0.1-beta.10']
    assert publisher.select_delta_bases(releases, 'v2.0.0-preview.45', 'v2.0.0') == ['v2.0.0']


def test_downloaded_split_base_uses_github_digests_and_skips_setup(tmp_path, monkeypatch):
    from tools import release_archives

    tag = 'v2.0.0'
    remote = tmp_path / 'remote'
    remote.mkdir()
    portable = tmp_path / f'OpenFocus-{tag}-windows-x64.zip'
    setup = tmp_path / f'OpenFocus-{tag}-setup.exe'
    recovery = tmp_path / 'join-openfocus-parts.ps1'
    portable.write_bytes(b'windows split portable asset')
    setup.write_bytes(b'setup')
    recovery.write_bytes(b'recover')
    stage_windows = release_archives.stage_windows
    stage_windows(tag, portable, setup, remote, recovery, limit=20, part_size=15)
    real_reassemble = publisher.reassemble
    monkeypatch.setattr(publisher, 'load_manifest', lambda path: release_archives.load_manifest(path, limit=20))
    monkeypatch.setattr(publisher, 'reassemble',
                        lambda *args, **kwargs: real_reassemble(*args, limit=20, **kwargs))
    metadata = [{'name': path.name, 'size': path.stat().st_size,
                 'digest': 'sha256:' + publisher.sha256_file(path)} for path in sorted(remote.iterdir())]
    monkeypatch.setattr(publisher, 'api', lambda *_: {
        'id': 42, 'tag_name': tag, 'draft': False, 'prerelease': False})
    monkeypatch.setattr(publisher, 'release_assets', lambda *_: metadata)
    downloads = []

    def download(*args):
        name = args[args.index('--pattern') + 1]
        directory = publisher.Path(args[args.index('--dir') + 1])
        (directory / name).write_bytes((remote / name).read_bytes())
        downloads.append(name)

    monkeypatch.setattr(publisher, 'command', download)
    local = tmp_path / 'local'
    local.mkdir()
    assert publisher._base_archive('o/r', tag, 'windows-x64', local).read_bytes() == portable.read_bytes()
    assert set(downloads) == {f'{portable.name}.part001', f'{portable.name}.part002',
                              f'OpenFocus-{tag}-windows-assets.json'}
    (remote / f'{portable.name}.part001').write_bytes(b'X' * 15)
    with pytest.raises(ValueError, match='differs from GitHub'):
        publisher._base_archive('o/r', tag, 'windows-x64', local)


def test_publish_patch_identity_and_matching_legacy_bytes(tmp_path):
    import zipfile

    from tools.build_update_delta import build_bridge, build_delta
    from utils.release_delta import delta_name, legacy_patch_name

    base_tag, tag, platform = 'v2.0.0', 'v2.0.1', 'windows-x64'
    base, target, directory = (tmp_path / name for name in ('base', 'target', 'staged'))
    for path in (base, target, directory):
        path.mkdir()
    (base / 'OpenFocus.exe').write_bytes(b'base')
    (target / 'OpenFocus.exe').write_bytes(b'target')
    patch = directory / delta_name(tag, base_tag, platform)
    helper = tmp_path / 'OpenFocus-Patch.exe'
    helper.write_bytes(b'helper')
    full_sizes = {platform: 4096}
    build_delta(base, target, patch, base_tag=base_tag, target_tag='v2.0.2', platform=platform)
    with pytest.raises(ValueError, match='manifest identity differs'):
        publisher._validate_delta_assets(directory, tag, full_sizes)
    patch.unlink()
    build_delta(base, target, patch, base_tag=base_tag, target_tag=tag, platform=platform)
    assert publisher._validate_delta_assets(directory, tag, full_sizes) == [patch.name]
    build_bridge(patch, helper, directory)
    legacy = directory / legacy_patch_name(tag, platform)
    assert publisher._validate_delta_assets(directory, tag, full_sizes) == [patch.name, legacy.name]
    with zipfile.ZipFile(legacy, 'w') as archive:
        archive.writestr(helper.name, helper.read_bytes())
        archive.writestr('README.txt', b'Instructions')
        archive.writestr(patch.name, b'x' * patch.stat().st_size)
    with pytest.raises(ValueError, match='differs from standalone patch'):
        publisher._validate_delta_assets(directory, tag, full_sizes)

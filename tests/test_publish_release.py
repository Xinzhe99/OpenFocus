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

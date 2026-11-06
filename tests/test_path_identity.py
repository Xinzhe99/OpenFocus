"""Canonical path keys and file identities for CLI output preflight.

`same_path` keeps its original contract; `path_key` and `file_identity` are the
shared parts the CLI's destination plan uses instead of re-implementing them.
"""
import os

import pytest

from utils.path_identity import file_identity, path_key, same_path


def test_path_key_is_absolute_and_spelling_independent(tmp_path):
    target = tmp_path / "out" / "result.png"
    assert path_key(str(target)) == path_key(
        os.path.join(str(tmp_path), "out", ".", "result.png"))
    assert path_key("result.png") == path_key(os.path.abspath("result.png"))
    assert path_key("") == ""
    assert path_key(None) == ""


def test_file_identity_is_the_same_for_two_names_of_one_file(tmp_path):
    first = tmp_path / "a.png"
    first.write_bytes(b"pixels")
    alias = tmp_path / "b.png"
    try:
        os.link(first, alias)
    except OSError:
        pytest.skip("hard links are unavailable")
    assert file_identity(first) is not None
    assert file_identity(first) == file_identity(alias)
    assert path_key(first) != path_key(alias)      # two names ...
    assert same_path(first, alias)                 # ... one file


def test_file_identity_is_none_when_the_file_is_missing(tmp_path):
    assert file_identity(str(tmp_path / "missing.png")) is None
    assert file_identity(None) is None


def test_same_path_keeps_its_original_behaviour(tmp_path):
    existing = tmp_path / "x.png"
    existing.write_bytes(b"x")
    assert same_path(str(existing), str(existing))
    assert same_path(str(existing), str(tmp_path / "x.png"))
    assert same_path(tmp_path, str(tmp_path))      # path-like objects still work
    assert not same_path(str(existing), str(tmp_path / "y.png"))
    assert not same_path("", str(existing))
    assert not same_path(None, str(existing))

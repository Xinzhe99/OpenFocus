"""Tests for the update checker's version comparison and API parsing."""
import sys, types

# The updater module imports nothing Qt-specific, so plain import works
from utils.updater import parse_version, is_newer


class TestParseVersion:
    def test_v_prefix(self):
        assert parse_version("v1.10") == (1, 10)

    def test_plain(self):
        assert parse_version("1.11.2") == (1, 11, 2)

    def test_invalid(self):
        assert parse_version("latest") is None
        assert parse_version("") is None


class TestIsNewer:
    def test_newer(self):
        assert is_newer("v1.11", "1.10")
        assert is_newer("1.11.1", "1.11")

    def test_same_or_older(self):
        assert not is_newer("v1.11", "1.11")
        assert not is_newer("v1.9", "1.10")

    def test_multipart_padding(self):
        assert is_newer("1.10.1", "1.10")
        assert not is_newer("1.10", "1.10.1")

    def test_garbage_is_never_newer(self):
        assert not is_newer("garbage", "1.10")

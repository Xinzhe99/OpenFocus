"""Unit tests for the label range parser and the update checker."""
import pytest

from utils.validators import LabelAdder


class TestLabelRange:
    f = staticmethod(LabelAdder._index_in_range)

    def test_all_variants(self):
        assert self.f("All", 0)
        assert self.f("All", 99)
        assert self.f("", 5)
        assert self.f(None, 5)
        assert self.f("all", 3)

    def test_range_inclusive_1_based(self):
        assert self.f("3-7", 2)   # frame 3
        assert self.f("3-7", 6)   # frame 7
        assert not self.f("3-7", 1)  # frame 2
        assert not self.f("3-7", 7)  # frame 8

    def test_reversed_range_normalized(self):
        assert self.f("7-3", 2)
        assert self.f("7-3", 6)
        assert not self.f("7-3", 1)

    def test_explicit_list(self):
        assert self.f("1,4,9", 0)
        assert self.f("1,4,9", 3)
        assert self.f("1,4,9", 8)
        assert not self.f("1,4,9", 1)

    def test_mixed_list_and_ranges(self):
        assert self.f("1-3,7", 0)
        assert self.f("1-3,7", 6)
        assert not self.f("1-3,7", 3)

    def test_unparseable_falls_back_to_all(self):
        assert self.f("abc", 5)
        assert self.f("x,y", 5)

    def test_single_number(self):
        assert self.f("2", 1)
        assert not self.f("2", 0)
        assert not self.f("2", 2)

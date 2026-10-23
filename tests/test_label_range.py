"""Unit tests for the label range parser.

An unparseable range used to mean "every frame", so a typo silently changed
which images carried a caption. Invalid input is now refused and reported;
only an empty string or "All" means the whole stack.
"""
import pytest

from utils.validators import LabelAdder, parse_label_range


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

    def test_unparseable_matches_nothing(self):
        assert not self.f("abc", 5)
        assert not self.f("x,y", 5)

    def test_unparseable_is_reported(self):
        parsed = parse_label_range("3-5,x")
        assert not parsed.valid
        assert parsed.error == "range_not_a_number"
        assert not parsed.matches(2)

    def test_zero_and_negative_are_refused(self):
        for spec in ("0", "-2", "1-0"):
            assert not parse_label_range(spec).valid, spec

    def test_blank_piece_is_refused(self):
        assert not parse_label_range("1,,3").valid

    def test_all_and_empty_mean_the_whole_stack(self):
        assert parse_label_range("All").all_frames
        assert parse_label_range("").all_frames
        assert parse_label_range("all").all_frames

    def test_single_number(self):
        assert self.f("2", 1)
        assert not self.f("2", 0)
        assert not self.f("2", 2)

"""Tests for steps.nw_chant_allocator — Sub-plan 4b: allocate_lines."""

from unittest.mock import patch

from steps.nw_chant_allocator import (
    Anchor,
    AllocationResult,
    ChantSpan,
    FlatTextData,
    MidWordBreak,
    _split_word_at_syl_boundary,
    allocate_lines,
)


def _flat(words, anchors=None, initial_pointer=0):
    from steps.nw_chant_allocator import ChantSpan
    return FlatTextData(
        words=words,
        anchors=anchors or [],
        chant_spans=[ChantSpan(1, 0, len(words))],
        initial_pointer=initial_pointer,
    )


class TestAllocateLinesEmpty:
    def test_empty_labels(self):
        result = allocate_lines(_flat([]), [], {})
        assert result.manifest == {}
        assert result.flags == []
        assert result.text_pointer_end == 0

    def test_empty_flat_text_all_labels_get_empty_string(self):
        flat = _flat([])
        result = allocate_lines(
            flat,
            ["line0", "line1"],
            {"line0": "alleluia", "line1": "dominus"},
        )
        assert result.manifest["line0"] == ""
        assert result.manifest["line1"] == ""


class TestAllocateLinesStubMode:
    def test_stub_advances_to_next_anchor(self):
        # Two within_chant_7 anchors; two labels with empty OCR → each
        # advances to next anchor.
        words = ["a", "b", "c", "d"]
        anchors = [Anchor(2, "within_chant_7"), Anchor(4, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat, ["line0", "line1"], {"line0": "", "line1": ""}
        )
        assert result.manifest["line0"] == "a b"
        assert result.manifest["line1"] == "c d"
        assert result.text_pointer_end == 4

    def test_stub_no_anchor_consumes_one_word(self):
        # Single line, single word, no anchors: max(1, 1//1) = 1 word.
        flat = _flat(["word"])
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert result.manifest["line0"] == "word"

    def test_stub_no_anchor_distributes_uniformly(self):
        # 3 lines, 9 words, no anchors: each line gets floor(9/3)=3 words.
        words = [f"w{i}" for i in range(9)]
        flat = _flat(words)
        result = allocate_lines(
            flat,
            ["line0", "line1", "line2"],
            {"line0": "", "line1": "", "line2": ""},
        )
        assert result.manifest["line0"] == "w0 w1 w2"
        assert result.manifest["line1"] == "w3 w4 w5"
        assert result.manifest["line2"] == "w6 w7 w8"
        assert result.text_pointer_end == 9

    def test_stub_no_anchor_last_line_gets_remainder(self):
        # 10 words, 3 lines, no anchors: floor(10/3)=3 per line;
        # the last line consumes all remaining words (4, not 3).
        words = [f"w{i}" for i in range(10)]
        flat = _flat(words)
        result = allocate_lines(
            flat,
            ["line0", "line1", "line2"],
            {"line0": "", "line1": "", "line2": ""},
        )
        assert result.manifest["line0"] == "w0 w1 w2"
        assert result.manifest["line1"] == "w3 w4 w5"
        assert result.manifest["line2"] == "w6 w7 w8 w9"
        assert result.text_pointer_end == 10

    def test_stub_missing_label_treated_as_empty_ocr(self):
        words = ["a", "b"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(flat, ["line0"], {})  # no entry in ocr_texts
        assert result.manifest["line0"] == "a b"

    def test_stub_returns_same_collection_structure(self):
        words = ["a", "b", "c", "d"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(flat, ["l0", "l1"], {"l0": "", "l1": ""})
        assert isinstance(result, AllocationResult)
        assert set(result.manifest.keys()) == {"l0", "l1"}


class TestAllocateLinesNWMode:
    def test_ocr_matches_flat_text_exactly(self):
        # OCR matches first 2 words exactly; best_k should be 2.
        words = ["alleluia", "dominus", "kyrie", "eleison"]
        flat = _flat(words)
        result = allocate_lines(flat, ["line0"], {"line0": "alleluia dominus"})
        assert result.manifest["line0"] == "alleluia dominus"

    def test_ocr_exact_match_single_word(self):
        words = ["alleluia", "dominus", "kyrie"]
        flat = _flat(words)
        result = allocate_lines(flat, ["line0"], {"line0": "alleluia"})
        assert result.manifest["line0"] == "alleluia"

    def test_search_window_limits_lookahead(self):
        # flat_text has 100 words; search_window=3 limits candidate to 3.
        words = [f"word{i}" for i in range(100)]
        flat = _flat(words)
        result = allocate_lines(
            flat, ["line0"], {"line0": "word0"}, search_window=3
        )
        # best_k is at most 3 due to the search window cap
        assert result.text_pointer_end <= 3

    def test_consecutive_lines_advance_pointer(self):
        words = ["a", "b", "c", "d", "e", "f"]
        anchors = [Anchor(3, "within_chant_7")]
        flat = _flat(words, anchors)
        # Each line gets its own slice; second line starts where first ended.
        result = allocate_lines(
            flat,
            ["line0", "line1"],
            {"line0": "a b c", "line1": "d e f"},
        )
        assert result.manifest["line0"] == "a b c"
        assert result.manifest["line1"] == "d e f"
        assert result.text_pointer_end == 6


class TestAllocateLinesSnapping:
    def test_snap_to_within_chant_7_anchor(self):
        # NW best_k=1, anchor at word 2 → diff=1=snap_window → snap to 2.
        words = ["alleluia", "dominus", "laudate", "deum"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat, ["line0"], {"line0": "alleluia"}, snap_window=1
        )
        # Snapped from k=1 to the anchor position of 2.
        assert result.manifest["line0"] == "alleluia dominus"
        assert result.text_pointer_end == 2

    def test_no_snap_when_diff_exceeds_window(self):
        # NW best_k=5, anchor at 2 → diff=3 > snap_window=1 → flag emitted.
        words = ["a", "b", "c", "d", "e", "f", "g"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat, ["line0"], {"line0": "a b c d e"}, snap_window=1
        )
        assert any(
            f.flag_type == "nw_volpiano_disagreement" for f in result.flags
        )
        # pointer advanced by best_k (5), not snapped to anchor (2)
        assert result.text_pointer_end == 5

    def test_no_flag_when_no_anchor(self):
        # No anchors → no snap, no disagreement flag.
        words = ["a", "b", "c"]
        flat = _flat(words)
        result = allocate_lines(flat, ["line0"], {"line0": "a b c"})
        assert not any(
            f.flag_type == "nw_volpiano_disagreement" for f in result.flags
        )


class TestAllocateLinesValidationFlags:
    def test_line_count_mismatch_flag(self):
        # flat_text has 6 words across 3 expected lines; only 1 label provided.
        words = ["a", "b", "c", "d", "e", "f"]
        anchors = [Anchor(2, "within_chant_7"), Anchor(4, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert any(f.flag_type == "line_count_mismatch" for f in result.flags)

    def test_no_mismatch_flag_when_all_words_consumed(self):
        words = ["a", "b"]
        flat = _flat(words)
        result = allocate_lines(flat, ["line0"], {"line0": "a b"})
        assert not any(
            f.flag_type == "line_count_mismatch" for f in result.flags
        )

    def test_column_count_uncertain_flag_when_no_777_anchor(self):
        # column_count=2, left_column_count=1 but no column_break_777 anchor.
        words = ["a", "b", "c", "d"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat,
            ["l0", "r0"],
            {"l0": "", "r0": ""},
            column_count=2,
            left_column_count=1,
        )
        assert any(
            f.flag_type == "column_count_uncertain" for f in result.flags
        )


class TestAllocateLinesColumnBreak:
    def test_column_break_resets_text_pointer(self):
        # left column (words 0-1), column_break at 3, right column (words 3-4).
        # Word 2 would be "orphaned" between columns — the hard-reset skips it.
        words = ["l0", "l1", "orphan", "r0", "r1"]
        anchors = [
            Anchor(2, "within_chant_7"),    # end of left column first line
            Anchor(3, "column_break_777"),  # column boundary
            Anchor(5, "within_chant_7"),    # end of right column first line
        ]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat,
            ["left_line0", "left_line1", "right_line0"],
            {"left_line0": "", "left_line1": "", "right_line0": ""},
            column_count=2,
            left_column_count=2,
        )
        # left_line0: advances to anchor at 2 → consumes "l0 l1"
        assert result.manifest["left_line0"] == "l0 l1"
        # left_line1: advances to column_break anchor at 3 → consumes "orphan"
        assert result.manifest["left_line1"] == "orphan"
        # right_line0: hard-reset to column_break word 3, advances to anchor 5
        assert result.manifest["right_line0"] == "r0 r1"
        assert not any(
            f.flag_type == "column_count_uncertain" for f in result.flags
        )

    def test_col1_cap_prevents_snap_overrun(self):
        # col_break at word 3; within_chant_7 at word 5 (beyond boundary).
        # Column-1 last line must stop at word 3, not 5.
        words = ["a", "b", "c", "d", "e", "f"]
        anchors = [
            Anchor(3, "column_break_777"),
            Anchor(5, "within_chant_7"),
            Anchor(6, "within_chant_7"),
        ]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat,
            ["col1_line0", "col1_line1", "col2_line0"],
            {
                "col1_line0": "a b",
                "col1_line1": "c d e f",
                "col2_line0": "d e",
            },
            column_count=2,
            left_column_count=2,
            snap_window=2,
            force_window=5,
        )
        # col1_line1 must not consume past word 3 even though
        # within_chant_7 is at 5
        assigned = result.manifest["col1_line1"]
        assert "e" not in assigned and "f" not in assigned
        # col2_line0 starts from col_break at word 3
        assert result.manifest["col2_line0"] == "d e"

    def test_hard_reset_unconditional(self):
        # col_break at word 2; col1 stub advances exactly to word 2 (capped by
        # col_break_word). Hard-reset must fire and col2 must start at word 2.
        # within_chant_7 at word 4 gives col2_line0 a snap target so NW
        # produces exactly "w2 w3".
        words = ["w0", "w1", "w2", "w3", "w4"]
        anchors = [
            Anchor(2, "column_break_777"),
            Anchor(4, "within_chant_7"),  # snap target for col2 line
            Anchor(5, "within_chant_7"),
        ]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat,
            ["col1_line0", "col2_line0"],
            {"col1_line0": "", "col2_line0": "w2 w3"},
            column_count=2,
            left_column_count=1,
        )
        # col2_line0 must start from col_break word 2, not from col1 end
        assert result.manifest["col2_line0"] == "w2 w3"
        assert not any(
            f.flag_type == "column_count_uncertain" for f in result.flags
        )


class TestForceWindow:
    """Tests for the force_window mid-chant snap feature."""

    def _mid_chant_flat(self, words, anchor_word_index):
        """FlatTextData with one span covering all words and one anchor."""
        return FlatTextData(
            words=words,
            anchors=[Anchor(anchor_word_index, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, len(words))],
        )

    def test_force_fires_mid_chant(self):
        # Single span (0-30), pointer=6, anchor=14.
        # NW best_k=2 → raw_end=8, diff=6 in (snap_window=2, force_window=10].
        # No new chant starts in (6, 14] → force fires.
        words = ["a"] * 30
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(14, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 30)],
            initial_pointer=6,
        )
        result = allocate_lines(
            flat, ["line0"], {"line0": "a a"},
            snap_window=2, force_window=10,
        )
        assert result.manifest["line0"] == " ".join(["a"] * 8)
        assert result.text_pointer_end == 14
        assert any(
            f.flag_type == "forced_mid_chant_snap" for f in result.flags
        )
        assert not any(
            f.flag_type == "nw_volpiano_disagreement" for f in result.flags
        )

    def test_force_blocked_when_chant_starts_in_window(self):
        # Two chants: span A (0-15), span B (15-30).
        # Pointer=8, anchor=16, diff=6 in force range.
        # Span B starts at 15, which is in (8, 16] → force must NOT fire.
        words = ["a"] * 30
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(16, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 15), ChantSpan(2, 15, 30)],
            initial_pointer=8,
        )
        result = allocate_lines(
            flat, ["line0"], {"line0": "a a"},
            snap_window=2, force_window=10,
        )
        assert not any(
            f.flag_type == "forced_mid_chant_snap" for f in result.flags
        )
        assert any(
            f.flag_type == "nw_volpiano_disagreement" for f in result.flags
        )

    def test_force_not_blocked_at_span_start(self):
        # Pointer is exactly at a span start (pointer=15=span.start_word).
        # No new chant starts in (15, 23] → force SHOULD fire.
        # (Being at a span start means we're at the beginning of a chant,
        # not mid-transition; the within_chant_7 anchor is within that chant.)
        words = ["a"] * 30
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(23, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 15), ChantSpan(2, 15, 30)],
            initial_pointer=15,
        )
        result = allocate_lines(
            flat, ["line0"], {"line0": "a a"},
            snap_window=2, force_window=10,
        )
        assert any(
            f.flag_type == "forced_mid_chant_snap" for f in result.flags
        )

    def test_force_blocked_for_page_break_77(self):
        # Anchor is page_break_77, not within_chant_7 → force must not fire.
        words = ["a"] * 20
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(8, "page_break_77")],
            chant_spans=[ChantSpan(1, 0, 20)],
        )
        result = allocate_lines(
            flat, ["line0"], {"line0": "a a"},
            snap_window=2, force_window=10,
        )
        assert not any(
            f.flag_type == "forced_mid_chant_snap" for f in result.flags
        )

    def test_force_window_zero_disables(self):
        # force_window=0 → feature off; nw_volpiano_disagreement emitted
        # even when diff is within a non-zero force_window.
        words = ["a"] * 30
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(14, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 30)],
            initial_pointer=6,
        )
        result = allocate_lines(
            flat, ["line0"], {"line0": "a a"},
            snap_window=2, force_window=0,
        )
        assert not any(
            f.flag_type == "forced_mid_chant_snap" for f in result.flags
        )
        assert any(
            f.flag_type == "nw_volpiano_disagreement" for f in result.flags
        )


class TestSplitWordAtSylBoundary:
    """Unit tests for the _split_word_at_syl_boundary helper."""

    def test_dominus_split_2_1(self):
        # "dominus" → ["do-", "mi-", "nus"] (3 syllables); split 2+1.
        result = _split_word_at_syl_boundary("dominus", 2, 1)
        assert result == ("domi", "nus")

    def test_dominus_split_1_2(self):
        result = _split_word_at_syl_boundary("dominus", 1, 2)
        assert result == ("do", "minus")

    def test_count_mismatch_returns_none(self):
        # under-count: syllabifier gives 3 < syl_left+syl_right=7 → None.
        assert _split_word_at_syl_boundary("dominus", 2, 5) is None

    def test_overcount_syllabifier_splits_at_syl_left(self):
        # "mathathie" → 4 syllables but volpiano says 1+2=3.
        # Over-count is allowed: split at syl_left=1 → ("ma", "thathie").
        result = _split_word_at_syl_boundary("mathathie", 1, 2)
        assert result == ("ma", "thathie")

    def test_zero_left_returns_none(self):
        assert _split_word_at_syl_boundary("dominus", 0, 3) is None

    def test_zero_right_returns_none(self):
        assert _split_word_at_syl_boundary("dominus", 3, 0) is None

    def test_single_syllable_word_mismatch_returns_none(self):
        # "lux" has 1 syllable; 1+1=2 ≠ 1.
        assert _split_word_at_syl_boundary("lux", 1, 1) is None

    def test_non_ascii_word_normalised(self):
        # Accented input is normalised before syllabification.
        # "dóminus" should behave the same as "dominus".
        result = _split_word_at_syl_boundary("dóminus", 2, 1)
        assert result == ("domi", "nus")


class TestAllocateLinesWithMidWordBreaks:
    """Tests for mid-word splitting in allocate_lines."""

    def _mid_word_flat(self, words, anchor_wi, syl_left, syl_right):
        """FlatTextData with one mid-word break and matching within_chant_7 anchor."""
        return FlatTextData(
            words=words,
            anchors=[Anchor(anchor_wi, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, len(words))],
            mid_word_breaks=[MidWordBreak(anchor_wi, syl_left, syl_right)],
        )

    def test_stub_mode_split_applied(self):
        # Stub mode (no OCR): line0 snaps to anchor at word 2; "beta" is the
        # split word.  _split_word_at_syl_boundary is mocked to return ("be", "ta").
        words = ["alpha", "beta", "gamma"]
        flat = self._mid_word_flat(words, anchor_wi=2, syl_left=1, syl_right=1)
        with patch(
            "steps.nw_chant_allocator._split_word_at_syl_boundary",
            return_value=("be", "ta"),
        ):
            result = allocate_lines(
                flat, ["line0", "line1"], {"line0": "", "line1": ""}
            )
        assert result.manifest["line0"] == "alpha be"
        assert result.manifest["line1"] == "ta gamma"
        assert not any(
            f.flag_type == "mid_word_split_skipped" for f in result.flags
        )

    def test_nw_mode_split_applied(self):
        # NW mode: OCR snaps to the anchor; split fires for the boundary word.
        words = ["alpha", "beta", "gamma"]
        flat = self._mid_word_flat(words, anchor_wi=2, syl_left=1, syl_right=1)
        with patch(
            "steps.nw_chant_allocator._split_word_at_syl_boundary",
            return_value=("be", "ta"),
        ):
            result = allocate_lines(
                flat,
                ["line0", "line1"],
                {"line0": "alpha", "line1": "gamma"},
                snap_window=1,
            )
        assert result.manifest["line0"] == "alpha be"
        assert result.manifest["line1"] == "ta gamma"

    def test_split_skipped_when_helper_returns_none(self):
        # When _split_word_at_syl_boundary returns None, word stays intact
        # and a "mid_word_split_skipped" flag is emitted.
        words = ["alpha", "beta", "gamma"]
        flat = self._mid_word_flat(words, anchor_wi=2, syl_left=1, syl_right=1)
        with patch(
            "steps.nw_chant_allocator._split_word_at_syl_boundary",
            return_value=None,
        ):
            result = allocate_lines(
                flat, ["line0", "line1"], {"line0": "", "line1": ""}
            )
        assert "beta" in result.manifest["line0"]
        assert any(
            f.flag_type == "mid_word_split_skipped" for f in result.flags
        )

    def test_no_split_when_no_mid_word_breaks(self):
        # Clean word break with no MidWordBreaks → manifest unchanged.
        words = ["alpha", "beta", "gamma"]
        flat = FlatTextData(
            words=words,
            anchors=[Anchor(2, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 3)],
        )
        result = allocate_lines(
            flat, ["line0", "line1"], {"line0": "", "line1": ""}
        )
        assert result.manifest["line0"] == "alpha beta"
        assert result.manifest["line1"] == "gamma"
        assert not any(
            f.flag_type == "mid_word_split_skipped" for f in result.flags
        )

    def test_syllable_prefix_not_applied_when_no_break(self):
        # Ensure syllable_prefix starts None and doesn't bleed into unrelated lines.
        words = ["alpha", "beta", "gamma", "delta"]
        flat = FlatTextData(
            words=words,
            anchors=[
                Anchor(2, "within_chant_7"),
                Anchor(4, "within_chant_7"),
            ],
            chant_spans=[ChantSpan(1, 0, 4)],
        )
        result = allocate_lines(
            flat,
            ["line0", "line1"],
            {"line0": "", "line1": ""},
        )
        assert result.manifest["line0"] == "alpha beta"
        assert result.manifest["line1"] == "gamma delta"

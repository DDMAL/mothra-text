"""Tests for steps.nw_chant_allocator — Sub-plan 4b: allocate_lines."""

from unittest.mock import patch

from steps.nw_chant_allocator import (
    Anchor,
    AllocationResult,
    ChantSpan,
    FlatTextData,
    MidWordBreak,
    _page_line_scale,
    _split_word_at_syl_boundary,
    _usable_ocr_chars,
    allocate_lines,
    locate_first_chant_line,
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
        # force_window=0 pins this test to snap_window's own boundary,
        # independent of allocate_lines()'s force_window default.
        words = ["a", "b", "c", "d", "e", "f", "g"]
        anchors = [Anchor(2, "within_chant_7")]
        flat = _flat(words, anchors)
        result = allocate_lines(
            flat, ["line0"], {"line0": "a b c d e"},
            snap_window=1, force_window=0,
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


# ---------------------------------------------------------------------------
# Helpers for folio-start location tests
# ---------------------------------------------------------------------------

def _flat_no_volpiano(
    cont_words: list[str],
    folio_words: list[str],
    folio_seq: int = 1,
) -> FlatTextData:
    """FlatTextData with no anchors, optional continuation span + folio span."""
    words = cont_words + folio_words
    spans = []
    if cont_words:
        spans.append(ChantSpan(0, 0, len(cont_words)))
    spans.append(ChantSpan(folio_seq, len(cont_words), len(words)))
    return FlatTextData(
        words=words,
        anchors=[],
        chant_spans=spans,
        has_continuation=bool(cont_words),
    )


class TestLocateFirstChantLine:
    """Unit tests for locate_first_chant_line()."""

    def test_returns_none_when_no_folio_chant_span(self):
        # Only a continuation span (sequence=0); no folio chant.
        flat = FlatTextData(
            words=["carry"],
            anchors=[],
            chant_spans=[ChantSpan(0, 0, 1)],
        )
        result = locate_first_chant_line(flat, ["line0"], {"line0": "carry"})
        assert result is None

    def test_returns_none_when_all_ocr_empty(self):
        flat = _flat_no_volpiano([], ["alleluia", "dominus"])
        result = locate_first_chant_line(
            flat,
            ["line0", "line1"],
            {"line0": "", "line1": ""},
        )
        assert result is None

    def test_returns_none_when_no_probe_words(self):
        # Folio span exists but is empty (start == end).
        flat = FlatTextData(
            words=["carry"],
            anchors=[],
            chant_spans=[ChantSpan(0, 0, 1), ChantSpan(1, 1, 1)],
        )
        result = locate_first_chant_line(flat, ["line0"], {"line0": "carry"})
        assert result is None

    def test_exact_match_returns_correct_index(self):
        # Probe = first 8 words of folio chant = "alleluia dominus".
        # line0 OCR is gibberish; line1 OCR matches exactly.
        flat = _flat_no_volpiano(
            cont_words=["prev1", "prev2"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy qwerty", "line1": "alleluia dominus"}
        result = locate_first_chant_line(flat, ["line0", "line1"], ocr)
        assert result is not None
        idx, score = result
        assert idx == 1
        assert score > 0.0

    def test_returns_index_zero_when_first_line_best_matches(self):
        flat = _flat_no_volpiano([], ["alleluia", "dominus"])
        ocr = {"line0": "alleluia dominus", "line1": "xyzzy qwerty"}
        result = locate_first_chant_line(flat, ["line0", "line1"], ocr)
        assert result is not None
        idx, score = result
        assert idx == 0
        assert score > 0.0

    def test_uses_only_first_n_probe_words(self):
        # n_probe_words=1: probe = "alleluia" only.
        flat = _flat_no_volpiano([], ["alleluia", "dominus", "laudate"])
        ocr = {"line0": "alleluia extra", "line1": "dominus laudate"}
        result = locate_first_chant_line(
            flat, ["line0", "line1"], ocr, n_probe_words=1
        )
        assert result is not None
        idx, _ = result
        # "alleluia" probe should match line0 best.
        assert idx == 0

    def test_accepts_external_aligner(self):
        from Bio.Align import PairwiseAligner
        al = PairwiseAligner()
        al.mode = "global"
        al.match_score = 8.0
        al.mismatch_score = -5.0
        al.open_gap_score = -7.0
        al.extend_gap_score = -3.0
        flat = _flat_no_volpiano([], ["alleluia", "dominus"])
        ocr = {"line0": "xyzzy", "line1": "alleluia dominus"}
        result = locate_first_chant_line(
            flat, ["line0", "line1"], ocr, aligner=al
        )
        assert result is not None
        assert result[0] == 1


class TestFolioStartLocation:
    """Integration tests for folio-start detection inside allocate_lines()."""

    def test_folio_start_detected_flag_emitted(self):
        # line0 OCR gibberish; line1 OCR matches folio chant → L*=1, flag emitted.
        flat = _flat_no_volpiano(
            cont_words=["carry1", "carry2"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy qwerty", "line1": "alleluia dominus"}
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        flag_types = [f.flag_type for f in result.flags]
        assert "folio_start_detected" in flag_types

    def test_pre_start_lines_get_continuation_words(self):
        # cont_words=["carry1","carry2"], folio_words=["alleluia","dominus"]
        # line0 → carry words (force-snapped); line1 → folio chant start.
        flat = _flat_no_volpiano(
            cont_words=["carry1", "carry2"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy qwerty", "line1": "alleluia dominus"}
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        # Pre-start line should hold the continuation words.
        assert result.manifest["line0"] == "carry1 carry2"

    def test_folio_region_starts_at_first_chant(self):
        flat = _flat_no_volpiano(
            cont_words=["carry1", "carry2"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy qwerty", "line1": "alleluia dominus"}
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        # Folio line should contain only the folio chant words.
        assert result.manifest["line1"] == "alleluia dominus"

    def test_force_snap_consumes_all_continuation_words(self):
        # Two pre-start lines, 3 continuation words.
        # NW might under-assign, but force-snap on last pre-start line must
        # consume the remaining word.
        flat = _flat_no_volpiano(
            cont_words=["w1", "w2", "w3"],
            folio_words=["alleluia", "dominus"],
        )
        # line0 and line1 are pre-start; line2 is folio start.
        ocr = {
            "line0": "xyzzy",
            "line1": "qwerty",
            "line2": "alleluia dominus",
        }
        result = allocate_lines(flat, ["line0", "line1", "line2"], ocr)
        # All 3 continuation words must appear across line0 and line1.
        pre_words = (
            result.manifest["line0"].split()
            + result.manifest["line1"].split()
        )
        assert set(pre_words) == {"w1", "w2", "w3"}
        assert result.manifest["line2"] == "alleluia dominus"

    def test_pre_start_empty_when_no_continuation(self):
        # has_continuation=False: pre-start lines should get "".
        flat = _flat_no_volpiano(
            cont_words=[],       # no continuation
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy qwerty", "line1": "alleluia dominus"}
        # Inject a fake continuation scenario by patching locate to return 1.
        # Simpler: use folio_start_min_score so line0 is still detected as
        # pre-start without real continuation data.
        # We need folio_start_line > 0: make line0 OCR non-matching enough
        # that line1 scores higher on the probe "alleluia dominus".
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        flag_types = [f.flag_type for f in result.flags]
        if "folio_start_detected" in flag_types:
            # When L*=1, line0 is pre-start with no continuation → empty.
            assert result.manifest["line0"] == ""
        # Either way, line1 should contain the folio chant words.
        assert "alleluia" in result.manifest["line1"]

    def test_strategy_disabled_when_anchors_present(self):
        # flat_text.anchors is non-empty → strategy must NOT activate.
        flat = FlatTextData(
            words=["carry1", "alleluia", "dominus"],
            anchors=[Anchor(1, "within_chant_7")],
            chant_spans=[
                ChantSpan(0, 0, 1),
                ChantSpan(1, 1, 3),
            ],
            has_continuation=True,
        )
        ocr = {"line0": "xyzzy", "line1": "alleluia dominus"}
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        flag_types = [f.flag_type for f in result.flags]
        assert "folio_start_detected" not in flag_types
        assert "folio_start_not_located" not in flag_types

    def test_strategy_disabled_in_stub_mode(self):
        # All OCR empty → strategy must NOT activate even with no anchors.
        flat = _flat_no_volpiano(
            cont_words=["carry"],
            folio_words=["alleluia", "dominus"],
        )
        result = allocate_lines(
            flat,
            ["line0", "line1"],
            {"line0": "", "line1": ""},
        )
        flag_types = [f.flag_type for f in result.flags]
        assert "folio_start_detected" not in flag_types
        assert "folio_start_not_located" not in flag_types

    def test_fallback_flag_when_score_below_threshold(self):
        flat = _flat_no_volpiano(
            cont_words=["carry"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "carry", "line1": "alleluia dominus"}
        result = allocate_lines(
            flat,
            ["line0", "line1"],
            ocr,
            folio_start_min_score=100.0,  # impossibly high
        )
        flag_types = [f.flag_type for f in result.flags]
        assert "folio_start_not_located" in flag_types
        assert "folio_start_detected" not in flag_types

    def test_locate_folio_start_false_disables_strategy(self):
        flat = _flat_no_volpiano(
            cont_words=["carry"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "xyzzy", "line1": "alleluia dominus"}
        result = allocate_lines(
            flat, ["line0", "line1"], ocr, locate_folio_start=False
        )
        flag_types = [f.flag_type for f in result.flags]
        assert "folio_start_detected" not in flag_types
        assert "folio_start_not_located" not in flag_types

    def test_no_pre_start_lines_when_first_chant_at_line_zero(self):
        # Both line0 and line1 OCR match folio chant → L*=0, no pre-start.
        flat = _flat_no_volpiano(
            cont_words=["carry1", "carry2"],
            folio_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "alleluia dominus", "line1": "xyzzy"}
        result = allocate_lines(flat, ["line0", "line1"], ocr)
        flag_types = [f.flag_type for f in result.flags]
        # L*=0 → no pre-start → flag not emitted.
        assert "folio_start_detected" not in flag_types


# ---------------------------------------------------------------------------
# Helpers for suffix alignment tests
# ---------------------------------------------------------------------------

def _flat_suffix(
    folio_words: list[str],
    suffix_probe_words: list[str],
    folio_seq: int = 1,
) -> FlatTextData:
    """FlatTextData with no anchors, no continuation, and a suffix probe."""
    return FlatTextData(
        words=folio_words,
        anchors=[],
        chant_spans=[ChantSpan(folio_seq, 0, len(folio_words))],
        has_continuation=False,
        suffix_probe_words=suffix_probe_words,
    )


class TestPreStartSuffixAlignment:
    """Tests for suffix alignment of pre-start lines (no has_continuation)."""

    def test_suffix_probe_words_populated_by_build_flat_text(self):
        # build_flat_text_and_anchors should populate suffix_probe_words from
        # the preceding folio's last chant when has_continuation=False.
        from steps.nw_chant_allocator import build_flat_text_and_anchors
        csv_rows = [
            {"folio": "003v", "sequence": "1", "mode": "7",
             "fulltext_ms": "alleluia dominus", "volpiano": ""},
            {"folio": "004r", "sequence": "1", "mode": "7",
             "fulltext_ms": "kyrie eleison", "volpiano": ""},
        ]
        flat = build_flat_text_and_anchors(csv_rows, folio="004r")
        assert flat.suffix_probe_words == ["alleluia", "dominus"]

    def test_suffix_probe_empty_when_no_preceding_folio(self):
        from steps.nw_chant_allocator import build_flat_text_and_anchors
        csv_rows = [
            {"folio": "001r", "sequence": "1", "mode": "7",
             "fulltext_ms": "kyrie eleison", "volpiano": ""},
        ]
        flat = build_flat_text_and_anchors(csv_rows, folio="001r")
        assert flat.suffix_probe_words == []

    def test_suffix_probe_empty_when_preceding_row_has_no_text(self):
        from steps.nw_chant_allocator import build_flat_text_and_anchors
        csv_rows = [
            {"folio": "003v", "sequence": "1", "mode": "7",
             "fulltext_ms": "", "volpiano": ""},
            {"folio": "004r", "sequence": "1", "mode": "7",
             "fulltext_ms": "kyrie eleison", "volpiano": ""},
        ]
        flat = build_flat_text_and_anchors(csv_rows, folio="004r")
        assert flat.suffix_probe_words == []

    def test_suffix_probe_not_populated_when_has_continuation(self):
        # When 77 is found on a preceding folio, has_continuation=True and
        # suffix_probe_words should remain empty.
        from steps.nw_chant_allocator import build_flat_text_and_anchors
        csv_rows = [
            {"folio": "003v", "sequence": "1", "mode": "7",
             "fulltext_ms": "alleluia dominus carry", "volpiano": "1---d77d"},
            {"folio": "004r", "sequence": "1", "mode": "7",
             "fulltext_ms": "kyrie eleison", "volpiano": ""},
        ]
        flat = build_flat_text_and_anchors(csv_rows, folio="004r")
        assert flat.has_continuation is True
        assert flat.suffix_probe_words == []

    def test_suffix_alignment_detected_flag_emitted(self):
        # When pre-start OCR matches the suffix of the probe, the flag fires.
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus", "laudate"],
        )
        # line0 is pre-start (OCR matches "laudate"), line1 is folio start
        ocr = {"line0": "laudate", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0)
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_detected" in flag_types

    def test_pre_start_lines_get_suffix_words(self):
        # The pre-start line should receive words from the suffix probe, not "".
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus", "laudate"],
        )
        ocr = {"line0": "laudate", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0)
        # line0 is pre-start; should have some suffix words (not empty)
        assert result.manifest.get("line0", "") != ""

    def test_folio_region_unchanged(self):
        # The folio region (line1 onward) should still get folio words.
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "alleluia", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0)
        assert "kyrie" in result.manifest.get("line1", "")

    def test_force_snap_last_pre_start_line_consumes_all_suffix(self):
        # With 2 pre-start lines, the last one must consume all remaining
        # suffix words via force-snap.
        flat = _flat_suffix(
            folio_words=["kyrie"],
            suffix_probe_words=["a", "b", "c", "d", "e"],
        )
        # line0 and line1 are pre-start, line2 is folio start
        ocr = {"line0": "a b", "line1": "xyz", "line2": "kyrie"}
        result = allocate_lines(flat, ["line0", "line1", "line2"], ocr,
                                folio_start_min_score=-999.0)
        words0 = result.manifest.get("line0", "").split()
        words1 = result.manifest.get("line1", "").split()
        # All suffix words must be consumed across the two pre-start lines
        assert len(words0) + len(words1) == 5

    def test_below_threshold_falls_back_to_empty(self):
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "alleluia", "line1": "kyrie eleison"}
        # folio_start_min_score low so L*>0 is detected; suffix threshold high
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0,
                                pre_start_suffix_min_score=999.0)
        assert result.manifest.get("line0", "") == ""
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_skipped" in flag_types

    def test_disabled_when_has_continuation_true(self):
        # has_continuation=True → existing continuation path, not suffix path.
        flat = _flat_no_volpiano(
            cont_words=["carry1", "carry2"],
            folio_words=["kyrie", "eleison"],
        )
        flat = FlatTextData(
            words=flat.words,
            anchors=flat.anchors,
            chant_spans=flat.chant_spans,
            has_continuation=True,
            suffix_probe_words=["prev_chant_word"],
        )
        ocr = {"line0": "carry1", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0)
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_detected" not in flag_types

    def test_disabled_when_suffix_probe_empty(self):
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=[],
        )
        ocr = {"line0": "xyzzy", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0)
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_detected" not in flag_types
        assert "suffix_alignment_skipped" not in flag_types

    def test_disabled_when_pre_start_suffix_align_false(self):
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "alleluia", "line1": "kyrie eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                folio_start_min_score=-999.0,
                                pre_start_suffix_align=False)
        assert result.manifest.get("line0", "") == ""
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_detected" not in flag_types
        assert "suffix_alignment_skipped" not in flag_types

    def test_no_activation_when_folio_start_line_is_zero(self):
        # Disable folio-start detection → L*=0 always → suffix alignment
        # never activates (no pre-start region exists).
        flat = _flat_suffix(
            folio_words=["kyrie", "eleison"],
            suffix_probe_words=["alleluia", "dominus"],
        )
        ocr = {"line0": "kyrie", "line1": "eleison"}
        result = allocate_lines(flat, ["line0", "line1"], ocr,
                                locate_folio_start=False)
        flag_types = [f.flag_type for f in result.flags]
        assert "suffix_alignment_detected" not in flag_types
        assert "suffix_alignment_skipped" not in flag_types


# ---------------------------------------------------------------------------
# Helpers for mixed-line detection tests
# ---------------------------------------------------------------------------

def _make_fused(label, constituent_labels, constituent_widths, column=1):
    from steps.column_clustering import FusedLine
    return FusedLine(
        label=label,
        constituent_labels=constituent_labels,
        constituent_widths=constituent_widths,
        xmin=0, xmax=sum(constituent_widths), ymin=0, ymax=30,
        column=column,
    )


def _mixed_flat(cont_words, folio_words):
    return _flat_no_volpiano(
        cont_words=cont_words,
        folio_words=folio_words,
    )


def _mixed_fused_2(seg_widths_0=(100, 100), seg_widths_1=(200,)):
    return [
        _make_fused("line0", ["seg0a", "seg0b"], list(seg_widths_0)),
        _make_fused("line1", ["seg1a"], list(seg_widths_1)),
    ]


class TestMixedLineDetection:
    """Tests for mixed-line detection: folio words on pre-start line."""

    # ------------------------------------------------------------------
    # Basic activation
    # ------------------------------------------------------------------

    def test_flag_emitted_when_mixed_line_detected(self):
        # L*=1; right constituent of line0 OCR matches first folio word.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        flag_types = [f.flag_type for f in result.flags]
        assert "mixed_start_detected" in flag_types

    def test_folio_word_moved_to_right_constituent(self):
        # Right constituent of line0 should receive the first folio word.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides.get("seg0b") == "alleluia"

    def test_left_constituent_gets_continuation_word(self):
        # Left constituent should receive the continuation word.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides.get("seg0a") == "carry"

    def test_folio_start_line_in_result(self):
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.folio_start_line == 1

    # ------------------------------------------------------------------
    # Pointer adjustment: moved words must not appear again on L*
    # ------------------------------------------------------------------

    def test_folio_region_does_not_repeat_moved_word(self):
        # "alleluia" moved to seg0b; L* (line1) must start at "dominus".
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.manifest["line1"] == "dominus"
        assert "alleluia" not in result.manifest["line1"]

    # ------------------------------------------------------------------
    # Multi-word detection (up to mixed_line_n_words)
    # ------------------------------------------------------------------

    def test_two_folio_words_detected(self):
        # L*=1 (line0 OCR is noise, line1 OCR matches probe).
        # Right two constituents of line0 OCR as first 2 folio words.
        flat = _mixed_flat(["carry"], ["alpha", "beta", "gamma"])
        fused = [
            _make_fused("line0", ["seg0a", "seg0b", "seg0c"],
                        [100, 100, 100]),
            _make_fused("line1", ["seg1a"], [200]),
        ]
        node_ocr = {
            "seg0a": "carry", "seg0b": "alpha", "seg0c": "beta",
            "seg1a": "alpha beta gamma",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "xyzzy qwerty", "line1": "alpha beta gamma"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides.get("seg0b") == "alpha"
        assert result.constituent_overrides.get("seg0c") == "beta"
        assert result.manifest["line1"] == "gamma"

    # ------------------------------------------------------------------
    # Noise constituent absorbed into right portion
    # ------------------------------------------------------------------

    def test_noise_constituent_absorbed_into_right_portion(self):
        # seg0c OCR "v" is noise; seg0b "reocupe" ~ "Preoccupemus".
        # Best split includes both seg0b + seg0c in the right portion;
        # proportional distribution gives seg0b the word, seg0c gets "".
        flat = _mixed_flat(["carry"], ["Preoccupemus", "faciem"])
        fused = [
            _make_fused("line0", ["seg0a", "seg0b", "seg0c"],
                        [100, 200, 60]),
            _make_fused("line1", ["seg1a"], [300]),
        ]
        node_ocr = {
            "seg0a": "carry", "seg0b": "reocupe",
            "seg0c": "v", "seg1a": "faciem",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry reocupe v", "line1": "faciem"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides.get("seg0b") == "Preoccupemus"
        assert result.constituent_overrides.get("seg0c") == ""
        assert result.manifest["line1"] == "faciem"

    # ------------------------------------------------------------------
    # No-activation cases
    # ------------------------------------------------------------------

    def test_no_activation_when_folio_start_line_is_zero(self):
        # locate_folio_start=False forces L*=0 — detection must not run.
        flat = _mixed_flat([], ["alpha", "beta"])
        fused = [_make_fused("line0", ["seg0a", "seg0b"], [100, 100])]
        node_ocr = {"seg0a": "alpha", "seg0b": "beta"}
        result = allocate_lines(
            flat, ["line0"], {"line0": "alpha beta"},
            fused_lines=fused, node_ocr=node_ocr,
            locate_folio_start=False,
        )
        assert result.constituent_overrides == {}
        flag_types = [f.flag_type for f in result.flags]
        assert "mixed_start_detected" not in flag_types

    def test_no_activation_when_fused_lines_is_none(self):
        # fused_lines not supplied — constituent_overrides stays empty.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
        )
        assert result.constituent_overrides == {}

    def test_no_activation_when_anchors_present(self):
        # Volpiano anchors present — no-volpiano guard fires, skipped.
        words = ["alpha", "beta", "gamma", "delta"]
        anchors = [Anchor(2, "within_chant_7"), Anchor(4, "within_chant_7")]
        flat = FlatTextData(
            words=words, anchors=anchors,
            chant_spans=[ChantSpan(1, 0, 4)],
        )
        fused = [
            _make_fused("line0", ["seg0a", "seg0b"], [100, 100]),
            _make_fused("line1", ["seg1a", "seg1b"], [100, 100]),
        ]
        node_ocr = {
            "seg0a": "alpha", "seg0b": "beta",
            "seg1a": "gamma", "seg1b": "delta",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "alpha beta", "line1": "gamma delta"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides == {}

    def test_no_activation_when_single_constituent_line(self):
        # Only one constituent on L*-1 — nothing to split.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = [
            _make_fused("line0", ["seg0a"], [200]),
            _make_fused("line1", ["seg1a"], [200]),
        ]
        node_ocr = {"seg0a": "carry alleluia", "seg1a": "dominus"}
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides == {}

    def test_below_threshold_falls_back_cleanly(self):
        # Score 9.0 exceeds the theoretical max (~8.0) — no detection.
        flat = _mixed_flat(["carry"], ["alleluia", "dominus"])
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia", "seg1a": "dominus",
        }
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "dominus"},
            fused_lines=fused, node_ocr=node_ocr,
            mixed_line_min_score=9.0,
        )
        assert result.constituent_overrides == {}
        flag_types = [f.flag_type for f in result.flags]
        assert "mixed_start_detected" not in flag_types

    def test_no_activation_when_no_pre_start_content(self):
        # No suffix_probe_words → suffix alignment block never runs
        # → _suffix_words stays [] → _ml_suf = [] → detection blocked.
        # Models the A-Gu 29_013r regression: OCR on the last pre-start
        # line matches the first folio word, but there is no left-side
        # pre-start text to justify the fix, so NW must start at word 0.
        flat = FlatTextData(
            words=["alleluia", "dominus"],
            anchors=[],
            chant_spans=[ChantSpan(1, 0, 2)],
            has_continuation=False,
        )
        fused = _mixed_fused_2()
        node_ocr = {
            "seg0a": "carry", "seg0b": "alleluia",
            "seg1a": "alleluia dominus",
        }
        # line0 OCR is poor vs probe; line1 OCR matches exactly → L*=1.
        result = allocate_lines(
            flat, ["line0", "line1"],
            {"line0": "carry alleluia", "line1": "alleluia dominus"},
            fused_lines=fused, node_ocr=node_ocr,
        )
        assert result.constituent_overrides == {}
        flag_types = [f.flag_type for f in result.flags]
        assert "mixed_start_detected" not in flag_types
        # NW starts at word 0 — no cascade shift.
        assert result.manifest["line1"] == "alleluia dominus"


# ---------------------------------------------------------------------------
# Misdetected-line skipping (issue #46)
# ---------------------------------------------------------------------------

# Three full-width reference lines give _page_line_scale a trustworthy page
# scale.  With ref_width=1500px and the densest reference line averaging
# 1500/32 ≈ 47px per character, a 60px box holds ~1.3 characters — nowhere
# near enough for the multi-word spans the allocator wants to put in it.
_REF_WIDTH = 1500
_NARROW = 60
_W0 = ["alphabetum", "beatissimus", "confessorum"]
_W1 = ["seculorum", "amenissimo", "venerandus"]
_W2 = ["dominusque", "exultemus", "gloriosus"]
_W3 = ["laudemus", "canticum", "novum"]


def _misdetect_flat(anchors=True):
    """Nine words; anchors at 3 and 6 so each line closes on a word group."""
    return FlatTextData(
        words=_W0 + _W1 + _W2,
        anchors=(
            [Anchor(3, "within_chant_7"), Anchor(6, "within_chant_7")]
            if anchors else []
        ),
        chant_spans=[ChantSpan(1, 0, 9)],
    )


def _misdetect_page(phantom_width=_NARROW, phantom_ocr=""):
    """ref0, a narrow empty-OCR phantom, then ref1 and ref2."""
    labels = ["ref0", "phantom", "ref1", "ref2"]
    fused = [
        _make_fused("ref0", ["s0"], [_REF_WIDTH]),
        _make_fused("phantom", ["s1"], [phantom_width]),
        _make_fused("ref1", ["s2"], [_REF_WIDTH]),
        _make_fused("ref2", ["s3"], [_REF_WIDTH]),
    ]
    ocr = {
        "ref0": " ".join(_W0),
        "phantom": phantom_ocr,
        "ref1": " ".join(_W1),
        "ref2": " ".join(_W2),
    }
    return labels, fused, ocr


def _one_word_page(phantom_width):
    """Phantom is offered exactly one word ('iactatus', 8 chars).

    At 400px the box can hold it (8.5 char capacity vs a 3.2 char
    threshold) and must be kept; at 69px it cannot (1.5 chars) and must be
    skipped.  This pair pins misdetect_min_words=1: a word count alone
    cannot tell the two apart, only capacity can.
    """
    flat = FlatTextData(
        words=_W0 + ["iactatus"] + _W1 + _W2,
        anchors=[
            Anchor(3, "within_chant_7"),
            Anchor(4, "within_chant_7"),
            Anchor(7, "within_chant_7"),
        ],
        chant_spans=[ChantSpan(1, 0, 13)],
    )
    labels = ["ref0", "phantom", "ref1", "ref2"]
    fused = [
        _make_fused("ref0", ["s0"], [_REF_WIDTH]),
        _make_fused("phantom", ["s1"], [phantom_width]),
        _make_fused("ref1", ["s2"], [_REF_WIDTH]),
        _make_fused("ref2", ["s3"], [_REF_WIDTH]),
    ]
    ocr = {
        "ref0": " ".join(_W0),
        "phantom": "",
        "ref1": " ".join(_W1),
        "ref2": " ".join(_W2),
    }
    return flat, labels, fused, ocr


class TestPageLineScale:
    """Unit tests for the page-geometry helpers."""

    def test_punctuation_only_ocr_counts_as_no_text(self):
        assert _usable_ocr_chars("„") == 0
        assert _usable_ocr_chars(" . - ") == 0
        assert _usable_ocr_chars("") == 0
        assert _usable_ocr_chars(None) == 0
        assert _usable_ocr_chars("ab1") == 3

    def test_returns_none_with_too_few_reference_lines(self):
        fused = [
            _make_fused("a", ["s0"], [1000]),
            _make_fused("b", ["s1"], [1000]),
        ]
        ocr = {"a": "alphabetum", "b": "beatissimus"}
        assert _page_line_scale(fused, ocr) == (None, None)

    def test_returns_none_when_all_ocr_empty(self):
        # Global stub mode: no line qualifies as a reference line.
        _, fused, _ = _misdetect_page()
        ocr = {f.label: "" for f in fused}
        assert _page_line_scale(fused, ocr) == (None, None)

    def test_median_width_and_narrowest_char_width(self):
        fused = [
            _make_fused("a", ["s0"], [1000]),
            _make_fused("b", ["s1"], [1200]),
            _make_fused("c", ["s2"], [1400]),
        ]
        # 10 alnum chars each → 100, 120 and 140 px per character.
        ocr = {k: "abcdefghij" for k in ("a", "b", "c")}
        ref_width, px_per_char = _page_line_scale(fused, ocr)
        assert ref_width == 1200
        assert px_per_char == 100

    def test_short_ocr_lines_are_not_reference_lines(self):
        fused = [
            _make_fused("a", ["s0"], [1000]),
            _make_fused("b", ["s1"], [1000]),
            _make_fused("c", ["s2"], [1000]),
            _make_fused("tiny", ["s3"], [20]),
        ]
        ocr = {
            "a": "abcdefghij", "b": "abcdefghij", "c": "abcdefghij",
            "tiny": "ab",
        }
        ref_width, px_per_char = _page_line_scale(fused, ocr)
        # The 20px box would otherwise drag both statistics down.
        assert ref_width == 1000
        assert px_per_char == 100


class TestMisdetectedLineSkip:
    """Tests for skipping non-text boxes during allocation (issue #46)."""

    # ------------------------------------------------------------------
    # The core scenario, in both stub-mode demand paths
    # ------------------------------------------------------------------

    def test_words_go_to_next_line_in_anchor_advance_path(self):
        # Without the veto the phantom stub-advances to the anchor at word
        # 6 and swallows _W1, shifting every later line.
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.manifest["phantom"] == ""
        assert result.manifest["ref0"] == " ".join(_W0)
        assert result.manifest["ref1"] == " ".join(_W1)
        assert result.manifest["ref2"] == " ".join(_W2)

    def test_words_go_to_next_line_in_uniform_fallback_path(self):
        # No volpiano: demand comes from the uniform-distribution fallback
        # instead of an anchor advance, and the veto must still fire.
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(anchors=False), labels, ocr,
            fused_lines=fused, locate_folio_start=False,
        )
        assert result.manifest["phantom"] == ""
        assert result.manifest["ref1"] == " ".join(_W1)
        assert result.manifest["ref2"] == " ".join(_W2)

    def test_flag_and_skipped_labels_recorded(self):
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.skipped_labels == ["phantom"]
        skips = [
            f for f in result.flags
            if f.flag_type == "misdetected_line_skipped"
        ]
        assert len(skips) == 1
        assert "phantom" in skips[0].detail

    def test_pointer_not_advanced_and_all_words_allocated(self):
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.text_pointer_end == 9
        assert not any(
            f.flag_type == "line_count_mismatch" for f in result.flags
        )

    def test_debug_entry_records_the_skip(self):
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused, debug=True,
        )
        entry = next(
            e for e in result.debug_lines if e["label"] == "phantom"
        )
        assert entry["consumed"] == 0
        assert entry["pointer_start"] == entry["pointer_end"]
        assert "misdetected" in entry["alignment"]

    # ------------------------------------------------------------------
    # misdetect_min_words=1: capacity, not word count, does the work
    # ------------------------------------------------------------------

    def test_single_word_demand_that_fits_is_kept(self):
        flat, labels, fused, ocr = _one_word_page(phantom_width=400)
        result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.skipped_labels == []
        assert result.manifest["phantom"] == "iactatus"

    def test_single_word_demand_that_cannot_fit_is_skipped(self):
        flat, labels, fused, ocr = _one_word_page(phantom_width=69)
        result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.skipped_labels == ["phantom"]
        assert result.manifest["phantom"] == ""

    # ------------------------------------------------------------------
    # Precision guards — each must leave the line alone
    # ------------------------------------------------------------------

    def test_narrow_line_with_real_ocr_is_kept(self):
        # OCR evidence that the box holds text beats the geometry.
        labels, fused, ocr = _misdetect_page(phantom_ocr="amen")
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.skipped_labels == []

    def test_full_width_line_with_empty_ocr_is_kept(self):
        # A real line whose OCR simply failed must not be skipped.
        labels, fused, ocr = _misdetect_page(phantom_width=_REF_WIDTH)
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.skipped_labels == []
        assert result.manifest["phantom"] == " ".join(_W1)

    def test_inert_without_fused_lines(self):
        labels, _, ocr = _misdetect_page()
        result = allocate_lines(_misdetect_flat(), labels, ocr)
        assert result.skipped_labels == []
        assert result.manifest["phantom"] == " ".join(_W1)

    def test_inert_in_global_stub_mode(self):
        # Every OCR empty (--stub-mode): no reference lines, no page scale,
        # so no line may be judged a misdetection.
        labels, fused, ocr = _misdetect_page()
        ocr = {k: "" for k in ocr}
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
        )
        assert result.skipped_labels == []
        assert result.manifest["phantom"] == " ".join(_W1)

    def test_disabled_by_flag(self):
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
            skip_misdetected_lines=False,
        )
        assert result.skipped_labels == []
        assert result.manifest["phantom"] == " ".join(_W1)

    def test_width_ratio_can_be_tightened_to_spare_the_line(self):
        labels, fused, ocr = _misdetect_page()
        result = allocate_lines(
            _misdetect_flat(), labels, ocr, fused_lines=fused,
            misdetect_width_ratio=0.01,
        )
        assert result.skipped_labels == []

    # ------------------------------------------------------------------
    # Interaction with mid-word syllable carry-over
    # ------------------------------------------------------------------

    def test_syllable_prefix_survives_a_skip(self):
        # ref0 ends on a mid-word break; the carried fragment must land on
        # the next *real* line, not be lost with the skipped box.
        flat = FlatTextData(
            words=_W0 + _W1 + _W2,
            anchors=[Anchor(3, "within_chant_7"),
                     Anchor(6, "within_chant_7")],
            chant_spans=[ChantSpan(1, 0, 9)],
            mid_word_breaks=[MidWordBreak(3, 1, 1)],
        )
        labels, fused, ocr = _misdetect_page()
        with patch(
            "steps.nw_chant_allocator._split_word_at_syl_boundary",
            return_value=("confes", "sorum"),
        ):
            result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.manifest["ref0"].endswith("confes")
        assert result.manifest["phantom"] == ""
        assert result.manifest["ref1"].startswith("sorum ")


class TestMisdetectedLineSkipPreStart:
    """The veto also covers the pre-start region (no-volpiano folios)."""

    def _pre_start_page(self, phantom_index=1):
        """Four lines with the phantom at phantom_index; L* is line 3."""
        order = ["ref0", "ref1"]
        order.insert(phantom_index, "phantom")
        labels = order + ["ref2"]
        widths = {
            "ref0": _REF_WIDTH, "ref1": _REF_WIDTH,
            "ref2": _REF_WIDTH, "phantom": _NARROW,
        }
        fused = [
            _make_fused(lbl, [f"s{i}"], [widths[lbl]])
            for i, lbl in enumerate(labels)
        ]
        ocr = {
            "ref0": " ".join(_W0),
            "ref1": " ".join(_W1),
            "phantom": "",
            "ref2": " ".join(_W2 + _W3),
        }
        return labels, fused, ocr

    def test_skipped_in_continuation_sub_branch(self):
        # has_continuation: pre-start lines share the carried-over words.
        flat = _flat_no_volpiano(
            cont_words=_W0 + _W1, folio_words=_W2 + _W3,
        )
        labels, fused, ocr = self._pre_start_page()
        result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.folio_start_line == 3
        assert result.skipped_labels == ["phantom"]
        assert result.manifest["phantom"] == ""
        # Without the veto the phantom takes "seculorum" and ref1 is left
        # with the remainder.
        assert result.manifest["ref1"] == " ".join(_W1)

    def test_skipped_in_suffix_alignment_sub_branch(self):
        # No continuation: pre-start lines are fed from the preceding
        # folio's last chant via suffix alignment.
        flat = FlatTextData(
            words=_W2 + _W3,
            anchors=[],
            chant_spans=[ChantSpan(1, 0, 6)],
            has_continuation=False,
            suffix_probe_words=_W0 + _W1,
        )
        labels, fused, ocr = self._pre_start_page()
        result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.folio_start_line == 3
        assert result.skipped_labels == ["phantom"]
        assert result.manifest["phantom"] == ""
        assert result.manifest["ref1"] == " ".join(_W1)

    def test_vetoed_last_pre_start_line_is_not_force_snapped(self):
        # The last pre-start line is normally force-snapped to consume every
        # remaining continuation word.  A vetoed box must escape that.
        flat = _flat_no_volpiano(
            cont_words=_W0 + _W1 + ["extraordinarium", "praecedentibus"],
            folio_words=_W2 + _W3,
        )
        labels, fused, ocr = self._pre_start_page(phantom_index=2)
        result = allocate_lines(flat, labels, ocr, fused_lines=fused)
        assert result.folio_start_line == 3
        assert result.skipped_labels == ["phantom"]
        assert result.manifest["phantom"] == ""

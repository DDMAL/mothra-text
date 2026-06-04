"""Tests for steps.nw_chant_allocator — Sub-plan 4b: allocate_lines."""

from steps.nw_chant_allocator import (
    Anchor,
    AllocationResult,
    ChantSpan,
    FlatTextData,
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
        flat = _flat(["word"])
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert result.manifest["line0"] == "word"

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

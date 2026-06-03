"""Tests for steps.nw_chant_allocator — Sub-plan 4a: build_flat_text_and_anchors."""

from unittest.mock import MagicMock

from steps.nw_chant_allocator import (
    Anchor,
    ChantSpan,
    FlatTextData,
    build_flat_text_and_anchors,
)


def _row(
    folio: str,
    sequence: str,
    text: str,
    volpiano: str = "",
    mode: str = "",
) -> dict:
    return {
        "folio": folio,
        "sequence": sequence,
        "fulltext_ms": text,
        "volpiano": volpiano,
        "mode": mode,
    }


class TestEmptyAndBasicCases:
    def test_empty_rows(self):
        result = build_flat_text_and_anchors([], "001r")
        assert result.words == []
        assert result.anchors == []
        assert result.chant_spans == []
        assert result.initial_pointer == 0
        assert result.continuation_words == []

    def test_no_rows_for_folio(self):
        rows = [_row("002r", "1", "alleluia dominus")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == []
        assert result.chant_spans == []

    def test_single_chant_no_volpiano(self):
        rows = [_row("001r", "1", "alleluia dominus laudate")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alleluia", "dominus", "laudate"]
        assert result.anchors == []
        assert len(result.chant_spans) == 1
        assert result.chant_spans[0] == ChantSpan(sequence=1, start_word=0, end_word=3)
        assert result.continuation_words == []

    def test_two_chants_words_concatenated(self):
        rows = [
            _row("001r", "1", "alleluia dominus"),
            _row("001r", "2", "kyrie eleison"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alleluia", "dominus", "kyrie", "eleison"]
        assert len(result.chant_spans) == 2
        assert result.chant_spans[0] == ChantSpan(sequence=1, start_word=0, end_word=2)
        assert result.chant_spans[1] == ChantSpan(sequence=2, start_word=2, end_word=4)
        assert result.anchors == []


class TestVolpianoAnchors:
    def test_within_chant_7_anchor(self):
        # Volpiano: 2 word-groups, break, 2 word-groups
        rows = [_row("001r", "1", "alleluia dominus laudate deum",
                     volpiano="9---a---b7---c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alleluia", "dominus", "laudate", "deum"]
        assert len(result.anchors) == 1
        assert result.anchors[0].anchor_type == "within_chant_7"
        # 2 word-groups in segment 0 → break after word index 2
        assert result.anchors[0].word_index == 2
        assert result.continuation_words == []

    def test_page_break_77_anchor_and_truncation(self):
        # Volpiano: 2 word-groups, 7 break, 1 word-group, 77 break, 1 word-group
        rows = [_row("001r", "1", "word1 word2 word3 word4",
                     volpiano="a---b7---c77---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        # Words before 77: word1, word2, word3
        assert result.words == ["word1", "word2", "word3"]
        # Words after 77 go to next folio
        assert result.continuation_words == ["word4"]
        assert len(result.anchors) == 2
        assert result.anchors[0].anchor_type == "within_chant_7"
        assert result.anchors[1].anchor_type == "page_break_77"
        # 77 break occurs after 3 words (segment 0: 2 words, segment 1: 1 word)
        assert result.anchors[1].word_index == 3

    def test_column_break_777_anchor(self):
        rows = [_row("001r", "1", "word1 word2 word3",
                     volpiano="a---b777---c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.anchors) == 1
        assert result.anchors[0].anchor_type == "column_break_777"
        assert result.anchors[0].word_index == 2
        assert result.continuation_words == []

    def test_multiple_7_breaks_cumulative_word_indices(self):
        # Three segments separated by two 7 breaks
        rows = [_row("001r", "1", "a b c d e f",
                     volpiano="x---y7---z---w7---v---u")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.anchors) == 2
        assert result.anchors[0].anchor_type == "within_chant_7"
        assert result.anchors[1].anchor_type == "within_chant_7"
        # Break positions are cumulative across the whole row
        assert result.anchors[0].word_index < result.anchors[1].word_index


class TestFiltering:
    def test_mode_star_rows_excluded(self):
        rows = [
            _row("001r", "1", "alleluia dominus"),
            _row("001r", "2", "invisible text", mode="*"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alleluia", "dominus"]
        assert len(result.chant_spans) == 1

    def test_folio_filter_excludes_other_folios(self):
        rows = [
            _row("001r", "1", "this folio"),
            _row("001v", "2", "other folio"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["this", "folio"]

    def test_rows_sorted_by_sequence(self):
        # Rows given in reverse sequence order; should be processed in order
        rows = [
            _row("001r", "3", "third"),
            _row("001r", "1", "first"),
            _row("001r", "2", "second"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["first", "second", "third"]


class TestLineOffset:
    def test_line_offset_zero_no_skip(self):
        rows = [_row("001r", "1", "a b c d", volpiano="x---y7---z---w")]
        result = build_flat_text_and_anchors(rows, "001r", line_offset=0)
        assert result.initial_pointer == 0

    def test_line_offset_sets_initial_pointer(self):
        # Two within_chant_7 breaks; line_offset=1 → skip past first break
        rows = [_row("001r", "1", "a b c d e f",
                     volpiano="x---y7---z---w7---v---u")]
        result = build_flat_text_and_anchors(rows, "001r", line_offset=1)
        within_anchors = [a for a in result.anchors if a.anchor_type == "within_chant_7"]
        assert result.initial_pointer == within_anchors[0].word_index

    def test_line_offset_beyond_anchors_clamps_to_zero(self):
        # line_offset=5 but only 1 within_chant_7 anchor → initial_pointer stays 0
        rows = [_row("001r", "1", "a b c d", volpiano="x---y7---z---w")]
        result = build_flat_text_and_anchors(rows, "001r", line_offset=5)
        assert result.initial_pointer == 0


class TestPrevFolioState:
    def test_prev_state_prepends_remaining_words(self):
        prev = MagicMock()
        prev.remaining_words = ["carry1", "carry2"]
        rows = [_row("002r", "3", "new word")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.words == ["carry1", "carry2", "new", "word"]

    def test_prev_state_chant_span_has_sequence_zero(self):
        prev = MagicMock()
        prev.remaining_words = ["carry"]
        rows = [_row("002r", "3", "new")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.chant_spans[0].sequence == 0
        assert result.chant_spans[0].start_word == 0
        assert result.chant_spans[0].end_word == 1

    def test_prev_state_empty_remaining_words_ignored(self):
        prev = MagicMock()
        prev.remaining_words = []
        rows = [_row("002r", "1", "word")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.words == ["word"]
        assert len(result.chant_spans) == 1
        assert result.chant_spans[0].sequence == 1

    def test_prev_state_sets_has_continuation(self):
        prev = MagicMock()
        prev.remaining_words = ["carry"]
        rows = [_row("002r", "1", "word")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.has_continuation is True

    def test_no_prev_state_has_continuation_false(self):
        rows = [_row("001r", "1", "Word")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.has_continuation is False


class TestInferContinuation:
    def test_infer_prepends_from_prev_folio_77_row(self):
        # 005v row has 77 in volpiano; target is 006r.
        # Post-77 words from 005v should be prepended to 006r's words.
        rows = [
            _row("005v", "5", "alpha beta gamma delta",
                 volpiano="a---b7---c77---d"),
            _row("006r", "6", "epsilon"),
        ]
        result = build_flat_text_and_anchors(rows, "006r")
        # "delta" is the post-77 word from 005v
        assert result.words[0] == "delta"
        assert result.has_continuation is True
        assert result.chant_spans[0].sequence == 0

    def test_infer_continuation_false_when_no_77_in_prev_rows(self):
        rows = [
            _row("005v", "5", "alpha beta gamma"),
            _row("006r", "6", "epsilon"),
        ]
        result = build_flat_text_and_anchors(rows, "006r")
        assert result.has_continuation is False
        assert result.words == ["epsilon"]

    def test_infer_disabled_skips_lookup(self):
        rows = [
            _row("005v", "5", "alpha beta gamma delta",
                 volpiano="a---b7---c77---d"),
            _row("006r", "6", "epsilon"),
        ]
        result = build_flat_text_and_anchors(
            rows, "006r", infer_continuation=False
        )
        assert result.has_continuation is False
        assert result.words == ["epsilon"]

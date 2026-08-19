"""Tests for steps.nw_chant_allocator — Sub-plan 4a: build_flat_text_and_anchors."""

from unittest.mock import MagicMock

from steps.nw_chant_allocator import (
    ChantSpan,
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


class TestMidWordBreaks:
    """Tests for MidWordBreak population in build_flat_text_and_anchors."""

    def test_clean_word_break_no_mid_word_breaks(self):
        # '7' at a clean word boundary (segment after 7 starts with '---')
        # → no MidWordBreak produced.
        rows = [_row("001r", "1", "alpha beta gamma delta",
                     volpiano="a---b7---c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.mid_word_breaks == []

    def test_no_volpiano_no_mid_word_breaks(self):
        rows = [_row("001r", "1", "alpha beta")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.mid_word_breaks == []

    def test_single_mid_word_break_one_syl_each_side(self):
        # Volpiano: "a---b7--c---d"
        # seg0 "a---b" → last group "b" → 1 syl left
        # seg1 "--c---d" → continuation "--c", body "c" → 1 syl right
        # anchor_word_index = 2 (beta is words[1])
        rows = [_row("001r", "1", "alpha beta gamma",
                     volpiano="a---b7--c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        assert mwb.anchor_word_index == 2
        assert mwb.syl_left == 1
        assert mwb.syl_right == 1

    def test_mid_word_break_multiple_syls_left(self):
        # Volpiano: "a---b--c--d7--e---f"
        # seg0 "a---b--c--d" → last group "b--c--d" → 3 syls left
        # seg1 "--e---f" → continuation "--e", body "e" → 1 syl right
        rows = [_row("001r", "1", "alpha beta gamma",
                     volpiano="a---b--c--d7--e---f")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        assert mwb.anchor_word_index == 2
        assert mwb.syl_left == 3
        assert mwb.syl_right == 1

    def test_mid_word_break_multiple_syls_right(self):
        # Volpiano: "a---b7--c--d---e"
        # seg0 "a---b" → last group "b" → 1 syl left
        # seg1 "--c--d---e" → continuation "--c--d", body "c--d" → 2 syls right
        rows = [_row("001r", "1", "alpha beta gamma",
                     volpiano="a---b7--c--d---e")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        assert mwb.anchor_word_index == 2
        assert mwb.syl_left == 1
        assert mwb.syl_right == 2

    def test_mid_word_break_with_clef_digit(self):
        # Volpiano: "a---b71--c---d" (clef '1' after '7')
        # After stripping leading '1', seg1 is "--c---d" → same as one-syl case.
        rows = [_row("001r", "1", "alpha beta gamma",
                     volpiano="a---b71--c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        assert mwb.anchor_word_index == 2
        assert mwb.syl_left == 1
        assert mwb.syl_right == 1

    def test_mid_word_break_word_index_offset_across_rows(self):
        # Row 1 contributes 2 words with no breaks.
        # Row 2 has a mid-word break at row-local anchor_word_index=2.
        # The global anchor_word_index should be 2+2=4.
        rows = [
            _row("001r", "1", "alpha beta",
                 volpiano="a---b"),
            _row("001r", "2", "gamma delta epsilon",
                 volpiano="x---y7--z---w"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        # "delta" = words[3] = words[4-1]
        assert mwb.anchor_word_index == 4
        assert mwb.syl_left == 1
        assert mwb.syl_right == 1

    def test_mid_word_break_dropped_after_77_continuation(self):
        # Mid-word break that falls after the 77 boundary belongs to the
        # next folio and must not appear in mid_word_breaks.
        # Volpiano: "a---b77--c---d" (77 break mid-word)
        rows = [_row("001r", "1", "alpha beta gamma",
                     volpiano="a---b77--c---d")]
        result = build_flat_text_and_anchors(rows, "001r")
        # The mid-word break is after the 77 → dropped.
        assert result.mid_word_breaks == []

    def test_mid_word_break_after_pipe_separator_stays_in_range(self):
        # Regression for mothra-text#45: a Cantus '|' phrase separator is
        # its own whitespace-delimited token in the raw text and is
        # stripped entirely by clean_text(), but the volpiano field's own
        # word-group count still allocates a position for it. Before the
        # fix, a mid-word break after the pipe landed one index past the
        # end of `words`, crashing allocate_lines with an IndexError.
        #
        # Real data: Cantus source 123672 (CH-Fco Ms. 2), folio 155v,
        # sequence 5, cantus_id 006291. The '|' sits between "eius" and
        # "Exaudi"; the trailing mid-word break splits the row's actual
        # last word, "misericordia" (1 syllable + 5 syllables).
        text = (
            "Civitatem istam tu circunda domine et angeli tui custodiant "
            "muros eius | Exaudi domine populum tuum cum misericordia"
        )
        volpiano = (
            "1---a--a--cd--df---fe-fgde--dc---d---f--ffE--cd7-def-gefe---d"
            "--defede--ed---f---fgh--hg-ge--g---fgf-fede--ed---d--de--d--"
            "defedc---d--fe-fgf7---fede--ed---3---a--cde--d---de--d--d---"
            "de--d--d---dcd-fgfe--fgfg--gf---ffE---dc7--d--fd-efe--d--de-"
            "fede--ed---4"
        )
        rows = [_row("155v", "5", text, volpiano=volpiano)]
        result = build_flat_text_and_anchors(rows, "155v")

        assert len(result.words) == 17
        assert result.words[-1] == "misericordia"
        assert len(result.mid_word_breaks) == 1
        mwb = result.mid_word_breaks[0]
        assert mwb.anchor_word_index <= len(result.words)
        assert mwb.anchor_word_index == 17
        assert mwb.syl_left == 1
        assert mwb.syl_right == 5


class TestPipeSeparatorOffsetEdgeCases:
    """Further coverage for the mothra-text#45 pipe-offset fix (d8e7b34),
    beyond the single mid-word-break regression case above."""

    def test_anchor_offset_with_single_pipe(self):
        # A '|' before a plain (non-mid-word) within_chant_7 anchor must
        # shift that anchor's index too, not just mid-word breaks.
        # Raw tokens: alpha(0) |(1) beta(2) gamma(3) -> 4 volpiano word
        # groups before the break; real words: alpha beta gamma (3).
        rows = [_row("001r", "1", "alpha | beta gamma",
                     volpiano="a---b---c---d7e")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alpha", "beta", "gamma"]
        assert len(result.anchors) == 1
        # Raw anchor index 4, minus 1 pipe before it, minus nothing after
        # clamp -> but only 3 real words exist, so clamp caps it at 3.
        assert result.anchors[0].word_index == 3
        assert result.anchors[0].anchor_type == "within_chant_7"

    def test_anchor_offset_with_multiple_pipes(self):
        # Two '|' tokens before the anchor; pipe_offsets must accumulate
        # across both rather than only correcting for one.
        # Raw tokens: alpha(0) |(1) beta(2) gamma(3) |(4) delta(5) epsilon(6)
        # Anchor after 5 raw tokens (alpha | beta gamma |) -> 2 pipes -> -2.
        rows = [_row("001r", "1", "alpha | beta gamma | delta epsilon",
                     volpiano="a---b---c---d---e7f---g")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alpha", "beta", "gamma", "delta", "epsilon"]
        assert len(result.anchors) == 1
        assert result.anchors[0].word_index == 3
        assert result.anchors[0].anchor_type == "within_chant_7"

    def test_continuation_start_offset_with_pipe(self):
        # A '|' before a page_break_77 must shift continuation_start too,
        # not just within_chant_7 anchors or mid-word breaks.
        # Raw tokens: alpha(0) |(1) beta(2) gamma(3) delta(4) epsilon(5)
        # -> 4 raw tokens (alpha | beta gamma) before the 77 break, minus
        # 1 pipe -> continuation_start=3 in real word space.
        rows = [_row("001r", "1", "alpha | beta gamma delta epsilon",
                     volpiano="a---b---c---d77e---f")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alpha", "beta", "gamma"]
        assert result.continuation_words == ["delta", "epsilon"]
        assert len(result.anchors) == 1
        assert result.anchors[0].word_index == 3
        assert result.anchors[0].anchor_type == "page_break_77"

    def test_carry_words_from_prev_folio_with_pipe(self):
        # The infer_continuation CSV-scan path (build_flat_text_and_anchors'
        # own "carry words" call site) must also apply the pipe-index
        # correction to the continuation words it prepends onto the next
        # folio's word list.
        rows = [
            _row("001r", "1", "alpha | beta gamma delta epsilon",
                 volpiano="a---b---c---d77e---f"),
            _row("002r", "2", "zeta"),
        ]
        result = build_flat_text_and_anchors(rows, "002r")
        assert result.has_continuation is True
        assert result.words == ["delta", "epsilon", "zeta"]
        assert result.chant_spans[0].sequence == 0
        assert result.chant_spans[0].end_word == 2

    def test_suffix_probe_words_strip_pipe_without_volpiano(self):
        # The suffix-probe call site passes raw (uncleaned) text with an
        # empty volpiano string; _parse_row_words_and_anchors must still
        # clean it internally so '|' never leaks into suffix_probe_words.
        rows = [
            _row("001r", "1", "alpha | beta gamma"),  # no volpiano -> no 77
            _row("002r", "2", "zeta"),
        ]
        result = build_flat_text_and_anchors(rows, "002r")
        assert result.has_continuation is False
        assert result.suffix_probe_words == ["alpha", "beta", "gamma"]

    def test_anchor_index_clamped_when_volpiano_overcounts_words(self):
        # If a row's volpiano implies more word-groups than the row has
        # raw tokens (a real-world data inconsistency), _to_real_index
        # clamps the raw index to len(raw_tokens) before subtracting the
        # pipe offset, so the anchor lands at a valid in-bounds position
        # (here, exactly at the end of `words`) instead of past it.
        rows = [_row("001r", "1", "alpha beta", volpiano="a---b---c---d7")]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["alpha", "beta"]
        assert len(result.anchors) == 1
        assert result.anchors[0].word_index == 2
        assert result.anchors[0].word_index <= len(result.words)

    def test_pipe_offset_composes_across_multiple_rows(self):
        # Row 1 (no volpiano) contributes 2 real words; row 2 has its own
        # pipe-corrected anchor. The global anchor index must be row 1's
        # real word count plus row 2's own corrected offset, not row 2's
        # raw-token-space offset added on top of row 1's word count.
        rows = [
            _row("001r", "1", "foo | bar"),
            _row("001r", "2", "alpha | beta gamma", volpiano="a---b---c7"),
        ]
        result = build_flat_text_and_anchors(rows, "001r")
        assert result.words == ["foo", "bar", "alpha", "beta", "gamma"]
        assert len(result.anchors) == 1
        assert result.anchors[0].word_index == 4
        assert result.anchors[0].anchor_type == "within_chant_7"

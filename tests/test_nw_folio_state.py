"""Tests for steps.nw_chant_allocator — Sub-plan 4c: FolioState and continuation."""

import json

from steps.nw_chant_allocator import (
    Anchor,
    AllocationResult,
    ChantSpan,
    FlatTextData,
    FolioState,
    ValidationFlag,
    allocate_lines,
    build_flat_text_and_anchors,
    build_folio_state,
    read_folio_state,
    write_folio_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(words, anchors=None, chant_spans=None, continuation_words=None,
          has_continuation=False):
    return FlatTextData(
        words=words,
        anchors=anchors or [],
        chant_spans=chant_spans or [ChantSpan(1, 0, len(words))],
        continuation_words=continuation_words or [],
        has_continuation=has_continuation,
    )


def _result(pointer_end, manifest=None):
    return AllocationResult(
        manifest=manifest or {},
        flags=[],
        text_pointer_end=pointer_end,
    )


def _row(folio, sequence, text, volpiano="", mode=""):
    return {
        "folio": folio,
        "sequence": sequence,
        "fulltext_ms": text,
        "volpiano": volpiano,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# build_folio_state
# ---------------------------------------------------------------------------

class TestBuildFolioState:
    def test_fully_consumed(self):
        # All flat_text words allocated, no continuation → fully_consumed=True.
        flat = _flat(["a", "b", "c"])
        result = _result(pointer_end=3)
        state = build_folio_state(flat, result, source_id=42, folio="001r")
        assert state.fully_consumed is True
        assert state.remaining_words == []
        assert state.folio == "001r"
        assert state.source_id == 42

    def test_unconsumed_tail_fallback(self):
        # NW didn't reach the end of flat_text → remaining = words[pointer_end:].
        flat = _flat(["a", "b", "c", "d", "e"])
        result = _result(pointer_end=3)
        state = build_folio_state(flat, result, source_id=None, folio="001r")
        assert state.fully_consumed is False
        assert state.remaining_words == ["d", "e"]

    def test_77_truncation_takes_priority(self):
        # continuation_words from a 77 break takes priority over the unconsumed tail.
        flat = _flat(
            words=["a", "b", "c"],
            continuation_words=["carry1", "carry2"],
        )
        # pointer_end == 3 means all flat_text.words consumed, but continuation_words exist.
        result = _result(pointer_end=3)
        state = build_folio_state(flat, result, source_id=1, folio="001v")
        assert state.remaining_words == ["carry1", "carry2"]
        assert state.fully_consumed is False

    def test_last_chant_sequence_from_span(self):
        spans = [
            ChantSpan(sequence=3, start_word=0, end_word=2),
            ChantSpan(sequence=5, start_word=2, end_word=5),
        ]
        flat = FlatTextData(words=["a","b","c","d","e"], anchors=[], chant_spans=spans)
        # pointer_end=4: inside span with sequence=5
        result = _result(pointer_end=4)
        state = build_folio_state(flat, result, source_id=None, folio="001r")
        assert state.last_chant_sequence == 5

    def test_empty_chant_spans_yields_sequence_zero(self):
        flat = FlatTextData(words=[], anchors=[], chant_spans=[])
        result = _result(pointer_end=0)
        state = build_folio_state(flat, result, source_id=None, folio="001r")
        assert state.last_chant_sequence == 0


# ---------------------------------------------------------------------------
# write_folio_state / read_folio_state
# ---------------------------------------------------------------------------

class TestWriteReadFolioState:
    def test_roundtrip(self, tmp_path):
        state = FolioState(
            source_id=123,
            folio="006r",
            last_chant_sequence=42,
            remaining_words=["alleluia", "dominus"],
            fully_consumed=False,
        )
        path = str(tmp_path / "state.json")
        write_folio_state(state, path)
        loaded = read_folio_state(path)
        assert loaded == state

    def test_written_file_is_valid_json(self, tmp_path):
        state = FolioState(
            source_id=None,
            folio="001r",
            last_chant_sequence=1,
            remaining_words=[],
            fully_consumed=True,
        )
        path = str(tmp_path / "state.json")
        write_folio_state(state, path)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["folio"] == "001r"
        assert d["fully_consumed"] is True


# ---------------------------------------------------------------------------
# build_flat_text_and_anchors with prev_folio_state
# ---------------------------------------------------------------------------

class TestPrevFolioStateIntegration:
    def test_prev_state_prepends_words(self):
        prev = FolioState(
            source_id=1, folio="001r",
            last_chant_sequence=3,
            remaining_words=["carry1", "carry2"],
            fully_consumed=False,
        )
        rows = [_row("002r", "4", "new word")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.words == ["carry1", "carry2", "new", "word"]

    def test_prev_state_chant_span_is_sequence_zero(self):
        prev = FolioState(
            source_id=1, folio="001r",
            last_chant_sequence=3,
            remaining_words=["carry"],
            fully_consumed=False,
        )
        rows = [_row("002r", "4", "new")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.chant_spans[0].sequence == 0
        assert result.chant_spans[0].start_word == 0
        assert result.chant_spans[0].end_word == 1  # 1 carry word

    def test_prev_state_fully_consumed_no_prepend(self):
        prev = FolioState(
            source_id=1, folio="001r",
            last_chant_sequence=3,
            remaining_words=[],
            fully_consumed=True,
        )
        rows = [_row("002r", "4", "word")]
        result = build_flat_text_and_anchors(rows, "002r", prev_folio_state=prev)
        assert result.words == ["word"]
        assert result.chant_spans[0].sequence == 4  # no sequence=0 span


# ---------------------------------------------------------------------------
# continuation_missing flag in allocate_lines
# ---------------------------------------------------------------------------

class TestContinuationMissingFlag:
    def test_emitted_when_first_word_lowercase_no_prev_state(self):
        flat = _flat(["kyrie", "eleison"])
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert any(f.flag_type == "continuation_missing" for f in result.flags)

    def test_not_emitted_when_prev_state_provided(self):
        flat = _flat(["kyrie", "eleison"], has_continuation=True)
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert not any(f.flag_type == "continuation_missing" for f in result.flags)

    def test_not_emitted_when_first_word_is_uppercase(self):
        flat = _flat(["Alleluia", "dominus"])
        result = allocate_lines(flat, ["line0"], {"line0": ""})
        assert not any(f.flag_type == "continuation_missing" for f in result.flags)

    def test_not_emitted_when_flat_text_empty(self):
        flat = _flat([])
        result = allocate_lines(flat, [], {})
        assert not any(f.flag_type == "continuation_missing" for f in result.flags)

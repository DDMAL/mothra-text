"""Tests for steps.gt_manifest."""

import logging
from unittest.mock import MagicMock

import pytest

from steps.gt_manifest import (
    build_page_manifest,
    clean_text,
    make_manifest_lookup,
    split_by_volpiano,
)


# ---------------------------------------------------------------------------
# split_by_volpiano
# ---------------------------------------------------------------------------

class TestSplitByVolpiano:
    def test_two_line_chant(self):
        # Volpiano: word-word | line-break | word-word
        result = split_by_volpiano("alpha beta gamma delta", "ab---cd7ef---gh")
        assert result == ["alpha beta", "gamma delta"]

    def test_page_break_77_splits_line(self):
        result = split_by_volpiano("alpha beta gamma delta", "ab---cd77ef---gh")
        assert result == ["alpha beta", "gamma delta"]

    def test_column_break_777_splits_line(self):
        result = split_by_volpiano("alpha beta gamma delta", "ab---cd777ef---gh")
        assert result == ["alpha beta", "gamma delta"]

    def test_no_volpiano_returns_full_text(self):
        result = split_by_volpiano("alleluia dominus", "")
        assert result == ["alleluia dominus"]

    def test_empty_text_returns_empty_list(self):
        assert split_by_volpiano("", "ab---cd7ef") == []

    def test_single_line_no_break(self):
        result = split_by_volpiano("dominus omnipotens", "ab---cd---ef")
        assert result == ["dominus omnipotens"]

    def test_volpiano_word_count_exceeds_text_words(self):
        # 3 word groups in segment 1, only 2 text words — must not crash
        result = split_by_volpiano("alpha beta", "ab---cd---ef7gh")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)
        # The two text words end up on the first line; no crash
        assert "alpha" in result[0]
        assert "beta" in result[0]

    def test_trailing_7_ignored(self):
        # Trailing break produces an empty segment which should be skipped
        result = split_by_volpiano("alpha beta", "ab---cd7")
        assert result == ["alpha beta"]

    def test_leading_clef_character(self):
        # '9' is a clef/custos in Volpiano; should not add a spurious word
        result = split_by_volpiano("alleluia", "9---ab")
        # The '9' group has a non-hyphen character — it counts as a word group.
        # '9---ab'.split('---') = ['9', 'ab'] → 2 groups → takes 2 words.
        # But text only has 1 word, so result has 1 element.
        assert len(result) == 1

    def test_volpiano_shorter_than_text_appends_tail(self):
        # Volpiano covers only 2 words; remaining text appended to last fragment
        result = split_by_volpiano("alpha beta gamma", "ab---cd")
        assert result == ["alpha beta gamma"]

    def test_three_lines(self):
        result = split_by_volpiano("a b c d e f", "x7y---z7w---v---u")
        assert len(result) == 3
        assert result[0] == "a"
        assert result[1] == "b c"
        assert result[2] == "d e f"


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_pipes(self):
        assert clean_text("hello | world") == "hello world"

    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_like_empty(self):
        assert clean_text("") == ""

    def test_pipe_with_no_spaces(self):
        assert clean_text("hello|world") == "hello|world"  # only ` | ` is stripped


# ---------------------------------------------------------------------------
# build_page_manifest
# ---------------------------------------------------------------------------

def _make_rows(folio_chants: list[tuple[str, str, str]], folio: str = "006r") -> list[dict]:
    """Create minimal CSV rows: [(text, volpiano, sequence), ...]"""
    rows = []
    for i, (text, volpiano, seq) in enumerate(folio_chants):
        rows.append({
            "folio": folio,
            "sequence": seq or str(i + 1),
            "fulltext_standardized": text,
            "fulltext_ms": "",
            "volpiano": volpiano,
        })
    return rows


class TestBuildPageManifest:
    def test_two_chants_two_lines_each(self):
        rows = _make_rows([
            ("alpha beta gamma delta", "ab---cd7ef---gh", "1"),
            ("iota kappa lambda mu",   "ij---kl7mn---op", "2"),
        ])
        labels = [
            "page_006r_region0_line0",
            "page_006r_region0_line1",
            "page_006r_region0_line2",
            "page_006r_region0_line3",
        ]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest == {
            "page_006r_region0_line0": "alpha beta",
            "page_006r_region0_line1": "gamma delta",
            "page_006r_region0_line2": "iota kappa",
            "page_006r_region0_line3": "lambda mu",
        }

    def test_filters_other_folios(self):
        rows = _make_rows([("unrelated text", "", "1")], folio="005v")
        rows += _make_rows([("alpha beta", "ab---cd", "1")], folio="006r")
        labels = ["page_006r_region0_line0"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest == {"page_006r_region0_line0": "alpha beta"}

    def test_more_cantus_lines_than_nodes_warns(self, caplog):
        # 4 Cantus lines, only 2 node labels — extra Cantus lines dropped
        rows = _make_rows([("a b c d", "a7b7c7d", "1")])
        labels = ["page_region0_line0", "page_region0_line1"]
        with caplog.at_level(logging.WARNING, logger="steps.gt_manifest"):
            manifest = build_page_manifest(rows, "006r", labels)
        assert len(manifest) == 2
        assert "WARNING" in caplog.text or caplog.records  # warning was logged

    def test_more_nodes_than_cantus_lines(self):
        # 2 Cantus lines, 4 node labels — extra labels absent from manifest
        rows = _make_rows([("alpha beta", "ab7cd", "1")])
        labels = [
            "page_region0_line0",
            "page_region0_line1",
            "page_region0_line2",
            "page_region0_line3",
        ]
        manifest = build_page_manifest(rows, "006r", labels)
        # Only the first 2 labels are matched
        assert len(manifest) == 2
        assert "page_region0_line2" not in manifest
        assert "page_region0_line3" not in manifest

    def test_empty_folio_returns_empty_dict(self, caplog):
        rows = _make_rows([("alpha", "ab", "1")], folio="005v")
        with caplog.at_level(logging.WARNING, logger="steps.gt_manifest"):
            manifest = build_page_manifest(rows, "006r", ["page_region0_line0"])
        assert manifest == {}

    def test_node_labels_sorted_numerically(self):
        # line10 must come after line9, not before line1
        rows = _make_rows([
            ("word " * 11, "7".join(["a"] * 11), "1"),  # 11 lines
        ])
        labels = [f"page_region0_line{i}" for i in range(11)]
        manifest = build_page_manifest(rows, "006r", labels)
        assert "page_region0_line10" in manifest
        # line10 should map to the 11th word
        assert manifest["page_region0_line10"] == "word"

    def test_chant_with_no_text_is_skipped(self):
        rows = _make_rows([
            ("", "ab7cd", "1"),        # no text → skipped
            ("alpha beta", "ef7gh", "2"),
        ])
        labels = ["page_region0_line0", "page_region0_line1"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest == {
            "page_region0_line0": "alpha",
            "page_region0_line1": "beta",
        }

    def test_sequence_ordering(self):
        # Rows given out of sequence order — must be sorted by sequence number
        rows = [
            {"folio": "006r", "sequence": "2", "fulltext_standardized": "gamma delta",
             "fulltext_ms": "", "volpiano": "gh---ij"},
            {"folio": "006r", "sequence": "1", "fulltext_standardized": "alpha beta",
             "fulltext_ms": "", "volpiano": "ab---cd"},
        ]
        labels = ["page_region0_line0", "page_region0_line1"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest["page_region0_line0"] == "alpha beta"
        assert manifest["page_region0_line1"] == "gamma delta"

    def test_falls_back_to_fulltext_ms_when_standardized_absent(self):
        rows = [{"folio": "006r", "sequence": "1",
                 "fulltext_standardized": "", "fulltext_ms": "alpha beta",
                 "volpiano": "ab---cd"}]
        labels = ["page_region0_line0"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest["page_region0_line0"] == "alpha beta"


# ---------------------------------------------------------------------------
# make_manifest_lookup
# ---------------------------------------------------------------------------

class TestMakeManifestLookup:
    def test_returns_text_for_known_label(self):
        manifest = {"page_region0_line0": "dominus omnipotens"}
        lookup = make_manifest_lookup(manifest)
        node = MagicMock()
        node.label = "page_region0_line0"
        assert lookup(node) == "dominus omnipotens"

    def test_returns_none_for_unknown_label(self):
        lookup = make_manifest_lookup({"page_region0_line0": "text"})
        node = MagicMock()
        node.label = "page_region0_line99"
        assert lookup(node) is None

    def test_empty_manifest_always_returns_none(self):
        lookup = make_manifest_lookup({})
        node = MagicMock()
        node.label = "anything"
        assert lookup(node) is None

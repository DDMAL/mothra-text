"""Tests for steps.gt_manifest."""

import codecs
import logging
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from steps.gt_manifest import (
    build_page_manifest,
    clean_text,
    fetch_cantus_csv,
    load_local_csv,
    make_manifest_lookup,
    make_output_stem,
    split_by_volpiano,
)


# ---------------------------------------------------------------------------
# make_output_stem
# ---------------------------------------------------------------------------

def _make_csv_rows(holding_institution: str, shelfmark: str) -> list[dict]:
    return [{"holding_institution": holding_institution, "shelfmark": shelfmark}]


class TestMakeOutputStem:
    def test_basic_rism_extraction(self):
        rows = _make_csv_rows("Einsiedeln, Stiftsbibliothek (CH-E)", "611")
        assert make_output_stem(rows, "001r") == "CH-E_611_001r"

    def test_institution_with_space_in_rism_code(self):
        # RISM codes do not contain spaces, but spaces within the full name
        # before the parenthetical should not bleed into the code.
        rows = _make_csv_rows("Graz, Universitätsbibliothek (A-Gu)", "29 (olim 38/8 f.)")
        assert make_output_stem(rows, "001v") == "A-Gu_29_001v"

    def test_shelfmark_parenthetical_stripped(self):
        rows = _make_csv_rows("Somewhere, Library (XX-Xx)", "29 (olim 38/8 f.)")
        assert make_output_stem(rows, "005r") == "XX-Xx_29_005r"

    def test_shelfmark_with_internal_space_replaced(self):
        # "Ms. 2" → "Ms._2" (space → underscore, period kept)
        rows = _make_csv_rows("Fribourg, Bibliothèque des Cordeliers (CH-Fco)", "Ms. 2")
        assert make_output_stem(rows, "108v") == "CH-Fco_Ms._2_108v"

    def test_shelfmark_no_parenthetical(self):
        rows = _make_csv_rows("Paris, Bibliothèque nationale de France (F-Pn)", "Latin 17436")
        assert make_output_stem(rows, "029r") == "F-Pn_Latin_17436_029r"

    def test_no_parenthetical_in_institution_uses_full_value(self):
        # Fallback: if no parenthetical, use the whole holding_institution string.
        rows = _make_csv_rows("UnknownLib", "99")
        assert make_output_stem(rows, "001r") == "UnknownLib_99_001r"

    def test_uses_first_row_only(self):
        # Even if multiple rows are present, only the first is used for metadata.
        rows = [
            {"holding_institution": "A, Library (AA-Aa)", "shelfmark": "1"},
            {"holding_institution": "B, Library (BB-Bb)", "shelfmark": "2"},
        ]
        assert make_output_stem(rows, "001r") == "AA-Aa_1_001r"

    def test_folio_appended_verbatim(self):
        rows = _make_csv_rows("City, Library (ZZ-Zz)", "123")
        assert make_output_stem(rows, "012v") == "ZZ-Zz_123_012v"


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

    def test_clef_digits_not_counted_as_word_groups(self):
        # Volpiano digits (1–9) are clef/custos markers, not note groups.
        # A leading '1' clef must not steal a word from the text.
        # '1---ab---cd' has 2 real note groups → 2 words per line.
        # Without the fix, '1' would count as a 3rd group, shifting
        # all subsequent line assignments by one word.
        result = split_by_volpiano("alpha beta gamma delta", "1---ab---cd7ef---gh")
        assert result == ["alpha beta", "gamma delta"]

    def test_mid_segment_clef_not_counted(self):
        # Clef changes (3, 4) mid-volpiano segment must also be ignored.
        result = split_by_volpiano("alpha beta gamma", "ab---3---cd---ef")
        assert result == ["alpha beta gamma"]

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

    def test_mid_word_line_break_does_not_consume_word(self):
        # '7' falls mid-word: segment 2 starts with '--' (syllable boundary).
        # 'ab---cd7--ef---gh': seg1 has 2 groups (ab, cd) → 2 words;
        # seg2 starts mid-word (--ef), so its continuation group is excluded
        # → seg2 contributes 1 new word (gh).
        result = split_by_volpiano("alpha beta gamma", "ab---cd7--ef---gh")
        assert result == ["alpha beta", "gamma"]

    def test_mid_word_line_break_with_clef_digit(self):
        # '7' followed by a clef digit, then mid-word continuation.
        # After stripping '1', remainder '--ef---gh' starts with '--' → mid-word.
        result = split_by_volpiano("alpha beta gamma", "ab---cd71--ef---gh")
        assert result == ["alpha beta", "gamma"]

    def test_clean_word_break_at_7_not_penalized(self):
        # '7' at a clean word boundary: segment 2 starts with '---'.
        # Both segments contribute their full word count.
        result = split_by_volpiano("alpha beta gamma delta", "ab---cd7---ef---gh")
        assert result == ["alpha beta", "gamma delta"]


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

    def test_prefers_fulltext_ms_over_standardized(self):
        # manuscript spelling takes priority; standardized is the fallback
        rows = [{"folio": "006r", "sequence": "1",
                 "fulltext_standardized": "standardized beta",
                 "fulltext_ms": "alpha beta",
                 "volpiano": "ab---cd"}]
        labels = ["page_region0_line0"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest["page_region0_line0"] == "alpha beta"

    def test_falls_back_to_standardized_when_ms_absent(self):
        # fulltext_standardized used only when fulltext_ms is empty
        rows = [{"folio": "006r", "sequence": "1",
                 "fulltext_standardized": "standardized beta",
                 "fulltext_ms": "",
                 "volpiano": "ab---cd"}]
        labels = ["page_region0_line0"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest["page_region0_line0"] == "standardized beta"


    def test_mode_star_rows_excluded(self):
        rows = [
            {"folio": "006r", "sequence": "1", "fulltext_standardized": "alpha beta",
             "fulltext_ms": "", "volpiano": "ab7cd", "mode": "*"},
            {"folio": "006r", "sequence": "2", "fulltext_standardized": "gamma delta",
             "fulltext_ms": "", "volpiano": "ef7gh", "mode": "1"},
        ]
        labels = ["page_region0_line0", "page_region0_line1"]
        manifest = build_page_manifest(rows, "006r", labels)
        assert manifest == {
            "page_region0_line0": "gamma",
            "page_region0_line1": "delta",
        }


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


def _mock_urlopen_response(content_bytes):
    mock_response = MagicMock()
    mock_response.read.return_value = content_bytes
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_response
    mock_cm.__exit__.return_value = False
    return mock_cm


class TestFetchCantusCsv:
    @pytest.fixture(autouse=True)
    def _clear_lru_cache(self):
        # fetch_cantus_csv is @lru_cache'd by source_id -- clear before AND
        # after each test so results (or a mocked urlopen) from one test
        # can't leak into another via the cache.
        fetch_cantus_csv.cache_clear()
        yield
        fetch_cantus_csv.cache_clear()

    def test_parses_csv_response_into_dicts(self):
        csv_bytes = b"folio,sequence,fulltext_ms\n001r,1,alleluia dominus\n"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen_response(csv_bytes),
        ) as mock_urlopen:
            rows = fetch_cantus_csv(999001)
        assert rows == [
            {"folio": "001r", "sequence": "1",
             "fulltext_ms": "alleluia dominus"}
        ]
        requested_url = mock_urlopen.call_args.args[0].full_url
        assert "999001" in requested_url

    def test_handles_utf8_bom(self):
        csv_bytes = codecs.BOM_UTF8 + b"folio\n001r\n"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen_response(csv_bytes),
        ):
            rows = fetch_cantus_csv(999002)
        assert rows == [{"folio": "001r"}]

    def test_http_error_propagates(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "url", 404, "Not Found", {}, None
            ),
        ):
            with pytest.raises(urllib.error.HTTPError):
                fetch_cantus_csv(999003)

    def test_result_is_cached_by_source_id(self):
        csv_bytes = b"folio\n001r\n"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen_response(csv_bytes),
        ) as mock_urlopen:
            fetch_cantus_csv(999004)
            fetch_cantus_csv(999004)
        assert mock_urlopen.call_count == 1


class TestLoadLocalCsv:
    def test_parses_csv_file_into_dicts(self, tmp_path):
        path = tmp_path / "source.csv"
        path.write_text("folio,sequence\n001r,1\n002r,2\n", encoding="utf-8")
        rows = load_local_csv(path)
        assert rows == [
            {"folio": "001r", "sequence": "1"},
            {"folio": "002r", "sequence": "2"},
        ]

    def test_handles_utf8_bom(self, tmp_path):
        path = tmp_path / "source_bom.csv"
        path.write_bytes(codecs.BOM_UTF8 + b"folio\n001r\n")
        rows = load_local_csv(path)
        assert rows == [{"folio": "001r"}]

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "source.csv"
        path.write_text("folio\n001r\n", encoding="utf-8")
        rows = load_local_csv(str(path))
        assert rows == [{"folio": "001r"}]

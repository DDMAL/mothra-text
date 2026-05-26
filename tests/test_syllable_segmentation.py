"""Tests for steps.syllable_segmentation."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np

from htrflow.results import TEXT_RESULT_KEY

from steps.syllable_segmentation import (
    SyllableSegmentation,
    _syllable_segmentation,
    normalize_word_text,
    syllabify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(text="alleluia", width=300, height=30):
    """Mock SegmentNode with the attributes used by syllable segmentation."""
    node = MagicMock()
    node.text = text
    node.width = width
    node.height = height
    node.mask = np.ones((height, width), dtype=np.uint8)
    return node


def make_collection(nodes):
    """Mock Collection whose active_leaves() yields the given nodes."""
    col = MagicMock()
    col.active_leaves.return_value = iter(nodes)
    return col


def segment_texts(result):
    """Extract top-candidate text from each segment in a Result."""
    return [
        seg.data[TEXT_RESULT_KEY].top_candidate()
        for seg in result.segments
    ]


# ---------------------------------------------------------------------------
# normalize_word_text
# ---------------------------------------------------------------------------

class TestNormalizeWordText:
    def test_plain_ascii_latin_word(self):
        assert normalize_word_text("dominus") == "dominus"

    def test_lowercases_output(self):
        assert normalize_word_text("DOMINUS") == "dominus"

    def test_word_with_macron(self):
        # d̄ = d + combining macron (U+0304); NFKD keeps 'd', strips macron
        assert normalize_word_text("d̄ns") == "dns"

    def test_ae_ligature(self):
        # æ (U+00E6): NFKD does not decompose to ASCII; encode drops it
        result = normalize_word_text("æterne")
        assert result == "terne"

    def test_oe_ligature(self):
        # œ (U+0153): same — dropped by encode("ascii","ignore")
        result = normalize_word_text("œuvre")
        assert result == "uvre"

    def test_accented_vowel(self):
        # é (U+00E9) → NFKD → e + combining accent → encode ASCII → 'e'
        assert normalize_word_text("éterne") == "eterne"

    def test_entirely_non_alphabetic_after_normalization(self):
        assert normalize_word_text("123---") == ""

    def test_empty_string_input(self):
        assert normalize_word_text("") == ""

    def test_strips_hyphens_and_punctuation(self):
        assert normalize_word_text("do-mi-nus.") == "dominus"

    def test_logs_warning_for_non_ascii(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="steps.syllable_segmentation"
        ):
            normalize_word_text("dōminus")  # ō = U+014D
        assert "Non-ASCII" in caplog.text

    def test_no_warning_for_plain_ascii(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="steps.syllable_segmentation"
        ):
            normalize_word_text("dominus")
        assert "Non-ASCII" not in caplog.text

    def test_warning_includes_original_text(self, caplog):
        original = "dōminus"
        with caplog.at_level(
            logging.WARNING, logger="steps.syllable_segmentation"
        ):
            normalize_word_text(original)
        assert original in caplog.text


# ---------------------------------------------------------------------------
# syllabify
# ---------------------------------------------------------------------------

class TestSyllabify:
    def test_normal_multisyllable_latin_word(self):
        # dominus → do-mi-nus
        result = syllabify("dominus")
        assert result == ["do-", "mi-", "nus"]

    def test_three_syllable_word_reassembles(self):
        result = syllabify("alleluia")
        assert len(result) >= 2
        reassembled = "".join(s.rstrip("-") for s in result)
        assert reassembled == "alleluia"

    def test_single_syllable_word(self):
        # "lux" has one syllable → syllabify_word returns [] → ["lux"]
        result = syllabify("lux")
        assert result == ["lux"]

    def test_empty_string_input_returns_single_element(self, caplog):
        with caplog.at_level(
            logging.ERROR, logger="steps.syllable_segmentation"
        ):
            result = syllabify("")
        assert result == [""]

    def test_empty_string_logs_error(self, caplog):
        with caplog.at_level(
            logging.ERROR, logger="steps.syllable_segmentation"
        ):
            syllabify("")
        assert "empty string" in caplog.text

    def test_latin_error_returns_single_element(self, caplog):
        from volpiano_display_utilities.latin_word_syllabification import (
            LatinError,
        )
        with patch(
            "steps.syllable_segmentation.syllabify_word",
            side_effect=LatinError("forced error"),
        ):
            with caplog.at_level(
                logging.ERROR, logger="steps.syllable_segmentation"
            ):
                result = syllabify("dominus")
        assert len(result) == 1

    def test_latin_error_logs_offending_text(self, caplog):
        from volpiano_display_utilities.latin_word_syllabification import (
            LatinError,
        )
        with patch(
            "steps.syllable_segmentation.syllabify_word",
            side_effect=LatinError("forced error"),
        ):
            with caplog.at_level(
                logging.ERROR, logger="steps.syllable_segmentation"
            ):
                syllabify("dominus")
        assert "LatinError" in caplog.text

    def test_non_ascii_input_is_normalized_first(self):
        result = syllabify("éterne")
        assert isinstance(result, list)
        assert len(result) >= 1
        reassembled = "".join(s.rstrip("-") for s in result)
        assert reassembled == "eterne"

    def test_returns_list(self):
        assert isinstance(syllabify("dominus"), list)

    def test_syllables_reassemble_to_normalized_word(self):
        result = syllabify("omnipotens")
        reassembled = "".join(s.rstrip("-") for s in result)
        assert reassembled == "omnipotens"


# ---------------------------------------------------------------------------
# _syllable_segmentation
# ---------------------------------------------------------------------------

class TestSyllableSegmentationFunction:
    def test_produces_at_least_one_segment(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) >= 1

    def test_segment_count_matches_syllable_count(self):
        # dominus → 3 syllables → 3 segments
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) == len(syllabify("dominus"))

    def test_segment_texts_match_syllabification(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert segment_texts(result) == ["do-", "mi-", "nus"]

    def test_single_syllable_word_produces_one_segment(self):
        node = make_node(text="lux", width=100, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) == 1

    def test_single_segment_spans_full_width(self):
        node = make_node(text="lux", width=100, height=30)
        result = _syllable_segmentation(node)
        bbox = result.segments[0].bbox
        assert bbox.xmin == 0
        assert bbox.xmax == 100

    def test_none_text_produces_one_segment(self):
        node = make_node(text=None, width=200, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) >= 1

    def test_empty_text_produces_one_segment(self):
        node = make_node(text="", width=200, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) >= 1


# ---------------------------------------------------------------------------
# Integration test: full coverage of word box, no gaps or overlaps
# ---------------------------------------------------------------------------

class TestSyllableSegmentationIntegration:
    """Verify that syllable bounding boxes tile the word box exactly."""

    def test_boxes_start_at_zero(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert result.segments[0].bbox.xmin == 0

    def test_boxes_end_at_node_width(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert result.segments[-1].bbox.xmax == 300

    def test_boxes_are_contiguous_no_gaps(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        bboxes = [seg.bbox for seg in result.segments]
        for prev, nxt in zip(bboxes[:-1], bboxes[1:]):
            assert prev.xmax == nxt.xmin, (
                f"Gap between syllable boxes: {prev.xmax} != {nxt.xmin}"
            )

    def test_boxes_are_left_to_right_ordered(self):
        node = make_node(text="alleluia", width=240, height=30)
        result = _syllable_segmentation(node)
        bboxes = [seg.bbox for seg in result.segments]
        for prev, nxt in zip(bboxes[:-1], bboxes[1:]):
            assert prev.xmin < nxt.xmin

    def test_texts_match_syllabification_of_word(self):
        # Two-syllable word: cantus → can-tus
        node = make_node(text="cantus", width=200, height=30)
        result = _syllable_segmentation(node)
        assert segment_texts(result) == syllabify("cantus")

    def test_full_coverage_with_irregular_width(self):
        """Width not evenly divisible; last box absorbs the remainder."""
        node = make_node(text="dominus", width=100, height=30)
        result = _syllable_segmentation(node)
        assert result.segments[0].bbox.xmin == 0
        assert result.segments[-1].bbox.xmax == 100

    def test_three_syllable_word_produces_three_children(self):
        node = make_node(text="dominus", width=300, height=30)
        result = _syllable_segmentation(node)
        assert len(result.segments) == 3


# ---------------------------------------------------------------------------
# SyllableSegmentation.run()
# ---------------------------------------------------------------------------

class TestSyllableSegmentationStep:
    def test_run_passes_one_result_per_node_to_update(self):
        nodes = [make_node(text="dominus"), make_node(text="alleluia")]
        collection = make_collection(nodes)
        step = SyllableSegmentation()

        step.run(collection)

        collection.update.assert_called_once()
        (passed_results,) = collection.update.call_args.args
        assert len(passed_results) == 2

    def test_run_returns_collection(self):
        nodes = [make_node(text="dominus")]
        collection = make_collection(nodes)
        step = SyllableSegmentation()

        result = step.run(collection)
        assert result is collection

    def test_run_calls_update_with_results_list(self):
        nodes = [
            make_node(text="lux"),
            make_node(text="et"),
            make_node(text="pax"),
        ]
        collection = make_collection(nodes)
        step = SyllableSegmentation()

        step.run(collection)

        collection.update.assert_called_once()
        (passed_results,) = collection.update.call_args.args
        assert len(passed_results) == len(nodes)

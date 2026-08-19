"""Tests for steps.ground_truth_word_segmentation."""

import logging
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from steps.ground_truth_word_segmentation import (
    GroundTruthWordSegmentation,
    _bbox_word_segmentation,
    _fallback_word_segmentation,
    _ground_truth_word_segmentation,
)

TEXT_RESULT_KEY = "text_result"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(text="recognised text", width=300, height=30):
    """Mock SegmentNode with the attributes used by word segmentation."""
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


# ---------------------------------------------------------------------------
# _ground_truth_word_segmentation
# ---------------------------------------------------------------------------

class TestGroundTruthWordSegmentationFunction:
    def test_returns_none_when_gt_lookup_returns_none(self):
        node = make_node()
        assert _ground_truth_word_segmentation(node, lambda _: None) is None

    def test_returns_none_when_gt_lookup_returns_empty_string(self):
        node = make_node()
        assert _ground_truth_word_segmentation(node, lambda _: "") is None

    def test_single_word_returns_result(self):
        node = make_node(width=200, height=30)
        result = _ground_truth_word_segmentation(node, lambda _: "alleluia")
        assert result is not None

    def test_multi_word_returns_correct_segment_count(self):
        node = make_node(width=300, height=30)
        result = _ground_truth_word_segmentation(node, lambda _: "dominus omnipotens")
        assert result is not None
        assert len(result.segments) == 2

    def test_uses_gt_text_not_node_text(self):
        # node.text intentionally differs from the ground-truth lookup
        node = make_node(text="wrong recognised text", width=400, height=30)
        result = _ground_truth_word_segmentation(node, lambda _: "dominus omnipotens")
        assert result is not None
        words = [seg.data[TEXT_RESULT_KEY].top_candidate() for seg in result.segments]
        assert words == ["dominus", "omnipotens"]


# ---------------------------------------------------------------------------
# GroundTruthWordSegmentation.run()
# ---------------------------------------------------------------------------

class TestGroundTruthWordSegmentationStep:
    def test_run_passes_one_result_per_node_to_update(self):
        nodes = [make_node(), make_node()]
        collection = make_collection(nodes)
        step = GroundTruthWordSegmentation(gt_lookup=lambda _: "dominus omnipotens")

        mock_result = MagicMock()
        with patch(
            "steps.ground_truth_word_segmentation._ground_truth_word_segmentation",
            return_value=mock_result,
        ):
            step.run(collection)

        collection.update.assert_called_once()
        (passed_results,) = collection.update.call_args.args
        assert len(passed_results) == len(nodes)

    def test_run_falls_back_to_simple_when_gt_is_none(self):
        node = make_node()
        collection = make_collection([node])
        step = GroundTruthWordSegmentation(gt_lookup=lambda _: None)

        fallback = MagicMock()
        with patch(
            "steps.ground_truth_word_segmentation._fallback_word_segmentation",
            return_value=fallback,
        ) as mock_fallback:
            step.run(collection)

        mock_fallback.assert_called_once_with(node)
        (passed_results,) = collection.update.call_args.args
        assert passed_results == [fallback]

    def test_run_logs_warning_when_gt_is_none(self, caplog):
        node = make_node()
        collection = make_collection([node])
        step = GroundTruthWordSegmentation(gt_lookup=lambda _: None)

        with patch(
            "steps.ground_truth_word_segmentation._fallback_word_segmentation",
            return_value=MagicMock(),
        ):
            with caplog.at_level(
                logging.WARNING,
                logger="steps.ground_truth_word_segmentation",
            ):
                step.run(collection)

        assert "No ground truth" in caplog.text

    def test_run_falls_back_when_gt_returns_empty_string(self):
        node = make_node()
        collection = make_collection([node])
        step = GroundTruthWordSegmentation(gt_lookup=lambda _: "")

        fallback = MagicMock()
        with patch(
            "steps.ground_truth_word_segmentation._fallback_word_segmentation",
            return_value=fallback,
        ) as mock_fallback:
            step.run(collection)

        mock_fallback.assert_called_once_with(node)


# ---------------------------------------------------------------------------
# _bbox_word_segmentation / _fallback_word_segmentation
#
# The step-level tests above always mock _fallback_word_segmentation out,
# so its actual geometry math and empty-text guard were never executed by
# the suite. These call the real functions directly.
# ---------------------------------------------------------------------------

class TestBboxWordSegmentation:
    def test_segment_count_matches_word_count(self):
        node = make_node(width=100, height=30)
        result = _bbox_word_segmentation(node, ["ab", "cd"], "ab cd")
        assert len(result.segments) == 2

    def test_words_and_bboxes_are_left_to_right_in_order(self):
        node = make_node(width=100, height=30)
        result = _bbox_word_segmentation(node, ["ab", "cd"], "ab cd")
        words = [
            seg.data[TEXT_RESULT_KEY].top_candidate()
            for seg in result.segments
        ]
        assert words == ["ab", "cd"]
        # chars=5, pixels_per_char=max(1, 100//5)=20.
        # "ab": x2=min(0+20*3,100)=60 -> bbox (0,0,60,30).
        # "cd": x1=60, x2=min(60+20*3,100)=100 -> bbox (60,0,100,30).
        first, second = result.segments
        assert (first.bbox.xmin, first.bbox.xmax) == (0, 60)
        assert (second.bbox.xmin, second.bbox.xmax) == (60, 100)

    def test_single_word_spans_full_width(self):
        node = make_node(width=80, height=20)
        result = _bbox_word_segmentation(node, ["alleluia"], "alleluia")
        assert len(result.segments) == 1
        assert result.segments[0].bbox.xmax == 80


class TestFallbackWordSegmentation:
    def test_splits_recognised_text_into_words(self):
        node = make_node(text="dominus omnipotens", width=300, height=30)
        result = _fallback_word_segmentation(node)
        words = [
            seg.data[TEXT_RESULT_KEY].top_candidate()
            for seg in result.segments
        ]
        assert words == ["dominus", "omnipotens"]

    def test_empty_text_produces_one_empty_word_segment(self):
        # text="" -> words=text.split()=[] -> falls to `[text] if text
        # else [""]`; text is falsy, so this is [""], not [text].
        node = make_node(text="", width=50, height=20)
        result = _fallback_word_segmentation(node)
        assert len(result.segments) == 1
        assert result.segments[0].data[TEXT_RESULT_KEY].top_candidate() == ""

    def test_none_text_is_treated_like_empty_text(self):
        node = make_node(text=None, width=50, height=20)
        result = _fallback_word_segmentation(node)
        assert len(result.segments) == 1
        assert result.segments[0].data[TEXT_RESULT_KEY].top_candidate() == ""

    def test_whitespace_only_text_is_kept_as_a_single_word(self):
        # text="   " -> text.split()=[] -> falls to `[text] if text else
        # [""]`; text is truthy (non-empty string), so the whole
        # whitespace string becomes the sole "word" verbatim, unlike the
        # fully-empty-text case above.
        node = make_node(text="   ", width=50, height=20)
        result = _fallback_word_segmentation(node)
        assert len(result.segments) == 1
        assert (
            result.segments[0].data[TEXT_RESULT_KEY].top_candidate() == "   "
        )

"""Tests for run_pipeline.py: the pre-NW music-region overlap filter.

_music_overlap_ratio is a pure geometry function, tested directly. The
surrounding keep/drop filter block (run_pipeline.py's run(), just before
Stage 4) is inline code inside a ~900-line function that also drives real
Kraken/HTRflow stages — rather than mock the entire pipeline end-to-end
(low value, high fragility), these tests mock only Stages 1-3 (segmentation,
column clustering, recognition) to reach the filter block with controlled
fake line nodes, then let a mocked Stage-4 call raise a sentinel exception
so the test can inspect the real filter code's effect on those nodes before
the run() call is abandoned partway through.
"""

import types
from unittest.mock import patch

import pytest

import run_pipeline
from run_pipeline import _music_overlap_ratio


def _bbox(x0, y0, x1, y1):
    return types.SimpleNamespace(xmin=x0, ymin=y0, xmax=x1, ymax=y1)


class TestMusicOverlapRatio:
    def test_zero_overlap_returns_zero(self):
        line = _bbox(0, 0, 10, 10)
        assert _music_overlap_ratio(line, [20, 20, 30, 30]) == 0.0

    def test_full_overlap_returns_one(self):
        line = _bbox(0, 0, 10, 10)
        assert _music_overlap_ratio(line, [0, 0, 10, 10]) == 1.0

    def test_degenerate_zero_area_line_returns_zero(self):
        # xmin == xmax -> line_area == 0; guarded to avoid ZeroDivisionError.
        line = _bbox(5, 0, 5, 10)
        assert _music_overlap_ratio(line, [0, 0, 10, 10]) == 0.0

    def test_partial_overlap_numeric_case(self):
        # Line is a 10x10 box at (0,0); music box overlaps its right half:
        # intersection = 5x10 = 50, line_area = 100 -> ratio = 0.5.
        line = _bbox(0, 0, 10, 10)
        assert _music_overlap_ratio(line, [5, 0, 15, 10]) == 0.5

    def test_touching_edges_not_counted_as_overlap(self):
        # ix1 <= ix0 guard: boxes that only share an edge -> 0.0, not a
        # sliver of positive area.
        line = _bbox(0, 0, 10, 10)
        assert _music_overlap_ratio(line, [10, 0, 20, 10]) == 0.0


class _FakeNode:
    def __init__(self, label, bbox, text=""):
        self.label = label
        self.bbox = bbox
        self.text = text


class _FakePage:
    def __init__(self, children):
        self.children = children
        self.image = types.SimpleNamespace(shape=(100, 100, 3))


class _FakeCollection:
    def __init__(self, page):
        self._page = page

    def __iter__(self):
        return iter([self._page])

    def active_leaves(self):
        return list(self._page.children)


def _run_up_to_music_filter(page, music_boxes, music_overlap_threshold=0.30):
    """Drive run_pipeline.run() through its music filter, then abort.

    Stages 1-3 and Collection() are mocked to hand back `page` untouched;
    fuse_colinear_segments (the first call after the filter block) raises
    a sentinel so the real filter code's mutations on `page`/`collection`
    can be inspected without needing to mock the rest of the pipeline.
    """
    fake_collection = _FakeCollection(page)
    with patch("run_pipeline.Collection", return_value=fake_collection), \
         patch("run_pipeline.KrakenSegmentation") as mock_seg, \
         patch("run_pipeline.cluster_columns",
               return_value=(["line0", "line1"], 1, None)), \
         patch("run_pipeline.KrakenRecognition") as mock_rec, \
         patch("run_pipeline.fuse_colinear_segments",
               side_effect=RuntimeError("STOP_AFTER_FILTER")):
        mock_seg.return_value.run.return_value = fake_collection
        mock_rec.return_value.run.return_value = fake_collection
        with pytest.raises(RuntimeError, match="STOP_AFTER_FILTER"):
            run_pipeline.run(
                image_path="fake.jpg",
                folio="001r",
                music_boxes=music_boxes,
                music_overlap_threshold=music_overlap_threshold,
            )
    return fake_collection


class TestMusicFilterBlock:
    def test_no_music_boxes_skips_filter_and_keeps_all_lines(self):
        nodes = [
            _FakeNode("line0", _bbox(0, 0, 10, 10)),
            _FakeNode("line1", _bbox(0, 20, 10, 30)),
        ]
        page = _FakePage(list(nodes))
        collection = _run_up_to_music_filter(page, music_boxes=None)
        assert page.children == nodes
        assert collection._music_filter_dropped == []

    def test_line_over_threshold_is_dropped(self):
        # Full overlap (ratio=1.0) with a 0.30 threshold -> dropped.
        kept_node = _FakeNode("line0", _bbox(0, 0, 10, 10))
        dropped_node = _FakeNode(
            "line1", _bbox(0, 20, 10, 30), text="neume-adjacent"
        )
        page = _FakePage([kept_node, dropped_node])
        collection = _run_up_to_music_filter(
            page, music_boxes=[[0, 20, 10, 30]], music_overlap_threshold=0.30
        )
        assert page.children == [kept_node]
        assert collection._music_filter_dropped == [
            {"bbox": [0, 20, 10, 30], "text": "neume-adjacent"}
        ]

    def test_line_at_exact_threshold_is_kept_not_dropped(self):
        # Overlap ratio == threshold exactly: the filter uses strict `>`,
        # so an exact match must be KEPT, not dropped.
        line = _bbox(0, 0, 10, 10)
        node = _FakeNode("line0", line)
        page = _FakePage([node])
        # Music box overlaps exactly 30% of the line's area (3x10 of 10x10).
        collection = _run_up_to_music_filter(
            page, music_boxes=[[0, 0, 3, 10]], music_overlap_threshold=0.30
        )
        assert page.children == [node]
        assert collection._music_filter_dropped == []

    def test_line_just_over_threshold_is_dropped(self):
        line = _bbox(0, 0, 10, 10)
        node = _FakeNode("line0", line, text="x")
        page = _FakePage([node])
        # 3.01/10 = 0.301 > 0.30 -> dropped.
        collection = _run_up_to_music_filter(
            page, music_boxes=[[0, 0, 3.01, 10]], music_overlap_threshold=0.30
        )
        assert page.children == []
        # _music_filter_dropped records the LINE's own bbox, not the music
        # box it overlapped.
        assert collection._music_filter_dropped == [
            {"bbox": [0, 0, 10, 10], "text": "x"}
        ]

    def test_line_overlapping_any_of_multiple_music_boxes_is_dropped(self):
        # any(...) semantics: overlapping just one of several music boxes
        # is enough to drop the line, even if it misses the others.
        line = _bbox(0, 0, 10, 10)
        node = _FakeNode("line0", line, text="x")
        page = _FakePage([node])
        collection = _run_up_to_music_filter(
            page,
            music_boxes=[[100, 100, 110, 110], [0, 0, 10, 10]],
            music_overlap_threshold=0.30,
        )
        assert page.children == []
        assert len(collection._music_filter_dropped) == 1

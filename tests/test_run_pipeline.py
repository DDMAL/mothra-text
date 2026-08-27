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
from run_pipeline import (
    _AreaBounds,
    _area_overlap_ratio,
    _main_text_area,
    _music_overlap_ratio,
    _row_groups,
    _union_width,
)


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


class TestRowGroups:
    def test_empty_list_returns_empty(self):
        assert _row_groups([]) == []

    def test_single_node_forms_own_row(self):
        node = _FakeNode("a", _bbox(0, 0, 10, 10))
        assert _row_groups([node]) == [[node]]

    def test_overlapping_nodes_group_into_one_row(self):
        # Overlap of 6px against a 10px height -> 0.6 >= the 0.5 threshold.
        a = _FakeNode("a", _bbox(0, 0, 10, 10))
        b = _FakeNode("b", _bbox(20, 4, 30, 14))
        assert _row_groups([a, b]) == [[a, b]]

    def test_non_overlapping_nodes_form_separate_rows(self):
        a = _FakeNode("a", _bbox(0, 0, 10, 10))
        b = _FakeNode("b", _bbox(0, 30, 10, 40))
        assert _row_groups([a, b]) == [[a], [b]]

    def test_grouping_is_independent_of_input_order(self):
        a = _FakeNode("a", _bbox(0, 20, 10, 30))
        b = _FakeNode("b", _bbox(0, 0, 10, 10))
        assert _row_groups([a, b]) == [[b], [a]]


class TestUnionWidth:
    def test_single_interval(self):
        assert _union_width([(0, 10)], -100, 100) == 10

    def test_overlapping_intervals_merge(self):
        assert _union_width([(0, 10), (5, 15)], -100, 100) == 15

    def test_adjacent_intervals_merge(self):
        assert _union_width([(0, 10), (10, 20)], -100, 100) == 20

    def test_gap_between_intervals_not_counted(self):
        # Two 10px intervals with a real gap between them must sum to 20,
        # not the 30px outer span -- the regression this helper exists to
        # prevent (an outer-span version let a stray box inflate a row's
        # apparent coverage; see mothra-text#53).
        assert _union_width([(0, 10), (20, 30)], -100, 100) == 20

    def test_clipping_to_bounds(self):
        assert _union_width([(-5, 15)], 0, 10) == 10

    def test_interval_entirely_outside_bounds_contributes_nothing(self):
        assert _union_width([(200, 300)], 0, 10) == 0


class TestMainTextArea:
    def test_fragmented_row_becomes_x_reference(self):
        # Five real chant lines, each ONE wide box, establish the column.
        # A sixth physical line got fragmented by BLLA into three narrow
        # side-by-side boxes that together span nearly the same width --
        # it must be recognized even though no single fragment is
        # individually wide (032r/040v/008r/009r shape).
        wide = [
            _FakeNode(f"w{i}", _bbox(100, 100 + i * 100, 900, 140 + i * 100))
            for i in range(5)
        ]
        fragments = [
            _FakeNode("f0", _bbox(100, 600, 300, 640)),
            _FakeNode("f1", _bbox(320, 600, 600, 640)),
            _FakeNode("f2", _bbox(620, 600, 900, 640)),
        ]
        bounds = _main_text_area(wide + fragments, split_x=None)
        b = bounds[1]
        assert b.xmin == pytest.approx(100)
        assert b.xmax == pytest.approx(900)
        assert b.ymin <= 600
        assert b.ymax >= 640

    def test_low_coverage_row_still_extends_y_at_035(self):
        # A row whose members cover ~37.5% of the established column width
        # (real gaps between pieces, doxology shape from 005r) must still
        # extend the y-extent -- this is why y_reference_ratio is 0.35, not
        # the 0.5 that was tried first and left this case uncaught.
        wide = [
            _FakeNode(f"w{i}", _bbox(100, 300 + i * 100, 900, 340 + i * 100))
            for i in range(5)
        ]
        # Combined width = (200-100)+(500-400)+(700-600) = 300 of an 800
        # column -> 37.5% coverage.
        sparse_row = [
            _FakeNode("s0", _bbox(100, 50, 200, 90)),
            _FakeNode("s1", _bbox(400, 55, 500, 95)),
            _FakeNode("s2", _bbox(600, 60, 700, 100)),
        ]
        bounds = _main_text_area(wide + sparse_row, split_x=None)
        b = bounds[1]
        assert b.ymin is not None and b.ymin <= 50

    def test_isolated_narrow_row_does_not_extend_y(self):
        # A single narrow, isolated box far from the reference lines (top-
        # rubric / david shape) must NOT pull the y-extent out to cover it.
        wide = [
            _FakeNode(f"w{i}", _bbox(100, 300 + i * 100, 900, 340 + i * 100))
            for i in range(5)
        ]
        marginal = _FakeNode("m", _bbox(910, 0, 950, 40))
        bounds = _main_text_area(wide + [marginal], split_x=None)
        b = bounds[1]
        assert b.ymin == 300

    def test_no_row_meets_y_coverage_leaves_y_axis_inert(self):
        # Two equally-wide rows straddle a gap; their median-based x-extent
        # sits in the middle where NEITHER original row actually reaches,
        # so neither row's clipped coverage meets y_reference_ratio and the
        # y-axis is left inert (None) rather than accidentally extended.
        row_a = _FakeNode("a", _bbox(0, 0, 1000, 40))
        row_b = _FakeNode("b", _bbox(2000, 100, 3000, 140))
        bounds = _main_text_area([row_a, row_b], split_x=None)
        b = bounds[1]
        assert b.xmin == 1000 and b.xmax == 2000
        assert b.ymin is None and b.ymax is None

    def test_two_columns_get_independent_bounds(self):
        col1 = [
            _FakeNode(f"a{i}", _bbox(0, i * 100, 400, 40 + i * 100))
            for i in range(4)
        ]
        col2 = [
            _FakeNode(f"b{i}", _bbox(600, i * 100, 1000, 40 + i * 100))
            for i in range(4)
        ]
        bounds = _main_text_area(col1 + col2, split_x=500)
        assert bounds[1].xmax <= 500
        assert bounds[2].xmin >= 500


class TestAreaOverlapRatio:
    def test_fully_inside_returns_one(self):
        node = _bbox(0, 0, 10, 10)
        assert _area_overlap_ratio(node, _AreaBounds(0, 20, 0, 20)) == 1.0

    def test_fully_outside_returns_zero(self):
        node = _bbox(0, 0, 10, 10)
        bounds = _AreaBounds(100, 200, 100, 200)
        assert _area_overlap_ratio(node, bounds) == 0.0

    def test_partial_overlap(self):
        node = _bbox(0, 0, 10, 10)
        bounds = _AreaBounds(5, 20, 0, 20)
        assert _area_overlap_ratio(node, bounds) == pytest.approx(0.5)

    def test_unconstrained_x_axis_only_penalizes_on_y(self):
        node = _bbox(0, 0, 10, 10)
        bounds = _AreaBounds(None, None, 5, 20)
        assert _area_overlap_ratio(node, bounds) == pytest.approx(0.5)

    def test_both_axes_unconstrained_returns_one(self):
        node = _bbox(0, 0, 10, 10)
        bounds = _AreaBounds(None, None, None, None)
        assert _area_overlap_ratio(node, bounds) == 1.0


def _run_up_to_offarea_filter(
    page,
    drop_offarea_boxes=True,
    area_keep_threshold=0.50,
    column_count=1,
    split_x=None,
):
    """Drive run_pipeline.run() through its off-area filter, then abort.

    Same technique as _run_up_to_music_filter: mock Stages 1-3 and
    Collection() to hand back `page` untouched (with music_boxes left at
    its None default, so that earlier filter is a no-op), and let
    fuse_colinear_segments -- the first call after both pre-Stage-4 filters
    -- raise a sentinel so the real filter code's mutations on `page`/
    `collection` can be inspected without mocking the rest of the pipeline.
    """
    fake_collection = _FakeCollection(page)
    with patch("run_pipeline.Collection", return_value=fake_collection), \
         patch("run_pipeline.KrakenSegmentation") as mock_seg, \
         patch("run_pipeline.cluster_columns",
               return_value=(["line0", "line1"], column_count, split_x)), \
         patch("run_pipeline.KrakenRecognition") as mock_rec, \
         patch("run_pipeline.fuse_colinear_segments",
               side_effect=RuntimeError("STOP_AFTER_FILTER")):
        mock_seg.return_value.run.return_value = fake_collection
        mock_rec.return_value.run.return_value = fake_collection
        with pytest.raises(RuntimeError, match="STOP_AFTER_FILTER"):
            run_pipeline.run(
                image_path="fake.jpg",
                folio="001r",
                drop_offarea_boxes=drop_offarea_boxes,
                area_keep_threshold=area_keep_threshold,
            )
    return fake_collection


class TestOffAreaFilterBlock:
    def test_no_drop_offarea_boxes_flag_skips_filter(self):
        nodes = [
            _FakeNode("line0", _bbox(0, 0, 10, 10)),
            _FakeNode("line1", _bbox(0, 20, 10, 30)),
        ]
        page = _FakePage(list(nodes))
        collection = _run_up_to_offarea_filter(page, drop_offarea_boxes=False)
        assert page.children == nodes
        assert collection._offarea_filter_dropped == []

    def test_too_few_lines_leaves_filter_inert(self):
        lone = _FakeNode("lone", _bbox(0, 0, 10, 10), text="x")
        page = _FakePage([lone])
        collection = _run_up_to_offarea_filter(page)
        assert lone in page.children
        assert collection._offarea_filter_dropped == []

    def test_isolated_offcolumn_box_dropped(self):
        # Five wide reference lines establish the column; one small box far
        # to the side (the "david" shape) must be dropped.
        wide = [
            _FakeNode(f"w{i}", _bbox(100, i * 100, 900, 40 + i * 100))
            for i in range(5)
        ]
        marginal = _FakeNode("m", _bbox(0, 200, 40, 240), text="deus")
        page = _FakePage(wide + [marginal])
        collection = _run_up_to_offarea_filter(page)
        assert marginal not in page.children
        assert all(w in page.children for w in wide)
        dropped_texts = [d["text"] for d in collection._offarea_filter_dropped]
        assert dropped_texts == ["deus"]

    def test_isolated_offrow_box_dropped_above_block(self):
        # A small box well above the block, unaligned with anything (top-
        # rubric / folio-number shape).
        wide = [
            _FakeNode(f"w{i}", _bbox(100, 300 + i * 100, 900, 340 + i * 100))
            for i in range(5)
        ]
        header = _FakeNode("h", _bbox(700, 0, 800, 30), text="Fa")
        page = _FakePage(wide + [header])
        collection = _run_up_to_offarea_filter(page)
        assert header not in page.children
        assert len(collection._offarea_filter_dropped) == 1

    def test_fragmented_real_line_kept(self):
        wide = [
            _FakeNode(f"w{i}", _bbox(100, i * 100, 900, 40 + i * 100))
            for i in range(5)
        ]
        fragments = [
            _FakeNode("f0", _bbox(100, 600, 300, 640), text="Qnoni"),
            _FakeNode("f1", _bbox(320, 600, 600, 640), text="am"),
            _FakeNode("f2", _bbox(620, 600, 900, 640), text="ego"),
        ]
        page = _FakePage(wide + fragments)
        collection = _run_up_to_offarea_filter(page)
        for f in fragments:
            assert f in page.children
        assert collection._offarea_filter_dropped == []

    def test_low_coverage_fragmented_line_still_kept(self):
        # The 005r doxology shape: real gaps between pieces, ~37.5%
        # coverage -- this is the case that set y_reference_ratio to 0.35.
        wide = [
            _FakeNode(f"w{i}", _bbox(100, 300 + i * 100, 900, 340 + i * 100))
            for i in range(5)
        ]
        sparse = [
            _FakeNode("s0", _bbox(100, 50, 200, 90), text="oria"),
            _FakeNode("s1", _bbox(400, 55, 500, 95), text="et"),
            _FakeNode("s2", _bbox(600, 60, 700, 100), text="Sancto"),
        ]
        page = _FakePage(wide + sparse)
        collection = _run_up_to_offarea_filter(page)
        for s in sparse:
            assert s in page.children
        assert collection._offarea_filter_dropped == []

    def test_two_column_box_evaluated_against_own_column(self):
        col1 = [
            _FakeNode(f"a{i}", _bbox(0, i * 100, 400, 40 + i * 100))
            for i in range(4)
        ]
        col2 = [
            _FakeNode(f"b{i}", _bbox(600, i * 100, 1000, 40 + i * 100))
            for i in range(4)
        ]
        # Outside column 1's bounds but numerically inside column 2's --
        # with split_x=500 it belongs to column 1 (xmin<500) and must be
        # judged against column 1's bounds, not column 2's.
        outlier = _FakeNode("o", _bbox(420, 0, 460, 40), text="stray")
        page = _FakePage(col1 + col2 + [outlier])
        _run_up_to_offarea_filter(page, column_count=2, split_x=500)
        assert outlier not in page.children
        assert all(n in page.children for n in col1 + col2)

    def test_ratio_exactly_at_threshold_is_kept_not_dropped(self):
        node = _FakeNode("edge", _bbox(0, 0, 10, 10), text="edge")
        page = _FakePage([node])
        fixed_bounds = {1: _AreaBounds(5, 20, 0, 20)}  # ratio == 0.5 exactly
        with patch("run_pipeline._main_text_area", return_value=fixed_bounds):
            collection = _run_up_to_offarea_filter(page)
        assert node in page.children
        assert collection._offarea_filter_dropped == []


class _FakeWordNode:
    """Node-like fake exposing the .get() interface _build_pipeline_payload
    reads `source` through (see htrflow.volume.node.Node.get)."""

    def __init__(self, label, bbox, text="", source=None):
        self.label = label
        self.bbox = bbox
        self.text = text
        self.children = []
        self.data = {} if source is None else {"source": source}

    def get(self, key, default=None):
        return self.data.get(key, default)


def _fake_line_node(label, word):
    return types.SimpleNamespace(
        label=label,
        bbox=_bbox(0, 0, 100, 10),
        polygon=types.SimpleNamespace(points=[types.SimpleNamespace(x=0, y=0)]),
        text=word.text,
        children=[word],
    )


class TestBuildPipelinePayloadSourceTag:
    """Regression coverage for mothra-text#59.

    `source` must be read off the word node's own data (stamped at Stage 5
    build time by GroundTruthWordSegmentation), not re-derived from
    manifest.get(line_node.label) -- a relabel between Stage 5 and export
    can desync line_node.label from the keys `manifest` was built with, so
    tagging must not depend on that agreement holding.
    """

    def test_source_read_from_word_node_survives_label_manifest_mismatch(self):
        word = _FakeWordNode(
            "line0_word0", _bbox(0, 0, 10, 10), text="dominus", source="gt"
        )
        line = _fake_line_node("line0", word)
        collection = _FakeCollection(_FakePage([line]))

        # Keyed by a label that does NOT match line.label, simulating the
        # post-Stage-5-relabel drift from mothra-text#59.
        manifest = {"stale_label_that_does_not_match": "dominus"}

        payload = run_pipeline._build_pipeline_payload(
            collection, "fake.jpg", manifest
        )

        assert payload["lines"][0]["words"][0]["source"] == "gt"

    def test_source_defaults_to_fallback_when_untagged(self):
        word = _FakeWordNode("line0_word0", _bbox(0, 0, 10, 10), text="x")
        line = _fake_line_node("line0", word)
        collection = _FakeCollection(_FakePage([line]))

        payload = run_pipeline._build_pipeline_payload(collection, "fake.jpg", {})

        assert payload["lines"][0]["words"][0]["source"] == "fallback"

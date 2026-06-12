"""Tests for steps.column_clustering."""

from unittest.mock import MagicMock

from steps.column_clustering import (
    cluster_columns,
    fuse_colinear_segments,
)


def _make_line(
    label: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int = 0,
) -> MagicMock:
    node = MagicMock()
    node.label = label
    node.bbox.xmin = xmin
    node.bbox.xmax = xmax
    node.bbox.ymin = ymin
    node.bbox.ymax = ymax if ymax else ymin + 100
    return node


class TestClusterColumns:
    # ---- edge cases ----

    def test_empty(self):
        labels, count, split_x = cluster_columns([], page_width=1000)
        assert labels == []
        assert count == 0
        assert split_x is None

    def test_single_node(self):
        node = _make_line("line_0", xmin=50, xmax=400, ymin=10)
        labels, count, split_x = cluster_columns([node], page_width=1000)
        assert labels == ["line_0"]
        assert count == 1
        assert split_x is None

    # ---- 1-column cases ----

    def test_single_column_sorted_by_y(self):
        nodes = [
            _make_line("line_2", xmin=50, xmax=800, ymin=200),
            _make_line("line_0", xmin=52, xmax=802, ymin=10),
            _make_line("line_1", xmin=51, xmax=801, ymin=100),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert labels == ["line_0", "line_1", "line_2"]

    def test_continuous_coverage_single_column(self):
        # All lines span the full text width — coverage is a flat plateau
        # in the text region, no bimodal valley.
        nodes = [
            _make_line("l0", xmin=50, xmax=900, ymin=10),
            _make_line("l1", xmin=50, xmax=900, ymin=100),
            _make_line("l2", xmin=50, xmax=900, ymin=200),
            _make_line("l3", xmin=50, xmax=900, ymin=300),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert labels == ["l0", "l1", "l2", "l3"]

    def test_gutter_in_margin_not_detected(self):
        # All lines occupy one contiguous region (200–700 px).  The zeros
        # in coverage outside that region are page margins, not a gutter,
        # and the text-region search excludes them.
        nodes = [
            _make_line("l0", xmin=200, xmax=700, ymin=10),
            _make_line("l1", xmin=200, xmax=700, ymin=100),
            _make_line("l2", xmin=200, xmax=700, ymin=200),
            _make_line("l3", xmin=200, xmax=700, ymin=300),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1

    # ---- 2-column cases ----

    def test_two_columns_detected(self):
        # Left col: 50–400, right col: 600–950, gutter: 200 px (20 %).
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=50,  xmax=400, ymin=100),
            _make_line("l2", xmin=50,  xmax=400, ymin=200),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
            _make_line("r1", xmin=600, xmax=950, ymin=100),
            _make_line("r2", xmin=600, xmax=950, ymin=200),
        ]
        labels, count, split_x = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert split_x is not None
        assert 400 < split_x < 600
        assert labels[:3] == ["l0", "l1", "l2"]
        assert labels[3:] == ["r0", "r1", "r2"]

    def test_two_columns_reading_order(self):
        # Lines in each column are deliberately out of sequence in the
        # input; reading order must be left-column top-to-bottom then
        # right-column top-to-bottom.
        nodes = [
            _make_line("r1", xmin=600, xmax=950, ymin=10),
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=50,  xmax=400, ymin=200),
            _make_line("r0", xmin=600, xmax=950, ymin=200),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels == ["l0", "l1", "r1", "r0"]

    def test_spanning_bbox_minority_still_two_columns(self):
        # 6 column-contained lines + 2 BLLA spanning artefacts.
        # Valley coverage = 2, column peaks = 5 → ratio 0.40 < threshold 0.50.
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=50,  xmax=400, ymin=100),
            _make_line("l2", xmin=50,  xmax=400, ymin=200),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
            _make_line("r1", xmin=600, xmax=950, ymin=100),
            _make_line("r2", xmin=600, xmax=950, ymin=200),
            _make_line("sp0", xmin=50, xmax=950, ymin=50),
            _make_line("sp1", xmin=50, xmax=950, ymin=150),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2

    def test_within_column_y_sort_preserved(self):
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("r2", xmin=600, xmax=950, ymin=300),
            _make_line("l1", xmin=50,  xmax=400, ymin=200),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
            _make_line("r1", xmin=600, xmax=950, ymin=150),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels == ["l0", "l1", "r0", "r1", "r2"]

    # ---- parameter control ----

    def test_sparse_page_below_min_peak_count_not_split(self):
        # Only 1 line per column side — each peak is 1 < min_peak_count=2.
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1

    def test_sparse_page_meets_min_peak_count_split(self):
        # 2 lines per side — each peak equals min_peak_count=2.
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=50,  xmax=400, ymin=100),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
            _make_line("r1", xmin=600, xmax=950, ymin=100),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2

    def test_custom_bimodal_threshold_prevents_split(self):
        # 6 column lines + 3 spanning → valley=3, peaks=6, ratio=0.50.
        # Default threshold 0.50: 3 < 3 is False → 1 column.
        # Raised threshold 0.60: 3 < 3.6 is True  → 2 columns.
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=50,  xmax=400, ymin=100),
            _make_line("l2", xmin=50,  xmax=400, ymin=200),
            _make_line("r0", xmin=600, xmax=950, ymin=10),
            _make_line("r1", xmin=600, xmax=950, ymin=100),
            _make_line("r2", xmin=600, xmax=950, ymin=200),
            _make_line("sp0", xmin=50, xmax=950, ymin=50),
            _make_line("sp1", xmin=50, xmax=950, ymin=150),
            _make_line("sp2", xmin=50, xmax=950, ymin=250),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1  # valley/peak = 3/6 = 0.50, not < 0.50

        _, count, _ = cluster_columns(
            nodes, page_width=1000, bimodal_threshold=0.60
        )
        assert count == 2  # 3/6 = 0.50 < 0.60

    def test_custom_min_gutter_fraction_controls_split(self):
        # Gutter: 40 px (x 480–520) = 4 % of page_width=1000.
        # Default min_gutter_fraction=0.02 (20 px) → passes.
        # Raised  min_gutter_fraction=0.08 (80 px) → fails.
        nodes = [
            _make_line("l0", xmin=50,  xmax=480, ymin=10),
            _make_line("l1", xmin=50,  xmax=480, ymin=100),
            _make_line("l2", xmin=50,  xmax=480, ymin=200),
            _make_line("l3", xmin=50,  xmax=480, ymin=300),
            _make_line("r0", xmin=520, xmax=950, ymin=10),
            _make_line("r1", xmin=520, xmax=950, ymin=100),
            _make_line("r2", xmin=520, xmax=950, ymin=200),
            _make_line("r3", xmin=520, xmax=950, ymin=300),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2  # default 2 % ≤ 4 %

        _, count, _ = cluster_columns(
            nodes, page_width=1000, min_gutter_fraction=0.08
        )
        assert count == 1  # 8 % > 4 %


class TestFuseColinearSegments:
    def test_single_node_no_fusion(self):
        node = _make_line("seg0", xmin=50, xmax=400, ymin=100, ymax=200)
        result = fuse_colinear_segments([node], split_x=None)
        assert len(result) == 1
        assert result[0].constituent_labels == ["seg0"]

    def test_two_overlapping_segments_fused(self):
        # seg0: y=100–200, seg1: y=150–250 → 50px overlap, shorter=100px → 50%
        seg0 = _make_line("seg0", xmin=50,  xmax=300, ymin=100, ymax=200)
        seg1 = _make_line("seg1", xmin=350, xmax=600, ymin=150, ymax=250)
        result = fuse_colinear_segments([seg0, seg1], split_x=None)
        assert len(result) == 1
        assert set(result[0].constituent_labels) == {"seg0", "seg1"}

    def test_two_non_overlapping_segments_separate(self):
        # seg0: y=100–190, seg1: y=300–390 → no overlap
        seg0 = _make_line("seg0", xmin=50,  xmax=300, ymin=100, ymax=190)
        seg1 = _make_line("seg1", xmin=50,  xmax=300, ymin=300, ymax=390)
        result = fuse_colinear_segments([seg0, seg1], split_x=None)
        assert len(result) == 2

    def test_overlap_below_threshold_not_fused(self):
        # seg0: y=100–200 (height=100), seg1: y=180–280 (height=100)
        # overlap = 20px, frac = 20/100 = 0.20 < 0.50 → separate
        seg0 = _make_line("seg0", xmin=50, xmax=300, ymin=100, ymax=200)
        seg1 = _make_line("seg1", xmin=50, xmax=300, ymin=180, ymax=280)
        result = fuse_colinear_segments([seg0, seg1], split_x=None)
        assert len(result) == 2

    def test_reading_order_within_group(self):
        # right-side seg (seg_right) has smaller ymin than left-side seg —
        # after fusion constituents must be sorted by xmin, not ymin.
        seg_left = _make_line("seg_l", xmin=50, xmax=300, ymin=105, ymax=205)
        seg_right = _make_line("seg_r", xmin=350, xmax=600, ymin=100, ymax=200)
        result = fuse_colinear_segments([seg_left, seg_right], split_x=None)
        assert len(result) == 1
        assert result[0].constituent_labels == ["seg_l", "seg_r"]

    def test_different_columns_not_fused(self):
        # Both segments overlap 100% in y but are on opposite sides of split_x.
        seg_l = _make_line("left",  xmin=50,  xmax=300, ymin=100, ymax=200)
        seg_r = _make_line("right", xmin=600, xmax=900, ymin=100, ymax=200)
        result = fuse_colinear_segments([seg_l, seg_r], split_x=400.0)
        assert len(result) == 2
        labels = [r.constituent_labels[0] for r in result]
        assert "left" in labels
        assert "right" in labels

    def test_three_segment_line_fused(self):
        # Three segments all on the same physical line.
        seg_a = _make_line("a", xmin=300, xmax=500, ymin=100, ymax=200)
        seg_b = _make_line("b", xmin=50,  xmax=250, ymin=105, ymax=205)
        seg_c = _make_line("c", xmin=550, xmax=750, ymin=110, ymax=210)
        result = fuse_colinear_segments([seg_a, seg_b, seg_c], split_x=None)
        assert len(result) == 1
        # Constituents must be in xmin order: b(50), a(300), c(550)
        assert result[0].constituent_labels == ["b", "a", "c"]

    def test_constituent_widths_correct(self):
        seg0 = _make_line("s0", xmin=50,  xmax=250, ymin=100, ymax=200)
        seg1 = _make_line("s1", xmin=300, xmax=600, ymin=105, ymax=205)
        result = fuse_colinear_segments([seg0, seg1], split_x=None)
        assert len(result) == 1
        assert result[0].constituent_widths == [200, 300]

    def test_reading_order_left_then_right_column(self):
        # 2 left-column and 2 right-column nodes at different y positions.
        # Right column lines come after all left column lines.
        l0 = _make_line("l0", xmin=50,  xmax=300, ymin=100, ymax=200)
        l1 = _make_line("l1", xmin=50,  xmax=300, ymin=500, ymax=600)
        r0 = _make_line("r0", xmin=600, xmax=900, ymin=100, ymax=200)
        r1 = _make_line("r1", xmin=600, xmax=900, ymin=500, ymax=600)
        result = fuse_colinear_segments([l0, l1, r0, r1], split_x=400.0)
        assert len(result) == 4
        # First two fused lines should be column 1, last two column 2
        assert result[0].column == 1
        assert result[1].column == 1
        assert result[2].column == 2
        assert result[3].column == 2
        # Within each column, sorted by ymin
        assert result[0].ymin < result[1].ymin
        assert result[2].ymin < result[3].ymin

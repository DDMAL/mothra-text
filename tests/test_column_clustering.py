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

    def test_single_column_sorted_by_y(self):
        nodes = [
            _make_line("line_2", xmin=50, xmax=400, ymin=200),
            _make_line("line_0", xmin=52, xmax=402, ymin=10),
            _make_line("line_1", xmin=51, xmax=401, ymin=100),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert labels == ["line_0", "line_1", "line_2"]

    def test_two_columns_reading_order(self):
        # Left column: xmin ≈ 50, right column: xmin ≈ 600
        nodes = [
            _make_line("r1", xmin=600, xmax=900, ymin=10),
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("l1", xmin=52,  xmax=402, ymin=100),
            _make_line("r0", xmin=602, xmax=902, ymin=100),
        ]
        labels, count, split_x = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert split_x is not None
        # Left column top-to-bottom, then right column top-to-bottom
        assert labels == ["l0", "l1", "r1", "r0"]

    def test_tight_clusters_detected_with_sufficient_typical_gap(self):
        # Tight xmin clusters (spread ≤ 4 px each) with a typical gap of
        # 138 px = 13.8 % of page_width=1000, above the 12 % threshold.
        # median(contained left xmax)=160, min_right_xmin=298 → gap=138.
        nodes = [
            _make_line("l0", xmin=48, xmax=150, ymin=10),
            _make_line("l1", xmin=50, xmax=165, ymin=100),
            _make_line("l2", xmin=52, xmax=160, ymin=200),
            _make_line("r0", xmin=300, xmax=480, ymin=10),
            _make_line("r1", xmin=302, xmax=482, ymin=100),
            _make_line("r2", xmin=298, xmax=478, ymin=200),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels[:3] == ["l0", "l1", "l2"]
        assert labels[3:] == ["r0", "r1", "r2"]

    def test_disjoint_extents_sufficient_without_variance(self):
        # Clusters are disjoint (left xmax < right xmin) but the variance
        # reduction is low because we only have 2 nodes.  The disjointness
        # check alone should declare two columns.
        # median(contained left xmax)=100, min_right_xmin=350 → gap=250=25%.
        nodes = [
            _make_line("l0", xmin=50, xmax=100, ymin=10),
            _make_line("r0", xmin=350, xmax=500, ymin=10),
        ]
        _, count, _ = cluster_columns(
            nodes, page_width=1000, variance_threshold=0.99
        )
        assert count == 2

    def test_non_disjoint_single_column(self):
        # Lines span much of the page width — not disjoint, not bimodal.
        nodes = [
            _make_line("l0", xmin=50, xmax=900, ymin=10),
            _make_line("l1", xmin=55, xmax=905, ymin=100),
            _make_line("l2", xmin=48, xmax=898, ymin=200),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert labels == ["l0", "l1", "l2"]

    def test_identical_xmins_single_column(self):
        nodes = [
            _make_line("a", xmin=100, xmax=800, ymin=10),
            _make_line("b", xmin=100, xmax=800, ymin=100),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1

    def test_overlapping_x_clusters_treated_as_single_column(self):
        # Left xmax (380) > right xmin (300): clusters physically overlap in x.
        # gutter = 300 - 380 = -80 < 0 → gutter_ok=False → single column.
        nodes = [
            _make_line("l0", xmin=0,   xmax=350, ymin=10),
            _make_line("l1", xmin=200, xmax=380, ymin=100),
            _make_line("r0", xmin=300, xmax=500, ymin=10),
            _make_line("r1", xmin=500, xmax=700, ymin=100),
        ]
        _, count, _ = cluster_columns(
            nodes, page_width=1000, variance_threshold=0.5
        )
        assert count == 1

    def test_near_touching_clusters_treated_as_single_column(self):
        # Left xmax=500 == right xmin=500: gap=0px < 2% threshold
        # → single-column.
        nodes = [
            _make_line("l0", xmin=100, xmax=500, ymin=10),
            _make_line("l1", xmin=110, xmax=510, ymin=100),
            _make_line("r0", xmin=500, xmax=900, ymin=10),
            _make_line("r1", xmin=510, xmax=910, ymin=100),
        ]
        labels, count, split_x = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert split_x is None

    def test_gutter_threshold_controls_detection(self):
        # l3 (xmax=510) spans past r0 (xmin=500), forcing the non-disjoint
        # (typical_gap) path where min_gutter_fraction controls the threshold.
        # contained left xmax values: [430, 432, 435] → median=432
        # typical_gap = 500 − 432 = 68px = 6.8% of page_width=1000.
        # ratio = 432/500 = 0.864 < 0.90, so the close-boundary path is off.
        # 6.8% < default 12% → single column.
        # With min_gutter_fraction=0.05 (5%) → 6.8% >= 5% → two columns.
        nodes = [
            _make_line("l0", xmin=50, xmax=430, ymin=10),
            _make_line("l1", xmin=55, xmax=435, ymin=100),
            _make_line("l2", xmin=52, xmax=432, ymin=200),
            _make_line("l3", xmin=58, xmax=510, ymin=300),
            _make_line("r0", xmin=500, xmax=900, ymin=10),
            _make_line("r1", xmin=502, xmax=902, ymin=100),
        ]
        _, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 1  # 6.8% < default 12%

        _, count, _ = cluster_columns(
            nodes, page_width=1000, min_gutter_fraction=0.05
        )
        assert count == 2  # 6.8% >= 5%

    def test_custom_variance_threshold_prevents_split(self):
        # Same data as above: variance_reduction ≈ 0.69. A stricter threshold
        # of 0.8 is not met and the clusters are not strictly disjoint, so
        # count=1.
        nodes = [
            _make_line("l0", xmin=0,   xmax=350, ymin=10),
            _make_line("l1", xmin=200, xmax=380, ymin=100),
            _make_line("r0", xmin=300, xmax=500, ymin=10),
            _make_line("r1", xmin=500, xmax=700, ymin=100),
        ]
        _, count, _ = cluster_columns(
            nodes, page_width=1000, variance_threshold=0.8
        )
        assert count == 1

    def test_within_column_y_sort_preserved(self):
        # Right column lines deliberately given out-of-order ymin values.
        nodes = [
            _make_line("l0", xmin=50,  xmax=400, ymin=10),
            _make_line("r2", xmin=600, xmax=900, ymin=300),
            _make_line("l1", xmin=52,  xmax=402, ymin=200),
            _make_line("r0", xmin=602, xmax=902, ymin=10),
            _make_line("r1", xmin=601, xmax=901, ymin=150),
        ]
        labels, count, _ = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels == ["l0", "l1", "r0", "r1", "r2"]


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

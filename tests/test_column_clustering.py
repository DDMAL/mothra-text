"""Tests for steps.column_clustering."""

import logging
from unittest.mock import MagicMock

from steps.column_clustering import cluster_columns


def _make_line(label: str, xmin: int, xmax: int, ymin: int) -> MagicMock:
    node = MagicMock()
    node.label = label
    node.bbox.xmin = xmin
    node.bbox.xmax = xmax
    node.bbox.ymin = ymin
    return node


class TestClusterColumns:
    def test_empty(self):
        labels, count = cluster_columns([], page_width=1000)
        assert labels == []
        assert count == 0

    def test_single_node(self):
        node = _make_line("line_0", xmin=50, xmax=400, ymin=10)
        labels, count = cluster_columns([node], page_width=1000)
        assert labels == ["line_0"]
        assert count == 1

    def test_single_column_sorted_by_y(self):
        nodes = [
            _make_line("line_2", xmin=50, xmax=400, ymin=200),
            _make_line("line_0", xmin=52, xmax=402, ymin=10),
            _make_line("line_1", xmin=51, xmax=401, ymin=100),
        ]
        labels, count = cluster_columns(nodes, page_width=1000)
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
        labels, count = cluster_columns(nodes, page_width=1000)
        assert count == 2
        # Left column top-to-bottom, then right column top-to-bottom
        assert labels == ["l0", "l1", "r1", "r0"]

    def test_tight_clusters_detected_regardless_of_gap_size(self):
        # Columns with only 200px separation on a 1000px page —
        # gap-threshold approach would miss this; variance ratio catches it.
        nodes = [
            _make_line("l0", xmin=48,  xmax=220, ymin=10),
            _make_line("l1", xmin=50,  xmax=222, ymin=100),
            _make_line("l2", xmin=52,  xmax=224, ymin=200),
            _make_line("r0", xmin=250, xmax=480, ymin=10),
            _make_line("r1", xmin=252, xmax=482, ymin=100),
            _make_line("r2", xmin=248, xmax=478, ymin=200),
        ]
        labels, count = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels[:3] == ["l0", "l1", "l2"]
        assert labels[3:] == ["r0", "r1", "r2"]

    def test_disjoint_extents_sufficient_without_variance(self):
        # Clusters are disjoint (left xmax < right xmin) but the variance
        # reduction is low because we only have 2 nodes.  The disjointness
        # check alone should declare two columns.
        nodes = [
            _make_line("l0", xmin=50, xmax=200, ymin=10),
            _make_line("r0", xmin=300, xmax=500, ymin=10),
        ]
        labels, count = cluster_columns(
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
        labels, count = cluster_columns(nodes, page_width=1000)
        assert count == 1
        assert labels == ["l0", "l1", "l2"]

    def test_identical_xmins_single_column(self):
        nodes = [
            _make_line("a", xmin=100, xmax=800, ymin=10),
            _make_line("b", xmin=100, xmax=800, ymin=100),
        ]
        labels, count = cluster_columns(nodes, page_width=1000)
        assert count == 1

    def test_uncertain_detection_logs_warning(self, caplog):
        # Two clusters with centroids 300px apart (> 5% of 1000px) and
        # variance_reduction ≈ 0.69 — above threshold=0.5 but below
        # 1.5×0.5=0.75, so the uncertain warning fires.  Not strictly disjoint
        # because left xmax (380) > right xmin (300), but approximately
        # disjoint: both left lines have xmax < median(right xmin)=400.
        nodes = [
            _make_line("l0", xmin=0,   xmax=350, ymin=10),
            _make_line("l1", xmin=200, xmax=380, ymin=100),
            _make_line("r0", xmin=300, xmax=500, ymin=10),
            _make_line("r1", xmin=500, xmax=700, ymin=100),
        ]
        with caplog.at_level(logging.WARNING, logger="steps.column_clustering"):
            _, count = cluster_columns(
                nodes, page_width=1000, variance_threshold=0.5
            )
        assert count == 2
        assert any("uncertain" in r.message for r in caplog.records)

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
        _, count = cluster_columns(
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
        labels, count = cluster_columns(nodes, page_width=1000)
        assert count == 2
        assert labels == ["l0", "l1", "r0", "r1", "r2"]

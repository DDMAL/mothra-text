"""Reading-order column clustering and co-linear segment fusion.

Detects whether a page has one or two columns by analysing the distribution
of line left-edge x-coordinates, then returns line node labels in reading
order (left column top-to-bottom, then right column top-to-bottom).

Two independent signals are used, either sufficient to declare two columns:
  1. Bimodal xmin variance ratio: the optimal 1D two-cluster split explains
     a large fraction of total xmin variance — i.e. lines form two tight
     clusters of start positions regardless of how close the columns are.
  2. Disjoint horizontal extents: every line in the left cluster ends before
     any line in the right cluster begins — a direct geometric test that does
     not depend on blank space between columns.

``fuse_colinear_segments`` groups segments that belong to the same physical
text line (identified by y-extent overlap) into logical ``FusedLine``
objects, correcting over-segmentation from BLLA on chant manuscripts.
"""

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusedLine:
    """A logical text line formed by fusing one or more adjacent segments.

    Constituents are the original line-node segments, sorted left-to-right
    by xmin within the physical line.
    """

    label: str            # synthetic label: "fused_0", "fused_1", ...
    constituent_labels: list  # original node labels sorted by xmin
    constituent_widths: list  # pixel width (xmax-xmin) per constituent
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    column: int           # 1 (left/only column) or 2 (right column)


def cluster_columns(
    line_nodes: list,
    page_width: int,
    variance_threshold: float = 0.5,
) -> tuple:
    """Return line node labels in reading order and the detected column count.

    Args:
        line_nodes:         Line-level nodes from page.children after
                            KrakenSegmentation.  Each node must expose
                            ``bbox.xmin``, ``bbox.xmax``, ``bbox.ymin``,
                            and ``label``.
        page_width:         Image width in pixels (``page.image.shape[1]``).
                            Reserved for future full-width line detection.
        variance_threshold: Minimum fraction of xmin variance that the
                            two-cluster split must explain over the single-
                            cluster baseline for the page to be treated as
                            two-column.  Default 0.5.

    Returns:
        ``(sorted_labels, column_count, split_x)`` where ``sorted_labels``
        is the list of ``node.label`` strings in left-column-first,
        top-to-bottom reading order, ``column_count`` is 1 or 2, and
        ``split_x`` is the x-coordinate of the column split (``None`` for
        single-column pages).
    """
    if not line_nodes:
        return [], 0, None
    if len(line_nodes) == 1:
        return [line_nodes[0].label], 1, None

    xmins = np.array([node.bbox.xmin for node in line_nodes], dtype=float)
    n = len(xmins)
    total_var = float(np.var(xmins))

    two_column = False
    split_x = None

    if total_var > 0:
        sorted_x = np.sort(xmins)
        best_combined_var = total_var
        best_split_idx = None

        for i in range(1, n):
            left_var = float(np.var(sorted_x[:i])) if i > 1 else 0.0
            right_var = float(np.var(sorted_x[i:])) if n - i > 1 else 0.0
            combined = (i * left_var + (n - i) * right_var) / n
            if combined < best_combined_var:
                best_combined_var = combined
                best_split_idx = i

        if best_split_idx is not None:
            variance_reduction = 1.0 - best_combined_var / total_var
            candidate_split_x = sorted_x[best_split_idx]

            left_cands = [
                nd for nd in line_nodes
                if nd.bbox.xmin < candidate_split_x
            ]
            right_cands = [
                nd for nd in line_nodes
                if nd.bbox.xmin >= candidate_split_x
            ]

            centroid_distance = (
                float(np.mean([nd.bbox.xmin for nd in right_cands]))
                - float(np.mean([nd.bbox.xmin for nd in left_cands]))
            ) if left_cands and right_cands else 0.0

            # Require centroids at least 5% of page width apart to avoid
            # trivially small splits being misclassified as two columns.
            min_separation = 0.05 * page_width

            disjoint = False
            approximately_disjoint = False
            if left_cands and right_cands:
                disjoint = (
                    max(nd.bbox.xmax for nd in left_cands)
                    < min(nd.bbox.xmin for nd in right_cands)
                )
                # At least 75% of left lines must end before the median
                # right-cluster xmin. Uses median (not minimum) to tolerate
                # one right-column line that starts unusually far left.
                right_xmin_median = float(
                    np.median([nd.bbox.xmin for nd in right_cands])
                )
                frac_contained = sum(
                    1 for nd in left_cands
                    if nd.bbox.xmax < right_xmin_median
                ) / len(left_cands)
                approximately_disjoint = frac_contained >= 0.75

            if (
                (
                    variance_reduction >= variance_threshold
                    and approximately_disjoint
                )
                or disjoint
            ) and centroid_distance >= min_separation:
                two_column = True
                split_x = candidate_split_x

                if (
                    variance_reduction >= variance_threshold
                    and variance_reduction < variance_threshold * 1.5
                    and not disjoint
                ):
                    logger.warning(
                        "Column detection uncertain for this page "
                        "(variance_reduction=%.2f, threshold=%.2f); "
                        "review result manually",
                        variance_reduction,
                        variance_threshold,
                    )

    if two_column:
        left_nodes = [nd for nd in line_nodes if nd.bbox.xmin < split_x]
        right_nodes = [nd for nd in line_nodes if nd.bbox.xmin >= split_x]
        sorted_nodes = (
            sorted(left_nodes, key=lambda nd: nd.bbox.ymin)
            + sorted(right_nodes, key=lambda nd: nd.bbox.ymin)
        )
        column_count = 2
        logger.info("Detected 2 columns (split at x=%.0f)", split_x)
    else:
        sorted_nodes = sorted(line_nodes, key=lambda nd: nd.bbox.ymin)
        column_count = 1
        logger.info("Detected 1 column")

    return [nd.label for nd in sorted_nodes], column_count, split_x


def fuse_colinear_segments(
    line_nodes: list,
    split_x,
    overlap_threshold: float = 0.5,
) -> list:
    """Group co-linear segments into logical lines using y-extent overlap.

    Segments in different columns are never fused. Within a column,
    segments whose y-extents overlap by at least
    ``overlap_threshold * min(height_a, height_b)`` are merged into one
    ``FusedLine``. Constituents within a group are sorted left-to-right
    by xmin. Groups are returned in reading order (left column
    top-to-bottom, then right column top-to-bottom).

    Args:
        line_nodes:        Line-level nodes (must expose bbox.xmin,
                           bbox.xmax, bbox.ymin, bbox.ymax, label).
        split_x:           x-coordinate of the column split, or ``None``
                           for single-column pages.
        overlap_threshold: Minimum overlap fraction (relative to the
                           shorter segment) to consider two segments
                           co-linear. Default 0.5 (50%).

    Returns:
        List of ``FusedLine`` objects in reading order.
    """
    if not line_nodes:
        return []

    # Assign each node to a column.
    def _col(node):
        return 1 if (split_x is None or node.bbox.xmin < split_x) else 2

    col1 = [nd for nd in line_nodes if _col(nd) == 1]
    col2 = [nd for nd in line_nodes if _col(nd) == 2]

    fused_lines = []
    counter = 0

    for column_id, column_nodes in ((1, col1), (2, col2)):
        if not column_nodes:
            continue
        sorted_by_y = sorted(column_nodes, key=lambda nd: nd.bbox.ymin)

        # Greedy grouping by y-overlap.
        groups = []
        current_group = [sorted_by_y[0]]
        group_ymin = sorted_by_y[0].bbox.ymin
        group_ymax = sorted_by_y[0].bbox.ymax

        for nd in sorted_by_y[1:]:
            nd_height = nd.bbox.ymax - nd.bbox.ymin
            group_height = group_ymax - group_ymin
            overlap_h = max(
                0, min(group_ymax, nd.bbox.ymax) - max(group_ymin, nd.bbox.ymin)
            )
            min_height = min(nd_height, group_height) or 1
            overlap_frac = overlap_h / min_height

            if overlap_frac >= overlap_threshold:
                current_group.append(nd)
                group_ymin = min(group_ymin, nd.bbox.ymin)
                group_ymax = max(group_ymax, nd.bbox.ymax)
            else:
                groups.append(current_group)
                current_group = [nd]
                group_ymin = nd.bbox.ymin
                group_ymax = nd.bbox.ymax

        groups.append(current_group)

        # Build a FusedLine per group, sorting constituents by xmin.
        column_fused = []
        for group in groups:
            ordered = sorted(group, key=lambda nd: nd.bbox.xmin)
            column_fused.append(FusedLine(
                label=f"fused_{counter}",
                constituent_labels=[nd.label for nd in ordered],
                constituent_widths=[
                    nd.bbox.xmax - nd.bbox.xmin for nd in ordered
                ],
                xmin=min(nd.bbox.xmin for nd in ordered),
                xmax=max(nd.bbox.xmax for nd in ordered),
                ymin=min(nd.bbox.ymin for nd in ordered),
                ymax=max(nd.bbox.ymax for nd in ordered),
                column=column_id,
            ))
            counter += 1

        # Sort this column's fused lines by ymin before appending.
        fused_lines.extend(
            sorted(column_fused, key=lambda fl: fl.ymin)
        )

    return fused_lines

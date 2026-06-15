"""Reading-order column clustering and co-linear segment fusion.

Detects whether a page has one or two columns using a horizontal
coverage-profile bimodal test.  A 1D array records, for each pixel column
x, how many line bounding boxes cover that x position.  The search is
scoped to the inner 20–80 % of the *text region* (min xmin → max xmax),
which excludes blank page margins that would otherwise produce false gaps.
A genuine 2-column page shows two distinct peaks (one per column) separated
by a valley; a 1-column page shows a single continuous plateau.

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
    bimodal_threshold: float = 0.5,
    min_gutter_fraction: float = 0.02,
    min_peak_count: int = 2,
    min_column_fraction: float = 0.15,
) -> tuple:
    """Return line node labels in reading order and the detected column count.

    Uses a horizontal coverage-profile bimodal test.  For each pixel column
    x, the coverage value is the number of line bounding boxes that include
    that x position.  The search is scoped to the inner 20–80 % of the text
    region (min xmin → max xmax) to exclude blank page margins.

    Two columns are declared when the coverage profile has a valley between
    two substantial peaks — i.e. the profile is bimodal within the text
    region.  A single-column page shows a continuous plateau; the test fails
    and 1 column is returned.

    BLLA spanning-bbox artefacts (where BLLA draws one bbox around lines in
    both columns at the same y-position) only slightly elevate the valley
    and do not eliminate the bimodal structure provided they are a minority
    of the total lines, which is the typical case.

    Args:
        line_nodes:           Line-level nodes from page.children after
                              KrakenSegmentation.  Each node must expose
                              ``bbox.xmin``, ``bbox.xmax``, ``bbox.ymin``,
                              and ``label``.
        page_width:           Image width in pixels (``page.image.shape[1]``).
                              Used only for the minimum gutter width check.
        bimodal_threshold:    Maximum ratio of valley coverage to the smaller
                              column peak for the valley to be treated as a
                              genuine gutter.  Default 0.5 (valley must be
                              less than half the height of the shorter peak).
        min_gutter_fraction:  Minimum gutter width as a fraction of
                              ``page_width``.  Default 0.02 (2 %).
        min_peak_count:       Minimum coverage value each column peak must
                              reach.  Guards against splitting a page with
                              only a handful of detected lines or where a
                              few outlier segments (initials, neumes) create
                              a spurious left-side peak.  Default 2.
        min_column_fraction:  Minimum fraction of all line nodes whose xmin
                              must fall on each side of the candidate split.
                              Rejects splits where one apparent "column"
                              contains only a small minority of the total
                              lines (e.g. isolated margin fragments).
                              Default 0.15 (15 %).

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

    # Build 1-D coverage array: coverage[x] = # line bboxes covering column x.
    coverage = np.zeros(page_width, dtype=int)
    for nd in line_nodes:
        x0 = max(0, nd.bbox.xmin)
        x1 = min(page_width, nd.bbox.xmax)
        if x1 > x0:
            coverage[x0:x1] += 1

    # Smooth with a 5-px window to suppress single-pixel gaps from polygon
    # boundaries that don't quite touch.
    kernel = np.ones(5) / 5
    smooth = np.convolve(coverage.astype(float), kernel, mode="same")

    # Scope the search to the inner 20–80 % of the text region so that blank
    # page margins are excluded.
    text_left = min(nd.bbox.xmin for nd in line_nodes)
    text_right = max(nd.bbox.xmax for nd in line_nodes)
    text_width = text_right - text_left
    search_start = text_left + int(0.20 * text_width)
    search_end = text_right - int(0.20 * text_width)

    two_column = False
    split_x = None

    if search_end > search_start:
        band = smooth[search_start:search_end]
        # Valley: x position of the minimum within the search band.
        valley_offset = int(np.argmin(band))
        valley_x = search_start + valley_offset
        valley_val = float(smooth[valley_x])

        # Peaks: maximum coverage on each side of the valley within the text
        # region.  Using text_left/text_right (not search boundaries) gives
        # the full column peak even when the valley is near the band edge.
        left_peak = float(smooth[text_left:valley_x].max()) if valley_x > text_left else 0.0
        right_peak = float(smooth[valley_x + 1:text_right].max()) if valley_x + 1 < text_right else 0.0
        min_peak = min(left_peak, right_peak)

        # Bimodal test: valley must be significantly lower than both peaks,
        # and each peak must meet the minimum density requirement.
        if (
            min_peak >= min_peak_count
            and valley_val < bimodal_threshold * min_peak
        ):
            # Gutter width: contiguous run of smooth values below the midpoint
            # between the valley and the bimodal threshold × min_peak.
            gutter_threshold = valley_val + (bimodal_threshold * min_peak - valley_val) * 0.5
            in_gutter = smooth <= gutter_threshold

            # Walk left from valley_x to find gutter left edge.
            gutter_left = valley_x
            while gutter_left > text_left and in_gutter[gutter_left - 1]:
                gutter_left -= 1

            # Walk right from valley_x to find gutter right edge.
            gutter_right = valley_x
            while gutter_right + 1 < text_right and in_gutter[gutter_right + 1]:
                gutter_right += 1

            gutter_width = gutter_right - gutter_left
            candidate_split = float((gutter_left + gutter_right) / 2)
            left_count = sum(
                1 for nd in line_nodes if nd.bbox.xmin < candidate_split
            )
            right_count = sum(
                1 for nd in line_nodes if nd.bbox.xmin >= candidate_split
            )
            total = len(line_nodes)
            if (
                gutter_width >= min_gutter_fraction * page_width
                and left_count / total >= min_column_fraction
                and right_count / total >= min_column_fraction
            ):
                two_column = True
                split_x = candidate_split

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
                0,
                min(group_ymax, nd.bbox.ymax) - max(group_ymin, nd.bbox.ymin),
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

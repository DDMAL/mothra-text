"""Reading-order column clustering for manuscript folios.

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
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cluster_columns(
    line_nodes: list,
    page_width: int,
    variance_threshold: float = 0.5,
) -> tuple[list[str], int]:
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
        ``(sorted_labels, column_count)`` where ``sorted_labels`` is the list
        of ``node.label`` strings in left-column-first, top-to-bottom reading
        order and ``column_count`` is 1 or 2.
    """
    if not line_nodes:
        return [], 0
    if len(line_nodes) == 1:
        return [line_nodes[0].label], 1

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
                nd for nd in line_nodes if nd.bbox.xmin < candidate_split_x
            ]
            right_cands = [
                nd for nd in line_nodes if nd.bbox.xmin >= candidate_split_x
            ]

            centroid_distance = (
                float(np.mean([nd.bbox.xmin for nd in right_cands]))
                - float(np.mean([nd.bbox.xmin for nd in left_cands]))
            ) if left_cands and right_cands else 0.0

            # Require the two candidate centroids to be at least 5% of the page
            # width apart. This prevents trivially small splits (e.g. 3 lines
            # with xmins {50, 51, 52}) from being misclassified as two columns
            # even though their variance ratio may appear high.
            min_separation = 0.05 * page_width

            disjoint = False
            approximately_disjoint = False
            if left_cands and right_cands:
                disjoint = (
                    max(nd.bbox.xmax for nd in left_cands)
                    < min(nd.bbox.xmin for nd in right_cands)
                )
                # Require that most left-cluster lines are horizontally contained
                # within the left column: at least 75% of left lines must end
                # before the median start of the right cluster.  Using the median
                # (not the minimum) of right xmin tolerates one right-column line
                # that starts unusually far left without rejecting a real split.
                # This guards against split-line false positives where the "left
                # cluster" actually contains wide spanning lines extending far right.
                right_xmin_median = float(
                    np.median([nd.bbox.xmin for nd in right_cands])
                )
                frac_contained = sum(
                    1 for nd in left_cands if nd.bbox.xmax < right_xmin_median
                ) / len(left_cands)
                approximately_disjoint = frac_contained >= 0.75

            if (
                (variance_reduction >= variance_threshold and approximately_disjoint)
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

    return [nd.label for nd in sorted_nodes], column_count

"""Inject mothra text-detection lines missed by Kraken into a collection.

Runs after KrakenSegmentation. Groups mothra classId-1 (text) word-level
bboxes into line-level bboxes via y-overlap clustering, then adds any that
have low IoU overlap with existing Kraken lines as new SegmentNodes.
"""

import json
import logging
from pathlib import Path

import numpy as np
from htrflow.utils.geometry import Bbox as _HtBbox
from htrflow.volume.volume import ImageNode as _HtImageNode

logger = logging.getLogger(__name__)


class _MothraLineNode(_HtImageNode):
    """Synthetic htrflow leaf node created from a grouped mothra text bbox."""

    def __init__(
        self,
        parent,
        label: str,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        image_crop: np.ndarray,
    ):
        super().__init__(parent=parent, label=label)
        self._bbox_obj = _HtBbox(xmin, ymin, xmax, ymax)
        self._image = image_crop

    @property
    def bbox(self) -> _HtBbox:
        return self._bbox_obj

    def _load_image(self):
        return self._image

    def asdict(self) -> dict:
        from dataclasses import asdict as _dc_asdict
        return super().asdict() | {
            "segmentation_label": "region",
            "segmentation_confidence": 1.0,
            "bbox": _dc_asdict(self._bbox_obj),
            "polygon": str(self.polygon),
        }


def _iou(
    b1: tuple[int, int, int, int],
    b2: tuple[int, int, int, int],
) -> float:
    """Compute IoU between two (xmin, ymin, xmax, ymax) bboxes."""
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class MothraUnionStep:
    """Add mothra classId-1 lines that Kraken missed to the collection.

    Groups mothra word-level bboxes into line-level bboxes using the same
    y-overlap merge logic as fuse_colinear_segments(), then injects any
    grouped line that has low IoU with all existing Kraken line nodes.

    Args:
        mothra_json_path: Path to a mothra annotation JSON.
        iou_threshold: Maximum IoU with any Kraken line for a mothra line
            to be injected. Default 0.3.
        min_width: Minimum pixel width of a merged mothra line to be
            considered (filters isolated single-glyph detections). Default 50.
        overlap_threshold: Minimum y-overlap fraction (relative to shorter
            segment) to merge two mothra bboxes into one line. Default 0.5.
    """

    def __init__(
        self,
        mothra_json_path: str | Path,
        iou_threshold: float = 0.3,
        min_width: int = 50,
        overlap_threshold: float = 0.5,
    ):
        data = json.loads(Path(mothra_json_path).read_text(encoding="utf-8"))
        self._bboxes = [
            ann["bbox"]
            for ann in data["annotations"]
            if ann.get("classId") == 1
        ]
        self.iou_threshold = iou_threshold
        self.min_width = min_width
        self.overlap_threshold = overlap_threshold

    def _group_into_lines(self) -> list[tuple[int, int, int, int]]:
        """Cluster mothra word-bboxes into line-level bboxes by y-overlap."""
        if not self._bboxes:
            return []

        # Convert [x, y, w, h] → (xmin, ymin, xmax, ymax), sort by ymin.
        converted = sorted(
            [
                (
                    int(b[0]),
                    int(b[1]),
                    int(b[0] + b[2]),
                    int(b[1] + b[3]),
                )
                for b in self._bboxes
            ],
            key=lambda b: b[1],
        )

        groups: list[list[tuple]] = []
        current = [converted[0]]
        g_ymin, g_ymax = converted[0][1], converted[0][3]

        for bbox in converted[1:]:
            _, ymin, _, ymax = bbox
            height = ymax - ymin
            g_height = g_ymax - g_ymin
            overlap_h = max(0, min(g_ymax, ymax) - max(g_ymin, ymin))
            min_h = min(height, g_height) or 1
            if overlap_h / min_h >= self.overlap_threshold:
                current.append(bbox)
                g_ymin = min(g_ymin, ymin)
                g_ymax = max(g_ymax, ymax)
            else:
                groups.append(current)
                current = [bbox]
                g_ymin, g_ymax = ymin, ymax

        groups.append(current)

        return [
            (
                min(b[0] for b in g),
                min(b[1] for b in g),
                max(b[2] for b in g),
                max(b[3] for b in g),
            )
            for g in groups
        ]

    def run(self, collection):
        page = next(iter(collection))
        page_img = page.image  # BGR numpy array (H, W, 3)
        img_h, img_w = page_img.shape[:2]

        kraken_bboxes = [
            (nd.bbox.xmin, nd.bbox.ymin, nd.bbox.xmax, nd.bbox.ymax)
            for nd in page.children
        ]

        mothra_lines = self._group_into_lines()
        added = 0

        for line_bbox in mothra_lines:
            xmin, ymin, xmax, ymax = line_bbox
            if xmax - xmin < self.min_width:
                continue

            max_iou = max(
                (_iou(line_bbox, kb) for kb in kraken_bboxes),
                default=0.0,
            )
            if max_iou >= self.iou_threshold:
                continue

            # Clamp crop to image bounds before slicing.
            cx0 = max(0, xmin)
            cy0 = max(0, ymin)
            cx1 = min(img_w, xmax)
            cy1 = min(img_h, ymax)
            crop = page_img[cy0:cy1, cx0:cx1]

            node = _MothraLineNode(page, "mothra_line", xmin, ymin, xmax, ymax, crop)
            page.children.append(node)
            added += 1
            logger.debug(
                "Injected mothra line bbox=(%d,%d,%d,%d) max_iou=%.2f",
                xmin, ymin, xmax, ymax, max_iou,
            )

        if added:
            page.relabel()
            logger.info("MothraUnionStep: injected %d new line(s)", added)
        else:
            logger.info("MothraUnionStep: no new lines to inject")

        return collection

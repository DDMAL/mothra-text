"""Ground-truth-aware word segmentation for HTRflow pipelines.

Implements a custom PipelineStep that substitutes Cantus ground-truth text
for the recognised transcription when estimating word boundaries within a
line, while preserving HTRflow's existing pixels-per-character geometry.
"""

import logging
from collections.abc import Callable
from typing import Optional

from htrflow.postprocess.word_segmentation import _simple_word_segmentation
from htrflow.results import Result
from htrflow.utils.geometry import bbox2mask
from htrflow.utils.imgproc import mask as apply_mask
from htrflow.volume.volume import Collection, SegmentNode

try:
    from htrflow.pipeline.steps import WordSegmentation as _WordSegmentationBase
except ImportError:
    # htrflow.pipeline.steps fails on Apple Silicon because its module-level code
    # imports RTMDet → mmcv C extension, which has an incompatible symbol with the
    # installed PyTorch. This stub exposes the same interface so the module and tests
    # load cleanly in this environment.
    class _WordSegmentationBase:  # type: ignore[no-redef]
        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def run(self, collection: Collection) -> Collection:  # pragma: no cover
            raise NotImplementedError


logger = logging.getLogger(__name__)

GroundTruthLookup = Callable[[SegmentNode], Optional[str]]


def _ground_truth_word_segmentation(
    node: SegmentNode,
    gt_lookup: GroundTruthLookup,
) -> Optional[Result]:
    """Segment a line node into word regions using Cantus ground-truth text.

    Mirrors _simple_word_segmentation but sources word strings and count from
    gt_lookup rather than the node's recognised transcription. If gt_lookup
    returns None or an empty string, returns None and the caller decides the
    fallback behaviour.

    Args:
        node: Line-level SegmentNode to segment.
        gt_lookup: Callable that maps a node to its expanded Cantus
            transcription, or None if no ground truth is available.

    Returns:
        A word-segmentation Result if ground truth is available, else None.
    """
    gt_text = gt_lookup(node)
    if not gt_text:
        return None
    words = gt_text.split()
    if not words:
        return None
    pixels_per_char = node.width // len(gt_text)
    x1, x2 = 0, 0
    bboxes = []
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), node.width)
        bboxes.append((x1, 0, x2, node.height))
        x1 = x2
    node_mask = node.mask
    masks = [apply_mask(node_mask, bbox2mask(bbox, node_mask.shape), fill=0) for bbox in bboxes]
    return Result.word_segmentation_result(
        orig_shape=(node.height, node.width),
        metadata={},
        masks=masks,
        words=words,
    )


class GroundTruthWordSegmentation(_WordSegmentationBase):
    """HTRflow pipeline step: word segmentation driven by Cantus ground truth.

    Drop-in replacement for WordSegmentation. Instead of splitting the
    recognised transcription on whitespace, calls gt_lookup to obtain the
    expanded Cantus text and uses that to determine word count and identity.
    The pixels-per-character geometry is otherwise unchanged.

    When gt_lookup returns None or an empty string for a line, falls back to
    the standard recognition-based segmentation and logs a warning so the
    result list always has exactly one entry per active leaf.

    Args:
        gt_lookup: Callable mapping a SegmentNode to its expanded Cantus
            transcription (or None if unavailable). Passed at construction
            time so it can be tested in isolation and swapped without
            modifying segmentation logic.
    """

    def __init__(self, gt_lookup: GroundTruthLookup, **kwargs):
        super().__init__(**kwargs)
        self.gt_lookup = gt_lookup

    def run(self, collection: Collection) -> Collection:
        nodes = list(collection.active_leaves())
        results = []
        for node in nodes:
            result = _ground_truth_word_segmentation(node, self.gt_lookup)
            if result is None:
                logger.warning(
                    "No ground truth available for %s; falling back to recognition output",
                    node,
                )
                result = _simple_word_segmentation(node)
            results.append(result)
        collection.update(results)
        return collection

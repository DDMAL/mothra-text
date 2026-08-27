"""Ground-truth-aware word segmentation for HTRflow pipelines.

Implements a custom PipelineStep that substitutes Cantus ground-truth text
for the recognised transcription when estimating word boundaries within a
line, while preserving HTRflow's existing pixels-per-character geometry.
"""

import logging
from collections.abc import Callable
from typing import Optional

from steps.gt_manifest import (  # noqa: F401
    make_manifest_lookup,
)

from htrflow.results import Result
from htrflow.volume.volume import Collection, SegmentNode

try:
    from htrflow.pipeline.steps import WordSegmentation as _WordSegmentationBase
except ImportError:
    # htrflow.pipeline.steps fails on Apple Silicon because its module-level
    # code imports RTMDet → mmcv C extension, which has an incompatible symbol
    # with the installed PyTorch. This stub exposes the same interface so the
    # module and tests load cleanly in this environment.
    class _WordSegmentationBase:  # type: ignore[no-redef]
        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def run(  # pragma: no cover
            self, collection: Collection
        ) -> Collection:
            raise NotImplementedError


logger = logging.getLogger(__name__)

GroundTruthLookup = Callable[[SegmentNode], Optional[str]]


def _bbox_word_segmentation(
    node: SegmentNode,
    words: list[str],
    text: str,
    source: str = "fallback",
) -> Result:
    """Divide a line node into per-word bounding boxes by character width.

    Uses pixels-per-character geometry (node.width / len(text)) to allocate
    horizontal space for each word.  Returns a word-segmentation Result with
    bbox-based segments (no mask required).

    Each segment's data is stamped with `source` (e.g. "gt" or "fallback")
    so that downstream consumers (e.g. _build_pipeline_payload) can read
    provenance directly off the resulting word node instead of re-deriving
    it later from a line label, which may have since shifted due to
    Collection.relabel() (see mothra-text#59).
    """
    chars = max(len(text), 1)
    pixels_per_char = max(1, node.width // chars)
    x1, x2 = 0, 0
    bboxes = []
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), node.width)
        bboxes.append((x1, 0, x2, node.height))
        x1 = x2
    result = Result.word_segmentation_result(
        words=words,
        orig_shape=(node.height, node.width),
        metadata={},
        bboxes=bboxes,
    )
    for segment in result.segments:
        segment.data["source"] = source
    return result


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
    return _bbox_word_segmentation(node, words, gt_text, source="gt")


def _fallback_word_segmentation(node: SegmentNode) -> Result:
    """Bbox-based word segmentation from the node's recognised text.

    Drop-in replacement for HTRflow's _simple_word_segmentation that avoids
    node.mask (which is not a public SegmentNode attribute for Kraken nodes).
    """
    text = node.text or ""
    words = text.split()
    if not words:
        words = [text] if text else [""]
    return _bbox_word_segmentation(node, words, text or " ", source="fallback")


class GroundTruthWordSegmentation(_WordSegmentationBase):
    """HTRflow pipeline step: word segmentation driven by Cantus ground truth.

    Drop-in replacement for WordSegmentation. Instead of splitting the
    recognised transcription on whitespace, calls gt_lookup to obtain the
    expanded Cantus text and uses that to determine word count and identity.
    The pixels-per-character geometry is otherwise unchanged.

    When gt_lookup returns None or an empty string for a line, falls back to
    recognition-based segmentation and logs a warning so the result list
    always has exactly one entry per active leaf.

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
                    "No ground truth for %s; falling back to"
                    " recognition output",
                    node,
                )
                result = _fallback_word_segmentation(node)
            results.append(result)
        collection.update(results)
        return collection

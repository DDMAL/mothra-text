"""Syllable segmentation for HTRflow pipelines.

Implements a custom PipelineStep that subdivides word-level nodes into
syllable-level nodes using Cantus syllabification via volpiano-display-utilities,
with character-proportional bounding-box geometry.
"""

import logging
import re
import unicodedata
from typing import Optional

from htrflow.results import TEXT_RESULT_KEY, Result
from htrflow.utils.geometry import bbox2mask
from htrflow.utils.imgproc import mask as apply_mask
from htrflow.volume.volume import Collection, SegmentNode
from volpiano_display_utilities.latin_word_syllabification import (
    LatinError,
    split_word_by_syl_bounds,
    syllabify_word,
)

try:
    from htrflow.pipeline.steps import PipelineStep as _PipelineStepBase
except ImportError:
    # htrflow.pipeline.steps fails on Apple Silicon because its module-level code
    # imports RTMDet → mmcv C extension, which has an incompatible symbol with the
    # installed PyTorch. This stub exposes the same interface so the module and tests
    # load cleanly in this environment.
    class _PipelineStepBase:  # type: ignore[no-redef]
        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def run(self, collection: Collection) -> Collection:  # pragma: no cover
            raise NotImplementedError


logger = logging.getLogger(__name__)


def normalize_word_text(text: str) -> str:
    """Normalize a word string to plain lowercase ASCII alphabetic characters.

    Applies NFKD decomposition and ASCII transliteration to handle accented
    vowels and ligatures (æ, œ, etc.) from Cantus fulltext_ms fields, then
    strips any remaining non-alphabetic characters.

    Args:
        text: Raw word string, potentially containing non-ASCII or
            non-alphabetic characters.

    Returns:
        Lowercased ASCII alphabetic string. Empty string if no alphabetic
        characters survive normalization; the caller handles that case.
    """
    if not text.isascii():
        logger.warning(
            "Non-ASCII characters present in word text before normalization: %r",
            text,
        )
    transliterated = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    alphabetic = re.sub(r"[^a-zA-Z]", "", transliterated)
    return alphabetic.lower()


def syllabify(text: str) -> list[str]:
    """Syllabify a word string using volpiano-display-utilities.

    Normalizes the input with normalize_word_text, then delegates to
    syllabify_word from latin_word_syllabification. This is the only
    permitted source of syllabification logic.

    Args:
        text: Word string to syllabify. May contain non-ASCII or
            non-alphabetic characters; these are normalized away first.

    Returns:
        Ordered list of syllable strings. Non-final syllables carry a
        trailing hyphen (e.g. ["do-", "mi-", "nus"]). Returns a
        one-element list in all fallback cases (empty normalization,
        single-syllable word, LatinError).
    """
    normalized = normalize_word_text(text)
    if not normalized:
        logger.error(
            "Word text %r normalized to empty string; treating as single syllable",
            text,
        )
        return [text]
    try:
        bounds = syllabify_word(normalized, return_string=False)
    except LatinError as exc:
        logger.error(
            "LatinError syllabifying %r: %s; treating as single syllable",
            normalized,
            exc,
        )
        return [normalized]
    # syllabify_word returns [] for single-syllable words;
    # split_word_by_syl_bounds([]) correctly returns [word].
    return split_word_by_syl_bounds(normalized, bounds)


def _syllable_segmentation(node: SegmentNode) -> Result:
    """Subdivide a word node into character-proportional syllable regions.

    Uses syllabify() to determine syllables, then lays out bounding boxes
    within the word node's coordinate space using the same mask-based geometry
    pattern as GroundTruthWordSegmentation. Always produces at least one child
    region, even for empty or unnormalizable text.

    Args:
        node: Word-level SegmentNode to subdivide.

    Returns:
        A Result whose segments are the syllable regions with text attached.
    """
    text = node.text or ""
    syllables = syllabify(text)

    char_count = sum(len(s) for s in syllables)

    cursor = 0
    bboxes = []
    for i, syllable in enumerate(syllables):
        if i == len(syllables) - 1:
            x2 = node.width
        else:
            syl_width = node.width * len(syllable) // max(char_count, 1)
            x2 = cursor + syl_width
        bboxes.append((cursor, 0, x2, node.height))
        cursor = x2

    node_mask = node.mask
    masks = [
        apply_mask(node_mask, bbox2mask(bbox, node_mask.shape), fill=0)
        for bbox in bboxes
    ]
    return Result.word_segmentation_result(
        orig_shape=(node.height, node.width),
        metadata={},
        masks=masks,
        words=syllables,
    )


class SyllableSegmentation(_PipelineStepBase):
    """HTRflow pipeline step: character-proportional syllable segmentation.

    Operates on word-level active leaves (produced by GroundTruthWordSegmentation
    or HTRflow's WordSegmentation). For each word node, syllabifies its text via
    volpiano-display-utilities and subdivides the word bounding box
    character-proportionally across the resulting syllables.

    Always produces at least one child per word node. Syllable strings match
    the output of volpiano-display-utilities (non-final syllables carry a
    trailing hyphen).
    """

    def run(self, collection: Collection) -> Collection:
        nodes = list(collection.active_leaves())
        results = [_syllable_segmentation(node) for node in nodes]
        collection.update(results)
        return collection

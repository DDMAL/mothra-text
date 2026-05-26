"""HTRflow pipeline step: Kraken BLLA line segmentation.

Wraps Kraken's BLLA segmenter as an HTRflow PipelineStep so it can be
used in the mothra-text PoC pipeline in place of HTRflow's native YOLO
or RTMDet segmentation steps.

Each page in the Collection is segmented independently. Lines whose
boundary polygon is None are intentionally skipped — they cannot produce
a geometrically valid SegmentNode. Note: run_kraken.py preserves
None-boundary lines in its evaluation JSON for mothra-evaluator line
count metrics; that divergence is intentional.
"""

import logging

import cv2
from kraken import blla
from PIL import Image

from htrflow.results import Result
from htrflow.volume.volume import Collection

try:
    from htrflow.pipeline.steps import PipelineStep as _PipelineStepBase
except ImportError:
    # htrflow.pipeline.steps fails on Apple Silicon because its module-level
    # code imports RTMDet → mmcv C extension with an incompatible symbol.
    class _PipelineStepBase:  # type: ignore[no-redef]
        def run(self, collection):  # pragma: no cover
            raise NotImplementedError

logger = logging.getLogger(__name__)


class KrakenSegmentation(_PipelineStepBase):
    """HTRflow pipeline step: line segmentation via Kraken BLLA.

    Drop-in replacement for HTRflow's Segmentation step when Kraken is
    preferred over YOLO or RTMDet. Calls blla.segment() on each page
    image in the Collection and updates it with the detected line polygons.

    Args:
        device: Kraken inference device string, e.g. ``"cpu"`` or
            ``"cuda"``. Defaults to ``"cpu"``.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def run(self, collection: Collection) -> Collection:
        results = []
        for page in collection:
            # HTRflow loads images as BGR (cv2.imread); convert to RGB for PIL.
            bgr = page.image
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            seg = blla.segment(pil_img, device=self.device)

            polygons = [
                line.boundary
                for line in seg.lines
                if line.boundary is not None
            ]
            n_skipped = len(seg.lines) - len(polygons)
            if n_skipped:
                logger.warning(
                    "Skipped %d line(s) with no boundary polygon on page %s",
                    n_skipped,
                    page.label,
                )

            shape = (bgr.shape[0], bgr.shape[1])
            results.append(
                Result.segmentation_result(shape, {}, polygons=polygons)
            )
            logger.info(
                "Segmented %s: %d lines (%d skipped)",
                page.label,
                len(polygons),
                n_skipped,
            )

        collection.update(results)
        return collection

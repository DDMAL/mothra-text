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
import threading

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


# The BLLA model, loaded once per process rather than once per page.
#
# blla.segment(model=None) loads kraken's bundled default INSIDE the call --
# `model = vgsl.TorchVGSLModel.load_model(resources.files('kraken')
# .joinpath('blla.mlmodel'))` -- so every page was paying a full model load
# before any inference happened. _default_segmentation_model() reproduces
# that exact resolution once and hands the result to blla.segment(), which
# then skips its own load.
#
# Lock discipline matches kraken_recognition.py's: held across the whole
# segment() call. torch modules in eval mode are usually safe to share, but
# blla.segment() is not documented as thread-safe and does touch the model
# object, and serializing costs nothing in the deployment this serves
# (text-service: replicas=1, no --workers, one image at a time).
_SEGMENTER_CACHE: dict[str, object] = {}
_SEGMENTER_LOCK = threading.Lock()


def _default_segmentation_model():
    """kraken's bundled BLLA model, memoized.

    Returns None on any failure, which makes the caller fall through to
    blla.segment(model=None) -- i.e. exactly the previous behaviour, loading
    per page. This is a performance optimization reaching into another
    project's resource layout, so it must degrade rather than break if a
    kraken upgrade moves things.
    """
    cached = _SEGMENTER_CACHE.get("default")
    if cached is not None:
        return cached
    try:
        from importlib import resources

        from kraken.lib import vgsl

        model = vgsl.TorchVGSLModel.load_model(
            resources.files("kraken").joinpath("blla.mlmodel")
        )
    except Exception:
        logger.warning(
            "KrakenSegmentation: could not pre-load kraken's default BLLA model; "
            "falling back to loading it per page",
            exc_info=True,
        )
        return None
    _SEGMENTER_CACHE["default"] = model
    return model


class KrakenSegmentation(_PipelineStepBase):
    """HTRflow pipeline step: line segmentation via Kraken BLLA.

    Drop-in replacement for HTRflow's Segmentation step when Kraken is
    preferred over YOLO or RTMDet. Calls blla.segment() on each page
    image in the Collection and updates it with the detected line polygons.

    Args:
        device: Kraken inference device string, e.g. ``"cpu"`` or
            ``"cuda"``. Defaults to ``"cpu"``.
        model: Optional path to a custom BLLA segmentation model. Accepts
            ``.mlmodel`` (CoreML) or ``.safetensors`` format. When ``None``
            (default) Kraken's built-in BLLA model is used.
    """

    def __init__(self, device: str = "cpu", model: str = None):
        self.device = device
        self._model = None
        if model:
            import os
            ext = os.path.splitext(model)[1].lower()
            if ext == '.safetensors':
                from kraken.models.loaders import load_safetensors
                self._model = load_safetensors(model)[0]
            else:
                from kraken.lib import vgsl
                self._model = vgsl.TorchVGSLModel.load_model(model)
            if 'hyper_params' not in self._model.user_metadata:
                self._model.user_metadata['hyper_params'] = {}
            logger.info("Loaded custom segmentation model: %s", model)

    def run(self, collection: Collection) -> Collection:
        results = []
        # A custom model was already loaded once in __init__; only the
        # default needs the process-wide cache.
        model = self._model if self._model is not None else _default_segmentation_model()
        for page in collection:
            # HTRflow loads images as BGR (cv2.imread); convert to RGB for PIL.
            bgr = page.image
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with _SEGMENTER_LOCK:
                seg = blla.segment(pil_img, model=model, device=self.device)

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

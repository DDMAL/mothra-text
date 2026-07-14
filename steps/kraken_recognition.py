"""HTRflow pipeline step: Kraken HTR text recognition.

Wraps Kraken's recognition engine as an HTRflow PipelineStep so that any
Kraken-compatible model can be dropped into the mothra-text pipeline with a
single parameter change.

When ``model=None`` (the default) the step runs in stub mode: all line nodes
receive empty text and the pipeline continues unchanged.  This lets every
downstream step — including GroundTruthWordSegmentation and the future
NWChantAllocator — run end-to-end while the fine-tuned recognition model is
still being developed.  To activate real OCR, pass a HuggingFace model ID or
a local path to a ``.mlmodel`` file via ``model=`` (or ``--recognition-model``
from the CLI).
"""

import logging

import cv2
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


class KrakenRecognition(_PipelineStepBase):
    """HTRflow pipeline step: text recognition via Kraken HTR.

    Runs a Kraken HTR model on each line node's image crop and sets
    ``node.text`` with the recognised transcription.

    When ``model=None`` (stub mode) all line nodes receive empty text and
    a WARNING is logged.  The pipeline continues normally; downstream steps
    that rely on ``node.text`` (e.g. the future NWChantAllocator) will see
    empty strings until a real model is provided.

    Args:
        model: HuggingFace model ID or local path to a ``.mlmodel`` file
            accepted by ``kraken.lib.models.load_any``.  Pass ``None``
            (default) to run in stub mode.
        device: Kraken inference device string, e.g. ``"cpu"`` or
            ``"cuda"``.  Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
        allow_stub: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.allow_stub = allow_stub

    def run(self, collection: Collection) -> Collection:
        if self.model is None:
            if not self.allow_stub:
                raise ValueError(
                    "KrakenRecognition: no recognition model provided "
                    "and stub mode was not explicitly requested.\n"
                    "  • Install the Tridis model: "
                    "python -m htrmopo get 10.5281/zenodo.7899855\n"
                    "  • Or pass a model:           --recognition-model PATH\n"
                    "  • Or opt into stub mode:     --stub-mode"
                )
            logger.warning(
                "KrakenRecognition: stub mode active (allow_stub=True); "
                "setting empty text on all line nodes."
            )
            nodes = list(collection.active_leaves())
            if not nodes:
                return collection
            collection.update([
                Result.text_recognition_result({}, [""], [0.0])
                for _ in nodes
            ])
            return collection

        # Lazy import: keeps the stub path free of the Kraken dependency so
        # the pipeline can run in environments where only the segmentation
        # model is installed.
        from kraken import rpred
        from kraken.containers import BBoxLine, Segmentation
        from kraken.lib import models

        logger.info(
            "KrakenRecognition: loading model %r on device=%r",
            self.model,
            self.device,
        )
        nn = models.load_any(self.model, device=self.device)

        nodes = list(collection.active_leaves())
        if not nodes:
            logger.warning("KrakenRecognition: no active leaf nodes; nothing to recognise")
            return collection

        results = []
        for node in nodes:
            # node.image is a BGR numpy array (HTRflow convention).
            # Convert BGR → RGB → PIL → grayscale; Kraken HTR models expect
            # single-channel input.
            bgr = node.image
            pil_img = (
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("L")
            )
            h, w = bgr.shape[:2]

            # Reconstruct a single-line bbox Segmentation from the HTRflow
            # crop.  We do NOT re-run blla.segment() — node.image is already
            # the extracted line region.  imagename is a sentinel string;
            # Kraken uses it only for logging/serialisation, not for I/O.
            bounds = Segmentation(
                type="bbox",
                imagename=node.label,
                text_direction="horizontal-lr",
                script_detection=False,
                lines=[
                    BBoxLine(
                        id=node.label,
                        bbox=(0, 0, w, h),
                        text_direction="horizontal-lr",
                    )
                ],
            )

            records = list(rpred.rpred(nn, pil_img, bounds))
            if records:
                rec = records[0]
                text = rec.prediction
                conf = (
                    sum(rec.confidences) / len(rec.confidences)
                    if rec.confidences
                    else 0.0
                )
            else:
                text, conf = "", 0.0

            results.append(Result.text_recognition_result({}, [text], [conf]))
            logger.debug(
                "KrakenRecognition: %s → %r (conf=%.3f)", node.label, text, conf
            )

        collection.update(results)
        logger.info("KrakenRecognition: recognised %d lines", len(nodes))
        return collection

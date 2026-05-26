"""HTRflow pipeline step: PyLaia text recognition via subprocess.

PyLaia 1.x requires torch 1.13, which conflicts with the torch 2.x in
line-seg-eval. This step runs pylaia-htr-decode-ctc from a separate
pylaia-env conda environment via subprocess, matching the approach used
by experiments/pylaia_baseline/*/03_run_pylaia.py.

Crop extraction and resizing logic mirrors 02_extract_crops.py.
"""

import logging
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from htrflow.results import Result
from htrflow.volume.volume import Collection

try:
    from htrflow.pipeline.steps import PipelineStep as _PipelineStepBase
except ImportError:
    class _PipelineStepBase:  # type: ignore[no-redef]
        def run(self, collection):  # pragma: no cover
            raise NotImplementedError

logger = logging.getLogger(__name__)

_DEFAULT_PYLAIA_ENV = os.path.join(
    os.path.expanduser("~"), "miniconda3", "envs", "pylaia-env"
)
_TARGET_HEIGHT = 128


class PyLaiaRecognition(_PipelineStepBase):
    """HTRflow pipeline step: text recognition via PyLaia (subprocess).

    For each active leaf node (line segment), extracts a grayscale crop
    at 128px height and runs pylaia-htr-decode-ctc from the pylaia-env
    conda environment. Returns one text recognition result per node.

    Args:
        model: HuggingFace model ID, e.g. ``"Teklia/pylaia-home-alcar"``.
        pylaia_env: Path to the pylaia conda environment. Defaults to
            ``~/miniconda3/envs/pylaia-env``.
    """

    def __init__(
        self,
        model: str = "Teklia/pylaia-home-alcar",
        pylaia_env: str = _DEFAULT_PYLAIA_ENV,
    ):
        self.model = model
        self.pylaia_bin = os.path.join(pylaia_env, "bin", "pylaia-htr-decode-ctc")
        self.pylaia_create_bin = os.path.join(
            pylaia_env, "bin", "pylaia-htr-create-model"
        )
        model_slug = model.replace("/", "-").lower()
        self.model_cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", model_slug
        )

    def _check_env(self):
        if not os.path.exists(self.pylaia_bin):
            sys.exit(
                f"ERROR: pylaia-htr-decode-ctc not found at {self.pylaia_bin}\n"
                "Create the pylaia-env conda environment:\n"
                "  conda create -n pylaia-env python=3.10 -y\n"
                "  conda run -n pylaia-env pip install pylaia huggingface_hub "
                "'setuptools<72' 'torchmetrics==0.4.1'"
            )

    def _download_model(self) -> str:
        from huggingface_hub import snapshot_download
        logger.info("Downloading/verifying model %s", self.model)
        return snapshot_download(self.model)

    def _create_model_if_needed(self, snap_dir: str) -> str:
        model_file = os.path.join(self.model_cache_dir, "model")
        if os.path.exists(model_file):
            return self.model_cache_dir

        os.makedirs(self.model_cache_dir, exist_ok=True)
        syms_path = os.path.join(snap_dir, "syms.txt")
        logger.info("Creating model architecture file (one-time setup)")
        cmd = [
            self.pylaia_create_bin,
            syms_path,
            "--common.train_path", self.model_cache_dir,
            "--common.model_filename", "model",
            "--common.experiment_dirname", ".",
            "--fixed_input_height", str(_TARGET_HEIGHT),
            "--crnn.num_input_channels", "1",
            "--crnn.cnn_num_features", "[12,24,48,48]",
            "--crnn.cnn_kernel_size", "[[3,3],[3,3],[3,3],[3,3]]",
            "--crnn.cnn_poolsize", "[[2,1],[2,1],[2,1],[1,1]]",
            "--crnn.cnn_batchnorm", "[true,true,true,true]",
            "--crnn.rnn_layers", "3",
            "--crnn.rnn_units", "256",
            "--logging.level", "WARNING",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(
                f"ERROR: pylaia-htr-create-model failed (exit {result.returncode})\n"
                + result.stderr[-2000:]
            )
        return self.model_cache_dir

    @staticmethod
    def _extract_crop(bgr_image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale and resize to TARGET_HEIGHT."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        new_w = max(1, int(w * _TARGET_HEIGHT / h))
        return cv2.resize(gray, (new_w, _TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    def run(self, collection: Collection) -> Collection:
        self._check_env()
        snap_dir = self._download_model()
        model_cache_dir = self._create_model_if_needed(snap_dir)
        syms_path = os.path.join(snap_dir, "syms.txt")
        ckpt_path = os.path.join(snap_dir, "weights.ckpt")

        nodes = list(collection.active_leaves())
        if not nodes:
            logger.warning("No active leaf nodes found; nothing to recognise")
            return collection

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write one crop PNG per node, named by index for ordered retrieval
            crop_paths = []
            for i, node in enumerate(nodes):
                crop = self._extract_crop(node.image)
                path = os.path.join(tmpdir, f"line_{i:04d}.png")
                cv2.imwrite(path, crop)
                crop_paths.append(path)

            # Write image list for pylaia
            img_list_path = os.path.join(tmpdir, "img_list.txt")
            with open(img_list_path, "w") as f:
                for p in crop_paths:
                    f.write(p + "\n")

            cmd = [
                self.pylaia_bin,
                syms_path,
                img_list_path,
                "--common.train_path", model_cache_dir,
                "--common.experiment_dirname", ".",
                "--common.model_filename", "model",
                "--common.checkpoint", ckpt_path,
                "--data.color_mode", "L",
                "--decode.include_img_ids", "true",
                "--decode.join_string", "",
                "--logging.level", "WARNING",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(
                    "pylaia-htr-decode-ctc failed (exit %d): %s",
                    result.returncode,
                    result.stderr[-2000:],
                )
                # Return empty results rather than crashing the pipeline
                empty = Result.text_recognition_result({}, [""], [0.0])
                collection.update([empty] * len(nodes))
                return collection

            # Parse output: each line is "<path_without_ext> <text>"
            transcriptions: dict[str, str] = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                img_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                transcriptions[img_id] = text

        # Match transcriptions back to nodes by index (line_0000, line_0001, ...)
        results = []
        for i in range(len(nodes)):
            stem = os.path.join(tmpdir, f"line_{i:04d}")
            # pylaia uses the path stem (no extension) as the key
            text = transcriptions.get(stem, "")
            results.append(
                Result.text_recognition_result({}, [text], [0.0])
            )
            logger.debug("line_%04d → %r", i, text)

        collection.update(results)
        logger.info("Recognised %d lines", len(nodes))
        return collection

"""
Stage 3: Run PyLaia inference on line crops using Teklia/pylaia-home-alcar.

Downloads the model from HuggingFace on first run
(cached to ~/.cache/huggingface/).
Calls pylaia-htr-decode-ctc from the separate pylaia-env conda environment.

Outputs:
  outputs/pylaia_baseline/transcriptions/{stem}.txt
  outputs/pylaia_baseline/results.csv  (folio, line_id, transcription)
"""

import csv
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")

CROPS_DIR = os.path.join(_ROOT, "outputs", "pylaia_baseline", "crops")
TRANS_DIR = os.path.join(_ROOT, "outputs", "pylaia_baseline", "transcriptions")
RESULTS_CSV = os.path.join(_ROOT, "outputs", "pylaia_baseline", "results.csv")

MODEL_ID = "Teklia/pylaia-home-alcar"

# pylaia lives in a separate conda env to avoid torch version conflicts
_PYLAIA_ENV = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "pylaia-env")
PYLAIA_BIN = os.path.join(_PYLAIA_ENV, "bin", "pylaia-htr-decode-ctc")
PYLAIA_CREATE_BIN = os.path.join(_PYLAIA_ENV, "bin", "pylaia-htr-create-model")

# Persistent cache for the created model architecture file
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pylaia-home-alcar")


def download_model():
    from huggingface_hub import snapshot_download
    print(f"Downloading/verifying model {MODEL_ID} ...")
    path = snapshot_download(MODEL_ID)
    print(f"  Model at: {path}")
    return path


def create_model_if_needed(snap_dir):
    """Create the model architecture .pt file if it doesn't already exist.

    pylaia-htr-decode-ctc requires a serialized model object (created by
    pylaia-htr-create-model) in addition to the weights checkpoint.
    Architecture recovered from weights.ckpt state_dict shapes.
    """
    model_file = os.path.join(MODEL_CACHE_DIR, "model")
    if os.path.exists(model_file):
        return MODEL_CACHE_DIR

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    syms_path = os.path.join(snap_dir, "syms.txt")

    print("  Creating model architecture file (one-time setup) ...")
    cmd = [
        PYLAIA_CREATE_BIN,
        syms_path,
        "--common.train_path", MODEL_CACHE_DIR,
        "--common.model_filename", "model",
        "--common.experiment_dirname", ".",
        "--fixed_input_height", "128",
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
    print(f"  Model architecture saved to {model_file}")
    return MODEL_CACHE_DIR


def run_pylaia_on_folio(stem, snap_dir, model_cache_dir):
    crop_dir = os.path.join(CROPS_DIR, stem)
    crops = sorted(
        f for f in os.listdir(crop_dir) if f.endswith(".png")
    )
    if not crops:
        print(f"  {stem}: no crops found, skipping")
        return {}

    syms_path = os.path.join(snap_dir, "syms.txt")
    ckpt_path = os.path.join(snap_dir, "weights.ckpt")

    # Full paths in image list — avoids --img_dirs JSON list parsing quirk
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as fh:
        for c in crops:
            fh.write(os.path.join(crop_dir, c) + "\n")
        img_list_path = fh.name

    cmd = [
        PYLAIA_BIN,
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
    os.unlink(img_list_path)

    if result.returncode != 0:
        print(f"  {stem}: pylaia failed (exit {result.returncode})")
        print(result.stderr[-2000:])
        return {}

    # Parse output: each line is "<stem> <transcription>"
    transcriptions = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        img_id = parts[0]
        text = parts[1] if len(parts) > 1 else ""
        transcriptions[img_id] = text

    return transcriptions


def main():
    if not os.path.exists(PYLAIA_BIN):
        sys.exit(
            f"ERROR: pylaia-htr-decode-ctc not found at {PYLAIA_BIN}\n"
            "Create the pylaia-env conda environment:\n"
            "  conda create -n pylaia-env python=3.10 -y\n"
            "  conda run -n pylaia-env pip install pylaia huggingface_hub "
            "'setuptools<72' 'torchmetrics==0.4.1'"
        )

    os.makedirs(TRANS_DIR, exist_ok=True)
    snap_dir = download_model()
    model_cache_dir = create_model_if_needed(snap_dir)

    folio_dirs = sorted(
        d for d in os.listdir(CROPS_DIR)
        if os.path.isdir(os.path.join(CROPS_DIR, d))
    )
    if not folio_dirs:
        sys.exit("No crop directories found. Run 02_extract_crops.py first.")

    all_rows = []

    for stem in folio_dirs:
        trans_path = os.path.join(TRANS_DIR, f"{stem}.txt")
        if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
            print(f"  {stem} ... skipped (transcription exists)")
            # Still load for CSV
            with open(trans_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    img_id = parts[0]
                    text = parts[1] if len(parts) > 1 else ""
                    all_rows.append({"folio": stem, "line_id": img_id,
                                     "transcription": text})
            continue

        print(f"  {stem} ...", end=" ", flush=True)
        transcriptions = run_pylaia_on_folio(stem, snap_dir, model_cache_dir)

        # Write raw output
        with open(trans_path, "w") as f:
            for img_id, text in sorted(transcriptions.items()):
                f.write(f"{img_id} {text}\n")

        for img_id, text in sorted(transcriptions.items()):
            all_rows.append({"folio": stem, "line_id": img_id,
                             "transcription": text})

        print(f"{len(transcriptions)} lines transcribed")

    # Write/overwrite results.csv
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folio", "line_id", "transcription"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults → {os.path.relpath(RESULTS_CSV, _ROOT)}")
    print(f"Total lines: {len(all_rows)}")
    print("\nStage 3 done.")


if __name__ == "__main__":
    main()

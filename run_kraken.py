"""
Run Kraken's BLLA baseline segmenter against folio images and save
visualised outputs (baselines + bounding polygons drawn on originals)
to outputs/kraken_blla/, plus segmentation JSON to
outputs/kraken_blla/segmentation/{stem}_kraken.json for use by
mothra-evaluator.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image


def _pdf_to_bgr(path):
    import fitz
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, code)

FOLIO_DIR = os.path.join(os.path.dirname(__file__), "data", "folios")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "kraken_blla")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}

BASELINE_COLOUR = (0, 140, 255)   # orange (BGR)
POLYGON_COLOUR = (180, 60, 255)   # purple (BGR)
BASELINE_THICKNESS = 3
POLYGON_THICKNESS = 2
POLYGON_ALPHA = 0.12


def load_images(folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {folder}")
    return paths


def draw_kraken_result(cv_img, segmentation):
    """Draw baselines and boundary polygons from a Kraken Segmentation."""
    out = cv_img.copy()
    fill_layer = out.copy()

    for line in segmentation.lines:
        # Draw boundary polygon fill
        if line.boundary:
            pts = np.array(line.boundary, dtype=np.int32)
            cv2.fillPoly(fill_layer, [pts], POLYGON_COLOUR)

    # Blend polygon fill
    cv2.addWeighted(fill_layer, POLYGON_ALPHA, out, 1 - POLYGON_ALPHA, 0, out)

    for line in segmentation.lines:
        # Draw boundary polygon outline
        if line.boundary:
            pts = np.array(line.boundary, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True,
                          color=POLYGON_COLOUR, thickness=POLYGON_THICKNESS)

        # Draw baseline
        if line.baseline and len(line.baseline) >= 2:
            pts = np.array(line.baseline, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False,
                          color=BASELINE_COLOUR, thickness=BASELINE_THICKNESS)

    return out


def main():
    from kraken import blla

    parser = argparse.ArgumentParser()
    parser.add_argument("--folios", default=FOLIO_DIR)
    parser.add_argument("--output", default=os.path.dirname(OUT_DIR))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    out_dir = os.path.join(args.output, "kraken_blla")
    os.makedirs(out_dir, exist_ok=True)
    paths = load_images(args.folios)
    print(f"Found {len(paths)} image(s)")
    print(f"Output → {out_dir}\n")

    custom_model = None
    if args.model:
        from kraken.lib import vgsl
        custom_model = vgsl.TorchVGSLModel.load_model(args.model)
        if 'hyper_params' not in custom_model.user_metadata:
            custom_model.user_metadata['hyper_params'] = {}
        print(f"Model: {args.model}\n")

    model_name = os.path.basename(args.model) if args.model else "kraken_blla"

    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{stem}_kraken.jpg")
        json_path = os.path.join(seg_dir, f"{stem}_kraken.json")
        if os.path.exists(out_path) and os.path.exists(json_path):
            print(f"  {stem} ... skipped (output exists)")
            continue
        print(f"  {stem} ...", end=" ", flush=True)

        if os.path.splitext(path)[1].lower() == ".pdf":
            cv_img = _pdf_to_bgr(path)
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.open(path).convert("RGB")
            cv_img = cv2.imread(path)
            if cv_img is None:
                arr = np.array(pil_img)
                cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        segmentation = blla.segment(pil_img, model=custom_model, device="cpu")
        n_lines = len(segmentation.lines)

        if not os.path.exists(out_path):
            annotated = draw_kraken_result(cv_img, segmentation)
            cv2.imwrite(out_path, annotated)

        # Save segmentation JSON for mothra-evaluator
        h, w = cv_img.shape[:2]
        seg_data = {
            "folio": stem,
            "source": os.path.relpath(path, os.path.dirname(out_dir)),
            "image_width": w,
            "image_height": h,
            "model_name": model_name,
            "run_date": datetime.now(timezone.utc).isoformat(),
            "lines": [
                {
                    "id": i,
                    "baseline": [list(pt) for pt in line.baseline] if line.baseline else None,
                    "boundary": [list(pt) for pt in line.boundary] if line.boundary else None,
                }
                for i, line in enumerate(segmentation.lines)
            ],
        }
        with open(json_path, "w") as f:
            json.dump(seg_data, f)

        print(f"{n_lines} lines  →  {os.path.relpath(out_path)}")

    print("\nKraken BLLA done.")


if __name__ == "__main__":
    main()

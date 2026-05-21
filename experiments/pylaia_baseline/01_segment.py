"""
Stage 1: Run Kraken BLLA on the 4 selected folios and save raw segmentation
data (line coordinates) as JSON files.

Also writes visualisation JPGs to outputs/kraken_blla/ using the same format
as run_kraken.py, skipping any that already exist there.

Outputs:
  outputs/pylaia_baseline/segmentation/{stem}.json
  outputs/kraken_blla/{stem}_kraken.jpg   (skipped if already present)
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")

FOLIO_DIR = os.path.join(_ROOT, "data", "folios")
FOLIOS_LIST = os.path.join(_HERE, "folios.txt")
SEG_DIR = os.path.join(_ROOT, "outputs", "pylaia_baseline", "segmentation")
VIZ_DIR = os.path.join(_ROOT, "outputs", "kraken_blla")

BASELINE_COLOUR = (0, 140, 255)
POLYGON_COLOUR = (180, 60, 255)
BASELINE_THICKNESS = 3
POLYGON_THICKNESS = 2
POLYGON_ALPHA = 0.12


def _pdf_to_bgr(path):
    import fitz
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, code)


def load_folio(path):
    if os.path.splitext(path)[1].lower() == ".pdf":
        cv_img = _pdf_to_bgr(path)
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.open(path).convert("RGB")
        cv_img = cv2.imread(path)
        if cv_img is None:
            arr = np.array(pil_img)
            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return pil_img, cv_img


def draw_kraken_result(cv_img, segmentation):
    out = cv_img.copy()
    fill_layer = out.copy()
    for line in segmentation.lines:
        if line.boundary:
            pts = np.array(line.boundary, dtype=np.int32)
            cv2.fillPoly(fill_layer, [pts], POLYGON_COLOUR)
    cv2.addWeighted(fill_layer, POLYGON_ALPHA, out, 1 - POLYGON_ALPHA, 0, out)
    for line in segmentation.lines:
        if line.boundary:
            pts = np.array(line.boundary, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True,
                          color=POLYGON_COLOUR, thickness=POLYGON_THICKNESS)
        if line.baseline and len(line.baseline) >= 2:
            pts = np.array(line.baseline, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False,
                          color=BASELINE_COLOUR, thickness=BASELINE_THICKNESS)
    return out


def main():
    from kraken import blla

    os.makedirs(SEG_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    with open(FOLIOS_LIST) as f:
        filenames = [l.strip() for l in f if l.strip()]

    for filename in filenames:
        stem = os.path.splitext(filename)[0]
        path = os.path.join(FOLIO_DIR, filename)
        json_path = os.path.join(SEG_DIR, f"{stem}.json")
        viz_path = os.path.join(VIZ_DIR, f"{stem}_kraken.jpg")

        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found")

        if os.path.exists(json_path):
            print(f"  {stem} ... skipped (segmentation JSON exists)")
            continue

        print(f"  {stem} ...", end=" ", flush=True)

        pil_img, cv_img = load_folio(path)
        segmentation = blla.segment(pil_img, device="cpu")
        n = len(segmentation.lines)

        # Save raw segmentation JSON
        h, w = cv_img.shape[:2]
        data = {
            "folio": stem,
            "source": os.path.relpath(path, _ROOT),
            "image_width": w,
            "image_height": h,
            "lines": [
                {
                    "id": i,
                    "baseline": [list(pt) for pt in (line.baseline or [])],
                    "boundary": [list(pt) for pt in (line.boundary or [])],
                }
                for i, line in enumerate(segmentation.lines)
            ],
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save visualisation (skip if already written by run_kraken.py)
        if not os.path.exists(viz_path):
            annotated = draw_kraken_result(cv_img, segmentation)
            cv2.imwrite(viz_path, annotated)

        print(f"{n} lines  →  {os.path.relpath(json_path, _ROOT)}")

    print("\nStage 1 done.")


if __name__ == "__main__":
    main()

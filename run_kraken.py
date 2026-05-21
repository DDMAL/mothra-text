"""
Run Kraken's BLLA baseline segmenter against folio images and save
visualised outputs (baselines + bounding polygons drawn on originals)
to outputs/kraken_blla/.
"""

import argparse
import glob
import os
import sys

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
    args = parser.parse_args()

    out_dir = os.path.join(args.output, "kraken_blla")
    os.makedirs(out_dir, exist_ok=True)
    paths = load_images(args.folios)
    print(f"Found {len(paths)} image(s)")
    print(f"Output → {out_dir}\n")

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{stem}_kraken.jpg")
        if os.path.exists(out_path):
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

        segmentation = blla.segment(pil_img, device="cpu")
        n_lines = len(segmentation.lines)

        annotated = draw_kraken_result(cv_img, segmentation)

        cv2.imwrite(out_path, annotated)
        print(f"{n_lines} lines  →  {os.path.relpath(out_path)}")

    print("\nKraken BLLA done.")


if __name__ == "__main__":
    main()

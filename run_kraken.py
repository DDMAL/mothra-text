"""
Run Kraken's BLLA baseline segmenter against folio images and save
visualised outputs (baselines + bounding polygons drawn on originals)
to outputs/kraken_blla/.
"""

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

FOLIO_DIR = os.path.join(os.path.dirname(__file__), "data", "folios")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "kraken_blla")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

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

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = load_images(FOLIO_DIR)
    print(f"Found {len(paths)} image(s)")
    print(f"Output → {OUT_DIR}\n")

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"  {stem} ...", end=" ", flush=True)

        pil_img = Image.open(path).convert("RGB")
        segmentation = blla.segment(pil_img, device="cpu")

        n_lines = len(segmentation.lines)

        cv_img = cv2.imread(path)
        if cv_img is None:
            # Fallback for PIL-only formats
            arr = np.array(pil_img)
            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        annotated = draw_kraken_result(cv_img, segmentation)

        out_path = os.path.join(OUT_DIR, f"{stem}_kraken.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"{n_lines} lines  →  {os.path.relpath(out_path)}")

    print("\nKraken BLLA done.")


if __name__ == "__main__":
    main()

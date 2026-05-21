"""
Stage 2: Extract individual line crops from the segmentation JSON files.

For each line in each folio's segmentation JSON, crops the bounding box of the
boundary polygon from the original image, converts to grayscale, and resizes to
128px height (PyLaia's required input height), preserving aspect ratio.

Outputs:
  outputs/pylaia_baseline/crops/{stem}/line_{id:04d}.png
"""

import json
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")

FOLIO_DIR = os.path.join(_ROOT, "data", "folios")
SEG_DIR = os.path.join(_ROOT, "outputs", "pylaia_baseline", "segmentation")
CROPS_DIR = os.path.join(_ROOT, "outputs", "pylaia_baseline", "crops")

TARGET_HEIGHT = 128


def _pdf_to_bgr(path):
    import fitz
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, code)


def load_bgr(source_rel):
    path = os.path.join(_ROOT, source_rel)
    if os.path.splitext(path)[1].lower() == ".pdf":
        return _pdf_to_bgr(path)
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    return img


def extract_crop(img, boundary, img_h, img_w):
    if not boundary:
        return None
    pts = np.array(boundary, dtype=np.int32)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    # Clamp to image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w - 1, x1), min(img_h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    # Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Resize to TARGET_HEIGHT, preserve aspect ratio
    h, w = gray.shape
    new_w = max(1, int(w * TARGET_HEIGHT / h))
    resized = cv2.resize(gray, (new_w, TARGET_HEIGHT),
                         interpolation=cv2.INTER_AREA)
    return resized


def main():
    json_files = sorted(
        f for f in os.listdir(SEG_DIR) if f.endswith(".json")
    )
    if not json_files:
        print(f"No segmentation JSON files found in {SEG_DIR}")
        print("Run 01_segment.py first.")
        return

    for jf in json_files:
        with open(os.path.join(SEG_DIR, jf)) as f:
            data = json.load(f)

        stem = data["folio"]
        out_dir = os.path.join(CROPS_DIR, stem)
        os.makedirs(out_dir, exist_ok=True)

        img = load_bgr(data["source"])
        img_h, img_w = img.shape[:2]

        n_written = 0
        n_skipped = 0
        n_empty = 0

        for line in data["lines"]:
            crop_path = os.path.join(out_dir, f"line_{line['id']:04d}.png")
            if os.path.exists(crop_path):
                n_skipped += 1
                continue

            crop = extract_crop(img, line["boundary"], img_h, img_w)
            if crop is None:
                n_empty += 1
                continue

            cv2.imwrite(crop_path, crop)
            n_written += 1

        total = len(data["lines"])
        print(
            f"  {stem}: {n_written} written, "
            f"{n_skipped} skipped, {n_empty} empty  "
            f"({total} lines total)"
        )

    print("\nStage 2 done.")


if __name__ == "__main__":
    main()

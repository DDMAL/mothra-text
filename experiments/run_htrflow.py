"""
Run htrflow YOLO and RTMDet line segmentation models against folio images
and save visualised outputs (line polygons drawn on originals) and
segmentation JSON files to outputs/.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

FOLIO_DIR = os.path.join(os.path.dirname(__file__), "data", "folios")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}

# Colours (BGR)
YOLO_COLOUR = (0, 200, 50)    # green
RTMDET_COLOUR = (255, 80, 0)  # blue-orange


def _pdf_to_bgr(path):
    import fitz
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, code)


def load_images(folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {folder}")
    images = []
    for p in paths:
        if os.path.splitext(p)[1].lower() == ".pdf":
            img = _pdf_to_bgr(p)
        else:
            img = cv2.imread(p)
        if img is None:
            print(f"  Warning: could not read {p}, skipping")
            continue
        images.append((p, img))
    return images


def draw_segments(image, result, colour):
    """Draw line polygons from a htrflow Result onto a copy of image."""
    from htrflow.utils.draw import draw_masks, draw_polygons

    out = image.copy()

    polys = [p for p in result.polygons if p is not None]
    if polys:
        out = draw_polygons(out, polys, color=colour, thickness=2, alpha=0.15)
        return out

    # RTMDet returns masks rather than explicit polygons
    masks = [m for m in result.local_mask if m is not None]
    if masks:
        out = draw_masks(out, masks, color=colour, alpha=0.25)

    return out


def _poly_to_boundary(poly):
    """Convert an htrflow polygon to a plain [[x, y], ...] list."""
    arr = np.asarray(poly)
    if arr.dtype == object:
        # List of Point-like objects with .x / .y attributes
        return [[float(pt.x), float(pt.y)] for pt in poly]
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.reshape(-1, 2)
    return arr.tolist()


def _result_to_lines(result):
    """Convert an htrflow Result to the standard segmentation lines list."""
    polygons = result.polygons if result.polygons is not None else []
    masks = result.local_mask if result.local_mask is not None else []
    lines = []
    for i, _ in enumerate(result.segments):
        poly = polygons[i] if i < len(polygons) else None
        mask = masks[i] if i < len(masks) else None
        if poly is not None:
            boundary = _poly_to_boundary(poly)
        elif mask is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                boundary = c.reshape(-1, 2).tolist()
            else:
                boundary = None
        else:
            boundary = None
        lines.append({"id": i, "boundary": boundary, "baseline": None})
    return lines


def run_yolo(images, out_dir):
    from htrflow.models.ultralytics.yolo import YOLO

    print(f"\n{'='*60}")
    print("Running htrflow YOLO (yolov9-lines-1)")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(
        model="Riksarkivet/yolov9-lines-1", device="cpu"
    )

    for path, img in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{stem}_yolo.jpg")
        json_path = os.path.join(out_dir, f"{stem}_yolo.json")
        if os.path.exists(out_path) and os.path.exists(json_path):
            print(f"  {stem} ... skipped (output exists)")
            continue
        print(f"  {stem} ...", end=" ", flush=True)

        results = list(model.predict([img]))
        result = results[0]

        n = len(result.segments)
        annotated = draw_segments(img, result, YOLO_COLOUR)

        if not os.path.exists(out_path):
            cv2.imwrite(out_path, annotated)

        h, w = img.shape[:2]
        seg_data = {
            "folio": stem,
            "source": os.path.relpath(path, os.path.dirname(out_dir)),
            "image_width": w,
            "image_height": h,
            "model_name": "yolov9_lines",
            "run_date": datetime.now(timezone.utc).isoformat(),
            "lines": _result_to_lines(result),
        }
        with open(json_path, "w") as f:
            json.dump(seg_data, f)

        print(f"{n} lines  →  {os.path.relpath(out_path)}")

    print("YOLO done.")


def _prepare_rtmdet():
    """Two workarounds needed for RTMDet on torch 2.10 / Apple Silicon:
    1. Load MPS stub so mmcv _ext.so resolves MPSStream::commit (removed in
       torch 2.x).
    2. Patch torch.load to use weights_only=False so mmengine checkpoints
       (which contain HistoryBuffer globals) can be unpickled.
    """
    import ctypes
    stub = "/tmp/libmps_stub.dylib"
    if os.path.exists(stub):
        try:
            ctypes.CDLL(stub, ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    import torch
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load


def run_rtmdet(images, out_dir):
    _prepare_rtmdet()
    from htrflow.models.openmmlab.rtmdet import RTMDet

    print(f"\n{'='*60}")
    print("Running htrflow RTMDet (rtmdet_lines)")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    model = RTMDet(model="Riksarkivet/rtmdet_lines", device="cpu")

    for path, img in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{stem}_rtmdet.jpg")
        json_path = os.path.join(out_dir, f"{stem}_rtmdet.json")
        if os.path.exists(out_path) and os.path.exists(json_path):
            print(f"  {stem} ... skipped (output exists)")
            continue
        print(f"  {stem} ...", end=" ", flush=True)

        results = list(model.predict([img]))
        result = results[0]

        n = len(result.segments)
        annotated = draw_segments(img, result, RTMDET_COLOUR)

        if not os.path.exists(out_path):
            cv2.imwrite(out_path, annotated)

        h, w = img.shape[:2]
        seg_data = {
            "folio": stem,
            "source": os.path.relpath(path, os.path.dirname(out_dir)),
            "image_width": w,
            "image_height": h,
            "model_name": "rtmdet_lines",
            "run_date": datetime.now(timezone.utc).isoformat(),
            "lines": _result_to_lines(result),
        }
        with open(json_path, "w") as f:
            json.dump(seg_data, f)

        print(f"{n} lines  →  {os.path.relpath(out_path)}")

    print("RTMDet done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["yolo", "rtmdet", "both"], default="both"
    )
    parser.add_argument("--folios", default=FOLIO_DIR)
    parser.add_argument("--output", default=OUT_DIR)
    args = parser.parse_args()

    print(f"Loading images from {args.folios}")
    images = load_images(args.folios)
    names = [os.path.basename(p) for p, _ in images]
    print(f"Found {len(images)} image(s): {names}")

    if args.model in ("yolo", "both"):
        run_yolo(images, os.path.join(args.output, "htrflow_yolo"))

    if args.model in ("rtmdet", "both"):
        run_rtmdet(images, os.path.join(args.output, "htrflow_rtmdet"))


if __name__ == "__main__":
    main()

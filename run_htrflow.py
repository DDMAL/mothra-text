"""
Run htrflow YOLO and RTMDet line segmentation models against folio images
and save visualised outputs (line polygons drawn on originals) to outputs/.
"""

import argparse
import glob
import os
import sys

import cv2

FOLIO_DIR = os.path.join(os.path.dirname(__file__), "data", "folios")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Colours (BGR)
YOLO_COLOUR = (0, 200, 50)    # green
RTMDET_COLOUR = (255, 80, 0)  # blue-orange


def load_images(folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {folder}")
    images = []
    for p in paths:
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
        if os.path.exists(out_path):
            print(f"  {stem} ... skipped (output exists)")
            continue
        print(f"  {stem} ...", end=" ", flush=True)

        results = list(model.predict([img]))
        result = results[0]

        n = len(result.segments)
        annotated = draw_segments(img, result, YOLO_COLOUR)

        cv2.imwrite(out_path, annotated)
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
        if os.path.exists(out_path):
            print(f"  {stem} ... skipped (output exists)")
            continue
        print(f"  {stem} ...", end=" ", flush=True)

        results = list(model.predict([img]))
        result = results[0]

        n = len(result.segments)
        annotated = draw_segments(img, result, RTMDET_COLOUR)

        cv2.imwrite(out_path, annotated)
        print(f"{n} lines  →  {os.path.relpath(out_path)}")

    print("RTMDet done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["yolo", "rtmdet", "both"], default="both"
    )
    parser.add_argument("--folios", default=FOLIO_DIR)
    args = parser.parse_args()

    print(f"Loading images from {args.folios}")
    images = load_images(args.folios)
    names = [os.path.basename(p) for p, _ in images]
    print(f"Found {len(images)} image(s): {names}")

    if args.model in ("yolo", "both"):
        run_yolo(images, os.path.join(OUT_DIR, "htrflow_yolo"))

    if args.model in ("rtmdet", "both"):
        run_rtmdet(images, os.path.join(OUT_DIR, "htrflow_rtmdet"))


if __name__ == "__main__":
    main()

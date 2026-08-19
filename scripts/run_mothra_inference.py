#!/usr/bin/env python3
"""Run the DDMAL mothra YOLOv11 models on folio images to produce annotation JSONs.

Downloads text_music_detector_fulldata.pt and stave_detector_fulldata.pt from
DDMAL-lab/mothra-yolov11-checkpoints (requires HF token at ~/.cache/huggingface/token).

Output JSON schema matches the mothra Annotator export format used by the pipeline:
  { imageName, imageWidth, imageHeight, annotations: [{id, classId, bbox, confidence, timestamp}] }
  bbox format: [x_topleft, y_topleft, width, height] in absolute pixels
  classId: 1=text, 2=music, 3=staves

Usage
-----
python scripts/run_mothra_inference.py \\
    --images path/to/folio1.jpg path/to/folio2.jpg \\
    --out-dir ~/Downloads/DDMAL/mothra-text-layer/JSONs/Uncorrected/
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

HF_REPO = "DDMAL-lab/mothra-yolov11-checkpoints"
TEXT_MUSIC_MODEL = "text_music_detector_fulldata.pt"
STAVE_MODEL = "stave_detector_fulldata.pt"

# YOLO class index → mothra classId
# text_music_detector: 0=text, 1=music
# stave_detector:      0=staves
TEXT_MUSIC_CLASS_MAP = {0: 1, 1: 2}
STAVE_CLASS_MAP = {0: 3}


def _hf_token() -> str:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if not token_path.exists():
        raise FileNotFoundError(
            "HuggingFace token not found at ~/.cache/huggingface/token. "
            "Run `huggingface-cli login` first."
        )
    return token_path.read_text().strip()


def _download_model(filename: str, token: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, filename, token=token)


def _predict(model, image_path: str, class_map: dict, conf: float) -> list[dict]:
    """Run YOLO inference and return mothra-format annotation dicts."""
    results = model.predict(image_path, conf=conf, verbose=False)
    result = results[0]
    annotations = []
    ts = datetime.now(timezone.utc).isoformat()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotations
    for box in boxes:
        cls_idx = int(box.cls.item())
        class_id = class_map.get(cls_idx)
        if class_id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        annotations.append({
            "id": str(uuid.uuid4()),
            "classId": class_id,
            "bbox": [x1, y1, w, h],
            "confidence": float(box.conf.item()),
            "timestamp": ts,
        })
    return annotations


def run_inference(image_paths: list[str], out_dir: Path, conf: float = 0.25) -> None:
    from ultralytics import YOLO

    token = _hf_token()
    print("Downloading models from HuggingFace...")
    tm_path = _download_model(TEXT_MUSIC_MODEL, token)
    st_path = _download_model(STAVE_MODEL, token)
    print(f"  text_music_detector: {tm_path}")
    print(f"  stave_detector:      {st_path}")

    tm_model = YOLO(tm_path)
    st_model = YOLO(st_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_path = str(Path(image_path).expanduser().resolve())
        stem = Path(image_path).name
        out_path = out_dir / (Path(image_path).stem + ".json")

        if out_path.exists():
            print(f"  {stem} — skipped (output exists: {out_path})")
            continue

        print(f"  {stem} ...", end=" ", flush=True)

        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            w, h = img.size

        annotations = []
        annotations += _predict(tm_model, image_path, TEXT_MUSIC_CLASS_MAP, conf)
        annotations += _predict(st_model, image_path, STAVE_CLASS_MAP, conf)

        payload = {
            "imageName": Path(image_path).name,
            "imageWidth": w,
            "imageHeight": h,
            "annotations": annotations,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        counts = {1: 0, 2: 0, 3: 0}
        for a in annotations:
            counts[a["classId"]] = counts.get(a["classId"], 0) + 1
        print(
            f"{len(annotations)} annotations  (text={counts[1]}, "
            f"music={counts[2]}, staves={counts[3]})  → {out_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", required=True, metavar="PATH",
                        help="Folio image paths to process.")
    parser.add_argument("--out-dir", required=True, metavar="PATH",
                        help="Directory to write output JSONs.")
    parser.add_argument("--conf", type=float, default=0.25, metavar="FLOAT",
                        help="YOLO confidence threshold (default 0.25).")
    args = parser.parse_args()

    run_inference(args.images, Path(args.out_dir).expanduser(), conf=args.conf)
    print("Done.")


if __name__ == "__main__":
    main()

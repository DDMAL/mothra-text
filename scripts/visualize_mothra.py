"""Overlay mothra object detection annotations on folio images.

Usage
-----
python scripts/visualize_mothra.py JSON_PATH IMAGE_PATH OUTPUT_PATH

Output is a JPEG with classId-1 (text) in green, classId-2 (music) in blue,
classId-3 (staves) in red.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_COLORS = {
    1: (0, 200, 0),    # text: green
    2: (0, 100, 255),  # music: blue
    3: (255, 50, 50),  # staves: red
}
CLASS_LABELS = {1: "text", 2: "music", 3: "staves"}


def visualize(json_path: Path, image_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    counts: dict[int, int] = {}
    for ann in data["annotations"]:
        cid = ann.get("classId")
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        color = CLASS_COLORS.get(cid, (200, 200, 0))
        fill = color + (50,)
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=color + (255,), width=2)
        counts[cid] = counts.get(cid, 0) + 1

    image.save(output_path)
    print(f"Saved: {output_path}")
    for cid in sorted(counts):
        label = CLASS_LABELS.get(cid, f"class{cid}")
        print(f"  classId {cid} ({label}): {counts[cid]} annotations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay mothra annotations on a folio image."
    )
    parser.add_argument("json_path", type=Path, help="Mothra annotation JSON")
    parser.add_argument("image_path", type=Path, help="Folio image (JPEG/PNG)")
    parser.add_argument("output_path", type=Path, help="Destination image path")
    args = parser.parse_args()
    visualize(args.json_path, args.image_path, args.output_path)


if __name__ == "__main__":
    main()

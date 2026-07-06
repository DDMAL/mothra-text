"""Apply a mothra text-detection mask to a folio image before Kraken BLLA.

Keeps only classId-1 (text) regions; everything else is blacked out.
Vertical and horizontal padding merges adjacent word-level detections into
continuous strips that Kraken can recognise as full lines.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw


class MothraImageMask:
    """Mask a folio image to show only mothra-detected text regions.

    Args:
        mothra_json_path: Path to a mothra annotation JSON file.
        padding_px: Pixels added around each classId-1 bbox on all sides.
            Larger values merge adjacent word-level detections into line-width
            strips. Default 25.
    """

    def __init__(self, mothra_json_path: str | Path, padding_px: int = 25):
        data = json.loads(Path(mothra_json_path).read_text(encoding="utf-8"))
        self._bboxes = [
            ann["bbox"]
            for ann in data["annotations"]
            if ann.get("classId") == 1
        ]
        self.padding_px = padding_px

    def apply(self, pil_image: Image.Image) -> Image.Image:
        """Return a copy of pil_image with only text regions visible.

        Non-text areas are set to black. Each classId-1 bbox [x, y, w, h]
        is expanded by padding_px on all sides before compositing.
        """
        W, H = pil_image.size
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        pad = self.padding_px
        for bbox in self._bboxes:
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x0 = max(0, int(x) - pad)
            y0 = max(0, int(y) - pad)
            x1 = min(W, int(x + w) + pad)
            y1 = min(H, int(y + h) + pad)
            draw.rectangle([x0, y0, x1, y1], fill=255)

        result = Image.new("RGB", (W, H), (0, 0, 0))
        result.paste(pil_image, mask=mask)
        return result

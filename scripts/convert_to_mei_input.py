"""Convert mothra-text pipeline JSON output to the Text Alignment JSON format expected by MEI encoding."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def convert_to_mei_input(pipeline_data: dict, exclude_fallback: bool = False) -> dict:
    """
    Pure function: takes a parsed mothra-text pipeline JSON dict, returns a MEI
    Text Alignment JSON dict.

    MEI encoding expects:
      {
        "median_line_spacing": float,
        "syl_boxes": [{"syl": str, "ul": [int, int], "lr": [int, int]}, ...]
      }
    """
    lines = pipeline_data.get("lines", [])

    y_centers = [
        (line["bbox"][1] + line["bbox"][3]) / 2
        for line in lines
        if "bbox" in line
    ]
    y_centers.sort()
    diffs = np.diff(y_centers)
    median_line_spacing = float(np.quantile(diffs, 0.75)) if len(diffs) > 0 else 0.0

    syl_boxes = []
    for line in lines:
        for word in line.get("words", []):
            if exclude_fallback and word.get("source") == "fallback":
                continue
            for syllable in word.get("syllables", []):
                text = syllable.get("text", "").rstrip("-")
                if not text:
                    continue
                bbox = syllable["bbox"]
                syl_boxes.append({
                    "syl": text,
                    "ul": [int(bbox[0]), int(bbox[1])],
                    "lr": [int(bbox[2]), int(bbox[3])],
                })

    return {"median_line_spacing": median_line_spacing, "syl_boxes": syl_boxes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mothra-text pipeline JSON to MEI encoding Text Alignment JSON."
    )
    parser.add_argument("--input", type=Path, required=True, metavar="PATH",
                        help="Path to mothra-text --export-json output file")
    parser.add_argument("--output", type=Path, required=True, metavar="PATH",
                        help="Destination path for MEI Text Alignment JSON")
    parser.add_argument("--exclude-fallback", action="store_true",
                        help="Skip syllables from lines that received no Cantus ground truth text")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    pipeline_data = json.loads(input_path.read_text(encoding="utf-8"))
    result = convert_to_mei_input(pipeline_data, exclude_fallback=args.exclude_fallback)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    n_syls = len(result["syl_boxes"])
    logging.info(
        "Wrote %s (%d syllables, median_line_spacing=%.1f)",
        output_path,
        n_syls,
        result["median_line_spacing"],
    )


if __name__ == "__main__":
    main()

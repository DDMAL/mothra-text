"""Convert Mothra Annotator JSON exports to PAGE XML for Kraken BLLA training."""

import argparse
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = (
    f"{PAGE_NS} "
    f"{PAGE_NS}/pagecontent.xsd"
)
CLASS_TEXT_LINE = 1
KNOWN_CLASS_IDS = {1, 3}

ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", XSI_NS)


def convert_mothra_to_page(
    mothra_data: dict,
    annotations: list[dict] | None = None,
) -> str:
    """
    Pure function: takes a parsed Mothra JSON dict, returns a PAGE XML string.
    No file I/O, no filtering.

    This is the integration point for the future Mothra Annotator
    "Download PAGE XML" feature — call with no `annotations` override to
    pass all annotations through unchanged.

    `annotations` overrides mothra_data["annotations"] when provided; pass a
    pre-filtered list from the CLI script, or omit to include every annotation.
    """
    required = {"imageName", "imageWidth", "imageHeight", "annotations"}
    missing = required - mothra_data.keys()
    if missing:
        raise ValueError(f"Mothra JSON missing required keys: {missing}")

    image_name: str = mothra_data["imageName"]
    W = int(mothra_data["imageWidth"])
    H = int(mothra_data["imageHeight"])
    annots = annotations if annotations is not None else mothra_data["annotations"]

    sorted_annots = sorted(annots, key=lambda a: a["bbox"][1] + a["bbox"][3] / 2)

    if not sorted_annots:
        logging.warning("No annotations to convert for %s", image_name)

    root = ET.Element(
        f"{{{PAGE_NS}}}PcGts",
        {f"{{{XSI_NS}}}schemaLocation": SCHEMA_LOCATION},
    )

    meta = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(meta, f"{{{PAGE_NS}}}Creator").text = "mothra_to_page.py"
    ET.SubElement(meta, f"{{{PAGE_NS}}}Created").text = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    )

    page = ET.SubElement(
        root,
        f"{{{PAGE_NS}}}Page",
        {"imageFilename": image_name, "imageWidth": str(W), "imageHeight": str(H)},
    )

    region = ET.SubElement(
        page,
        f"{{{PAGE_NS}}}TextRegion",
        {"id": "region_0", "type": "paragraph"},
    )
    ET.SubElement(
        region,
        f"{{{PAGE_NS}}}Coords",
        {"points": f"0,0 {W},0 {W},{H} 0,{H}"},
    )

    for i, ann in enumerate(sorted_annots):
        bbox = ann["bbox"]
        x0 = round(bbox[0])
        y0 = round(bbox[1])
        x1 = round(bbox[0] + bbox[2])
        y1 = round(bbox[1] + bbox[3])

        line = ET.SubElement(
            region,
            f"{{{PAGE_NS}}}TextLine",
            {"id": f"line_{i}"},
        )
        ET.SubElement(
            line,
            f"{{{PAGE_NS}}}Coords",
            {"points": f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"},
        )
        ET.SubElement(
            line,
            f"{{{PAGE_NS}}}Baseline",
            {"points": f"{x0},{y1} {x1},{y1}"},
        )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def filter_annotations(
    annotations: list[dict],
    image_height: float,
    height_filter_pct: float,
) -> list[dict]:
    """Keep classId==1 annotations whose height is below the threshold."""
    threshold = image_height * height_filter_pct
    kept = []
    for ann in annotations:
        cid = ann.get("classId")
        if cid not in KNOWN_CLASS_IDS:
            logging.debug("Unknown classId %s on annotation %s", cid, ann.get("id"))
        if cid != CLASS_TEXT_LINE:
            continue
        h = ann["bbox"][3]
        if h > threshold:
            logging.warning(
                "Skipped %s: height %.0fpx exceeds %.1f%% of %.0fpx",
                ann.get("id"),
                h,
                height_filter_pct * 100,
                image_height,
            )
            continue
        kept.append(ann)
    return kept


def process_directory(
    input_dir: Path,
    output_dir: Path,
    height_filter_pct: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logging.warning("No .json files found in %s", input_dir)
        return

    ok = skipped = 0
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            filtered = filter_annotations(
                data["annotations"], data["imageHeight"], height_filter_pct
            )
            xml_str = convert_mothra_to_page(data, annotations=filtered)
            out_path = output_dir / json_path.with_suffix(".xml").name
            out_path.write_text(xml_str, encoding="utf-8")
            logging.info("Wrote %s (%d lines)", out_path.name, len(filtered))
            ok += 1
        except Exception as exc:
            logging.error("Failed %s: %s", json_path.name, exc)
            skipped += 1

    logging.info("Done: %d converted, %d failed", ok, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mothra Annotator JSON exports to PAGE XML."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of .json files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .xml files (default: <input_dir>/page_xml/)",
    )
    parser.add_argument(
        "--height-filter",
        type=float,
        default=0.15,
        metavar="FRACTION",
        help="Discard annotations taller than this fraction of image height (default: 0.15)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    if not (0 < args.height_filter < 1):
        parser.error("--height-filter must be between 0 and 1 (exclusive)")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input_dir is not a directory: {input_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir / "page_xml"
    )

    process_directory(input_dir, output_dir, args.height_filter)


if __name__ == "__main__":
    main()

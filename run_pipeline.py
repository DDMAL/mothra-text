#!/usr/bin/env python3
"""PoC pipeline: Kraken BLLA → PyLaia HTR → GT word segmentation.

Runs a single folio image through the mothra-text proof-of-concept
pipeline and prints a summary of the resulting word-level nodes.

Usage
-----
python run_pipeline.py \\
    --image  "path/to/folio.jpeg" \\
    --source-id 123610 \\
    --folio  "002r"

The Cantus source ID is the integer on the source detail page at
cantusdatabase.org. The folio string must match exactly how it appears
in the Cantus CSV (e.g. "002r", not "2r").

Adding pipeline steps
---------------------
To insert a new step (e.g. SyllableSegmentation) once it is implemented,
add it to the ``_build_remaining_steps`` list below the PyLaia step.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from htrflow.volume.volume import Collection

from steps.column_clustering import cluster_columns
from steps.gt_manifest import (
    build_page_manifest,
    fetch_cantus_csv,
    load_local_csv,
    make_manifest_lookup,
)
from steps.ground_truth_word_segmentation import GroundTruthWordSegmentation
from steps.kraken_segmentation import KrakenSegmentation
from steps.pylaia_recognition import PyLaiaRecognition

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    image_path: str,
    folio: str,
    source_id: int | None = None,
    csv_path: str | None = None,
    pylaia_model: str = "Teklia/pylaia-home-alcar",
    device: str = "cpu",
    line_offset: int = 0,
    column_variance_threshold: float = 0.5,
) -> tuple[Collection, dict[str, str]]:
    """Run the PoC pipeline on one folio image.

    Args:
        image_path:                 Path to the folio JPEG/PNG/TIFF.
        folio:                      Folio string exactly as it appears in the CSV.
        source_id:                  Cantus source ID (fetches CSV from cantusdatabase.org).
        csv_path:                   Path to a local Cantus-format CSV (alternative to
                                    source_id).
        pylaia_model:               HuggingFace model ID for PyLaia.
        device:                     Kraken inference device (``"cpu"`` or ``"cuda"``).
        line_offset:                Cantus lines to skip before aligning with detected
                                    nodes.  Use when the image is a crop starting partway
                                    through the folio.
        column_variance_threshold:  Minimum fraction of xmin variance explained by the
                                    two-cluster split for the page to be treated as
                                    two-column.  See cluster_columns().

    Returns:
        Tuple of (collection, manifest) where collection has word-level nodes
        and manifest is the node-label → Cantus-text mapping used for GT.
    """
    collection = Collection([image_path])

    # Stage 1: Kraken BLLA line segmentation
    logger.info("Stage 1: Kraken line segmentation")
    collection = KrakenSegmentation(device=device).run(collection)
    n_lines = sum(1 for _ in collection.active_leaves())
    logger.info("  %d line nodes", n_lines)

    # Stage 2: Column clustering — sort line nodes into reading order
    logger.info("Stage 2: Column clustering")
    page = next(iter(collection))
    sorted_labels, column_count = cluster_columns(
        line_nodes=list(page.children),
        page_width=page.image.shape[1],
        variance_threshold=column_variance_threshold,
    )
    logger.info("  %d column(s) detected", column_count)

    # Stage 3: PyLaia text recognition (subprocess into pylaia-env)
    logger.info("Stage 3: PyLaia text recognition (%s)", pylaia_model)
    collection = PyLaiaRecognition(model=pylaia_model).run(collection)

    # Build GT manifest using column-ordered node labels.
    if csv_path:
        logger.info("Stage 3: Loading local CSV from %s", csv_path)
        csv_rows = load_local_csv(csv_path)
    else:
        logger.info("Stage 3: Fetching Cantus CSV for source %d", source_id)
        csv_rows = fetch_cantus_csv(source_id)
    manifest = build_page_manifest(
        csv_rows, folio, sorted_labels, line_offset=line_offset
    )
    logger.info(
        "  Manifest: %d / %d node labels matched to Cantus text",
        len(manifest),
        len(sorted_labels),
    )
    gt_lookup = make_manifest_lookup(manifest)

    # Remaining steps — add new steps to this list as they are implemented:
    remaining_steps = [
        GroundTruthWordSegmentation(gt_lookup=gt_lookup),
        # SyllableSegmentation(),   # add when implemented
        # Export(...),              # add when implemented
    ]
    for step in remaining_steps:
        collection = step.run(collection)

    return collection, manifest


def export_json(
    collection: Collection,
    image_path: str,
    manifest: dict[str, str],
    out_path: str,
) -> None:
    """Write line polygons and word bboxes to a GUI-compatible JSON file.

    All coordinates are absolute image pixels.

    Args:
        collection: Collection after GroundTruthWordSegmentation.
        image_path: Original folio image path (used for folio name).
        manifest:   Node-label → Cantus-text mapping; used to tag words
                    as 'gt' (label in manifest) or 'fallback'.
        out_path:   Destination path for the JSON file.
    """
    page = next(iter(collection))
    lines = []
    for line_node in page.children:
        lbbox = line_node.bbox
        lpoly = [[pt.x, pt.y] for pt in line_node.polygon.points]
        words = []
        for word_node in line_node.children:
            wbbox = word_node.bbox
            words.append({
                "label": word_node.label,
                "text": word_node.text or "",
                "bbox": [wbbox.xmin, wbbox.ymin, wbbox.xmax, wbbox.ymax],
                "source": "gt" if line_node.label in manifest else "fallback",
            })
        lines.append({
            "label": line_node.label,
            "bbox": [lbbox.xmin, lbbox.ymin, lbbox.xmax, lbbox.ymax],
            "polygon": lpoly,
            "text": line_node.text or "",
            "words": words,
        })

    img = page.image
    payload = {
        "folio": Path(image_path).stem,
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "lines": lines,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Exported pipeline JSON to %s", out_path)


def _summarise(collection: Collection) -> None:
    """Print a compact summary of word-level nodes."""
    nodes = list(collection.active_leaves())
    print(f"\n{'Label':<45} {'Text'}")
    print("-" * 80)
    for node in nodes:
        print(f"{node.label:<45} {node.text!r}")
    print(f"\nTotal word nodes: {len(nodes)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mothra-text PoC pipeline on one folio image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="Path to the folio image.")
    parser.add_argument("--folio", required=True, metavar="STR",
                        help="Folio string as it appears in the CSV.")
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--source-id", type=int, metavar="INT",
                           help="Cantus source ID (fetches CSV from "
                                "cantusdatabase.org).")
    csv_group.add_argument("--csv", metavar="PATH",
                           help="Path to a local Cantus-format CSV file.")
    parser.add_argument("--pylaia-model", default="Teklia/pylaia-home-alcar",
                        metavar="MODEL_ID",
                        help="HuggingFace PyLaia model ID.")
    parser.add_argument("--device", default="cpu",
                        help="Kraken inference device (default: cpu).")
    parser.add_argument(
        "--export-json", metavar="PATH", default=None,
        help="If given, write pipeline output JSON for the GUI to PATH.",
    )
    parser.add_argument(
        "--line-offset", type=int, default=0, metavar="N",
        help="Skip the first N Cantus lines before aligning with detected "
             "nodes. Use when the image is a crop starting partway through "
             "the folio (default: 0).",
    )
    parser.add_argument(
        "--column-variance-threshold", type=float, default=0.5, metavar="FLOAT",
        help="Minimum fraction of x-start variance explained by a two-cluster "
             "split for the page to be treated as two-column. Higher values "
             "require tighter clusters (default: 0.5).",
    )
    args = parser.parse_args()

    image_path = str(Path(args.image).expanduser().resolve())
    if not Path(image_path).exists():
        parser.error(f"Image not found: {image_path}")

    collection, manifest = run(
        image_path=image_path,
        folio=args.folio,
        source_id=args.source_id,
        csv_path=args.csv,
        pylaia_model=args.pylaia_model,
        device=args.device,
        line_offset=args.line_offset,
        column_variance_threshold=args.column_variance_threshold,
    )
    _summarise(collection)

    if args.export_json:
        export_json(collection, image_path, manifest, args.export_json)


if __name__ == "__main__":
    main()

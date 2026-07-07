#!/usr/bin/env python3
"""Mothra-aware pipeline: masks the folio image to mothra classId-1 (text) regions
before passing it through the standard pipeline.

run_pipeline.py is not modified. All new code is in new files.

Usage
-----
python run_pipeline_mothra.py \\
    --image path/to/folio.jpg \\
    --folio 12v \\
    --source-id 599679 \\
    --mothra-json path/to/folio.json \\
    --export-json ~/Downloads/DDMAL/mothra_masked_12v.json

The --approach union mode is deprecated and should not be used.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from htrflow.volume.volume import Collection
from steps.column_clustering import cluster_columns, fuse_colinear_segments
from steps.ground_truth_word_segmentation import GroundTruthWordSegmentation
from steps.gt_manifest import fetch_cantus_csv, load_local_csv, make_manifest_lookup
from steps.kraken_recognition import KrakenRecognition
from steps.kraken_segmentation import KrakenSegmentation
from steps.mothra_mask import MothraImageMask
from steps.mothra_union import MothraUnionStep
from steps.nw_chant_allocator import (
    allocate_lines,
    build_flat_text_and_anchors,
)
from steps.syllable_segmentation import SyllableSegmentation

# Private helpers from run_pipeline — same project, no external deps changed.
from run_pipeline import (
    _build_pipeline_payload,
    _defuse_manifest,
    _find_tridis_model,
    _split_spanning_nodes_in_tree,
    run as _run_standard,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_RECOGNITION_MODEL = _find_tridis_model()


def run_union(
    image_path: str,
    mothra_json_path: str,
    folio: str | None = None,
    source_id: int | None = None,
    csv_path: str | None = None,
    segmentation_model: str | None = None,
    recognition_model: str | None = None,
    device: str = "cpu",
    line_offset: int = 0,
    column_bimodal_threshold: float = 0.5,
    column_count: int | None = None,
    debug_ocr: bool = False,
    ocr_only_mode: bool = False,
    iou_threshold: float = 0.3,
    min_width: int = 50,
) -> tuple[Collection, dict]:
    """Run the pipeline with mothra union injection (Approach B).

    Identical to run() in run_pipeline.py except MothraUnionStep runs
    immediately after KrakenSegmentation and before column clustering.
    """
    collection = Collection([image_path])
    folio = folio or Path(image_path).stem

    # Stage 1: Kraken BLLA segmentation
    logger.info("Stage 1: Kraken line segmentation")
    collection = KrakenSegmentation(device=device, model=segmentation_model).run(collection)
    n_kraken = sum(1 for _ in collection.active_leaves())
    logger.info("  %d line nodes from Kraken", n_kraken)

    # Stage 1b: Inject mothra lines missed by Kraken
    logger.info("Stage 1b: Mothra union injection")
    collection = MothraUnionStep(
        mothra_json_path,
        iou_threshold=iou_threshold,
        min_width=min_width,
    ).run(collection)
    n_after = sum(1 for _ in collection.active_leaves())
    logger.info("  %d total line nodes (%+d from mothra)", n_after, n_after - n_kraken)

    # Stage 2: Column clustering
    logger.info("Stage 2: Column clustering")
    page = next(iter(collection))
    sorted_labels, col_count, split_x = cluster_columns(
        line_nodes=list(page.children),
        page_width=page.image.shape[1],
        bimodal_threshold=column_bimodal_threshold,
        forced_column_count=column_count,
    )
    logger.info("  %d column(s) detected", col_count)

    if col_count >= 2 and split_x is not None:
        _split_spanning_nodes_in_tree(page, split_x)

    # Stage 3: Kraken HTR recognition
    logger.info("Stage 3: Kraken HTR recognition (model=%r)", recognition_model)
    collection = KrakenRecognition(model=recognition_model, device=device).run(collection)

    # Stage 4: Line fusion + NW chant allocation
    node_ocr = {node.label: (node.text or "") for node in page.children}
    fused_lines = fuse_colinear_segments(list(page.children), split_x)
    left_column_count = sum(1 for f in fused_lines if f.column == 1)
    logger.info(
        "  Fused %d segments → %d logical lines",
        sum(len(f.constituent_labels) for f in fused_lines),
        len(fused_lines),
    )
    fused_sorted_labels = [f.label for f in fused_lines]
    fused_ocr_texts = {
        f.label: " ".join(node_ocr[lbl] for lbl in f.constituent_labels)
        for f in fused_lines
    }

    if ocr_only_mode:
        logger.info("Stage 4: OCR-only — skipping Cantus alignment")
        manifest = {}
    else:
        if csv_path:
            csv_rows = load_local_csv(csv_path)
        else:
            logger.info("  Fetching Cantus CSV for source %d", source_id)
            csv_rows = fetch_cantus_csv(source_id)

        flat_text = build_flat_text_and_anchors(
            csv_rows, folio, line_offset=line_offset,
        )
        alloc_result = allocate_lines(
            flat_text=flat_text,
            sorted_labels=fused_sorted_labels,
            ocr_texts=fused_ocr_texts,
            column_count=col_count,
            left_column_count=left_column_count,
            snap_window=2,
            force_window=10,
            debug=debug_ocr,
            fused_lines=fused_lines,
            node_ocr=node_ocr,
        )
        manifest = _defuse_manifest(alloc_result.manifest, fused_lines)
        for lbl, text in alloc_result.constituent_overrides.items():
            manifest[lbl] = text
        for flag in alloc_result.flags:
            logger.warning("Validation flag [%s]: %s", flag.flag_type, flag.detail)
        logger.info(
            "  Manifest: %d / %d node labels assigned text",
            sum(1 for v in manifest.values() if v),
            len(manifest),
        )

    gt_lookup = make_manifest_lookup(manifest)
    for step in [GroundTruthWordSegmentation(gt_lookup=gt_lookup), SyllableSegmentation()]:
        collection = step.run(collection)

    return collection, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mothra-text pipeline with text-detection integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="Path to the folio image.")
    parser.add_argument("--folio", default=None, metavar="STR",
                        help="Folio string as it appears in the CSV.")
    parser.add_argument("--mothra-json", required=True, metavar="PATH",
                        help="Mothra annotation JSON for this folio.")
    parser.add_argument(
        "--approach", default="masked", choices=["masked", "union"],
        help=(
            "Defaults to 'masked'. "
            "The 'union' mode is deprecated and should not be used."
        ),
    )
    parser.add_argument("--export-json", metavar="PATH", default=None,
                        help="Write pipeline output JSON for the GUI to PATH.")

    csv_group = parser.add_mutually_exclusive_group()
    csv_group.add_argument("--source-id", type=int, metavar="INT",
                           help="Cantus source ID (fetches CSV from cantusdatabase.org).")
    csv_group.add_argument("--csv", metavar="PATH",
                           help="Path to a local Cantus-format CSV.")

    parser.add_argument("--segmentation-model", default=None, metavar="PATH",
                        help="Custom Kraken BLLA model (.mlmodel or .safetensors).")
    parser.add_argument(
        "--recognition-model",
        default=_DEFAULT_RECOGNITION_MODEL, metavar="PATH",
        help="Kraken HTR model. Defaults to Tridis if installed.",
    )
    parser.add_argument("--stub-mode", action="store_true", default=False,
                        help="Skip text recognition (empty text for all lines).")
    parser.add_argument("--device", default="cpu",
                        help="Kraken inference device (default: cpu).")
    parser.add_argument("--column-count", type=int, choices=[1, 2], default=None)
    parser.add_argument("--column-bimodal-threshold", type=float, default=0.5,
                        metavar="FLOAT")
    parser.add_argument("--debug-ocr", action="store_true", default=False)

    # Approach A options
    parser.add_argument(
        "--padding", type=int, default=15, metavar="PX",
        help="[masked] Pixels added around each text bbox before masking (default 15).",
    )

    # Approach B options
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3, metavar="FLOAT",
        help="[union] Max IoU with any Kraken line to inject a mothra line (default 0.3).",
    )
    parser.add_argument(
        "--min-width", type=int, default=50, metavar="PX",
        help="[union] Min pixel width of a merged mothra line to inject (default 50).",
    )

    args = parser.parse_args()

    ocr_only_mode = not args.csv and not args.source_id
    if not ocr_only_mode and not args.folio:
        parser.error("--folio is required when --csv or --source-id is given")

    image_path = str(Path(args.image).expanduser().resolve())
    if not Path(image_path).exists():
        parser.error(f"Image not found: {image_path}")

    mothra_json_path = str(Path(args.mothra_json).expanduser().resolve())
    recognition_model = None if args.stub_mode else args.recognition_model

    if args.approach == "union":
        import warnings
        warnings.warn(
            "--approach union is deprecated and will be removed in a future version. "
            "Use the default masked approach instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if args.approach == "masked":
        logger.info("Masking image with mothra text regions")
        img = Image.open(image_path).convert("RGB")
        masker = MothraImageMask(mothra_json_path, padding_px=args.padding)
        masked_img = masker.apply(img)
        logger.info(
            "  Applied mask: %d text bboxes, padding=%dpx",
            len(masker._bboxes), args.padding,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            masked_img.save(tmp.name)
            tmp_path = tmp.name

        try:
            collection, manifest = _run_standard(
                image_path=tmp_path,
                folio=args.folio,
                source_id=args.source_id,
                csv_path=args.csv,
                segmentation_model=args.segmentation_model,
                recognition_model=recognition_model,
                device=args.device,
                column_bimodal_threshold=args.column_bimodal_threshold,
                column_count=args.column_count,
                debug_ocr=args.debug_ocr,
                ocr_only_mode=ocr_only_mode,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    else:  # union (deprecated)
        logger.warning("Union approach is deprecated. Use masked approach instead.")
        collection, manifest = run_union(
            image_path=image_path,
            mothra_json_path=mothra_json_path,
            folio=args.folio,
            source_id=args.source_id,
            csv_path=args.csv,
            segmentation_model=args.segmentation_model,
            recognition_model=recognition_model,
            device=args.device,
            column_bimodal_threshold=args.column_bimodal_threshold,
            column_count=args.column_count,
            debug_ocr=args.debug_ocr,
            ocr_only_mode=ocr_only_mode,
            iou_threshold=args.iou_threshold,
            min_width=args.min_width,
        )

    if args.export_json:
        resolved_folio = args.folio or Path(image_path).stem
        _mode = "ocr_only" if ocr_only_mode else "cantus_aligned"
        payload = _build_pipeline_payload(
            collection, image_path, manifest,
            folio=resolved_folio, mode=_mode,
        )
        out_path = Path(args.export_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Exported pipeline JSON to %s", out_path)


if __name__ == "__main__":
    main()

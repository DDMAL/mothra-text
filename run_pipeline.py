#!/usr/bin/env python3
"""PoC pipeline: Kraken BLLA → Kraken HTR → GT word/syllable segmentation.

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
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from htrflow.volume.volume import Collection  # noqa: E402

from steps.column_clustering import (  # noqa: E402
    cluster_columns,
    fuse_colinear_segments,
)
from steps.gt_manifest import (  # noqa: E402
    fetch_cantus_csv,
    load_local_csv,
    make_manifest_lookup,
)
from steps.ground_truth_word_segmentation import (  # noqa: E402
    GroundTruthWordSegmentation,
)
from steps.syllable_segmentation import SyllableSegmentation  # noqa: E402
from steps.kraken_recognition import KrakenRecognition  # noqa: E402
from steps.kraken_segmentation import KrakenSegmentation  # noqa: E402
from steps.nw_chant_allocator import (  # noqa: E402
    FolioState,
    allocate_lines,
    build_flat_text_and_anchors,
    build_folio_state,
    read_folio_state,
    write_folio_state,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _defuse_manifest(fused_manifest, fused_lines):
    """Distribute words from each fused label back to constituent node labels.

    Words are split proportionally by constituent bbox width. The last
    constituent receives any remainder words to avoid rounding loss.
    """
    manifest = {}
    for fused in fused_lines:
        text = fused_manifest.get(fused.label, "")
        words = text.split() if text else []
        if len(fused.constituent_labels) == 1:
            manifest[fused.constituent_labels[0]] = text
        elif not words:
            for lbl in fused.constituent_labels:
                manifest[lbl] = ""
        else:
            total_w = sum(fused.constituent_widths) or 1
            idx = 0
            pairs = zip(fused.constituent_labels, fused.constituent_widths)
            last = len(fused.constituent_labels) - 1
            for i, (lbl, w) in enumerate(pairs):
                if i == last:
                    manifest[lbl] = " ".join(words[idx:])
                else:
                    count = max(0, round(len(words) * w / total_w))
                    manifest[lbl] = " ".join(words[idx:idx + count])
                    idx += count
    return manifest


def run(
    image_path: str,
    folio: str,
    source_id: int | None = None,
    csv_path: str | None = None,
    recognition_model: str | None = None,
    device: str = "cpu",
    line_offset: int = 0,
    column_variance_threshold: float = 0.5,
    prev_folio_state: "FolioState | None" = None,
    folio_state_out: str | None = None,
    debug_ocr: bool = False,
) -> tuple[Collection, dict[str, str]]:
    """Run the PoC pipeline on one folio image.

    Args:
        image_path: Path to the folio JPEG/PNG/TIFF.
        folio: Folio string exactly as it appears in the CSV.
        source_id: Cantus source ID (fetches from cantusdatabase.org).
        csv_path: Path to a local Cantus-format CSV (alternative to
            source_id).
        recognition_model: HuggingFace model ID or local path to a
            Kraken ``.mlmodel`` file.  Pass ``None`` (default) to run
            in stub mode (empty text, pipeline still completes).
        device: Kraken inference device (``"cpu"`` or ``"cuda"``).
        line_offset: Cantus lines to skip before aligning with detected
            nodes.  Use when the image is a crop starting partway
            through the folio.
        column_variance_threshold: Minimum fraction of xmin variance
            explained by the two-cluster split for the page to be
            treated as two-column.  See cluster_columns().
        prev_folio_state: FolioState from the previous folio run.
            When provided, its remaining_words (post-77 continuation)
            are prepended to the flat text before alignment.
        folio_state_out: If given, write the folio state JSON to this
            path after allocation (for use as prev_folio_state on the
            next folio).
        debug_ocr: When True, print per-fused-line OCR text and NW
            alignment detail to stdout for diagnosis.

    Returns:
        Tuple of (collection, manifest) where collection has word-level
        nodes and manifest is the node-label → Cantus-text mapping.
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
    sorted_labels, column_count, split_x = cluster_columns(
        line_nodes=list(page.children),
        page_width=page.image.shape[1],
        variance_threshold=column_variance_threshold,
    )
    logger.info("  %d column(s) detected", column_count)

    # Stage 3: Kraken HTR text recognition
    logger.info(
        "Stage 3: Kraken HTR recognition (model=%r)", recognition_model
    )
    collection = KrakenRecognition(
        model=recognition_model, device=device
    ).run(collection)

    # Stage 4: NW chant allocation
    if csv_path:
        logger.info(
            "Stage 4: NW allocation — loading local CSV from %s", csv_path
        )
        csv_rows = load_local_csv(csv_path)
    else:
        logger.info(
            "Stage 4: NW allocation — fetching Cantus CSV for source %d",
            source_id,
        )
        csv_rows = fetch_cantus_csv(source_id)
    flat_text = build_flat_text_and_anchors(
        csv_rows, folio,
        line_offset=line_offset,
        prev_folio_state=prev_folio_state,
    )
    node_ocr = {node.label: (node.text or "") for node in page.children}
    fused_lines = fuse_colinear_segments(list(page.children), split_x)
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

    # --- DEBUG OCR: print per-fused-line OCR transcripts ---
    if debug_ocr:
        print("\n=== DEBUG OCR: fused line transcripts ===")
        for f in fused_lines:
            print(f"  [{f.label}] {fused_ocr_texts[f.label]!r}")
        print()
    # --------------------------------------------------------

    alloc_result = allocate_lines(
        flat_text=flat_text,
        sorted_labels=fused_sorted_labels,
        ocr_texts=fused_ocr_texts,
        column_count=column_count,
        snap_window=2,
        force_window=10,  # force within_chant_7 up to ±10 words mid-chant
        debug=debug_ocr,
    )

    # --- DEBUG OCR: print per-line NW alignment detail ---
    if debug_ocr and alloc_result.debug_lines:
        print("\n=== DEBUG NW ALIGNMENT ===")
        for entry in alloc_result.debug_lines:
            print(f"\n-- {entry['label']} --")
            print(f"  OCR:      {entry['ocr']!r}")
            print(f"  Assigned: {entry['assigned']!r}")
            ptr_s = entry['pointer_start']
            ptr_e = entry['pointer_end']
            print(
                f"  Words:    [{ptr_s}..{ptr_e})"
                f" consumed={entry['consumed']}"
            )
            if entry['best_norm'] is not None:
                k_pre = entry['best_k_pre_snap']
                norm = entry['best_norm']
                print(
                    f"  NW score: k={k_pre} (pre-snap),"
                    f" norm={norm:.4f}"
                )
            if entry['anchor_word'] is not None:
                nw_end = entry['pointer_start'] + entry['best_k_pre_snap']
                diff = abs(nw_end - entry['anchor_word'])
                print(
                    f"  Anchor:   word {entry['anchor_word']}"
                    f" ({entry['anchor_type']})"
                    f", diff={diff}, snapped={entry['snapped']}"
                    f", forced={entry['forced']}"
                )
            if entry['alignment']:
                print("  Alignment:")
                for aln_line in entry['alignment'].splitlines():
                    print(f"    {aln_line}")
        print("=== END DEBUG ===\n")
    # ------------------------------------------------------

    manifest = _defuse_manifest(alloc_result.manifest, fused_lines)
    for flag in alloc_result.flags:
        logger.warning(
            "Validation flag [%s]: %s", flag.flag_type, flag.detail
        )
    logger.info(
        "  Manifest: %d / %d node labels assigned text",
        sum(1 for v in manifest.values() if v),
        len(manifest),
    )
    if folio_state_out:
        state = build_folio_state(flat_text, alloc_result, source_id, folio)
        write_folio_state(state, folio_state_out)
        logger.info(
            "Folio state written to %s (fully_consumed=%s)",
            folio_state_out,
            state.fully_consumed,
        )
    gt_lookup = make_manifest_lookup(manifest)

    # Remaining steps — add new steps to this list as they are implemented:
    remaining_steps = [
        GroundTruthWordSegmentation(gt_lookup=gt_lookup),
        SyllableSegmentation(),
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
        collection: Collection after SyllableSegmentation.
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
            syllables = []
            for syl_node in word_node.children:
                sbbox = syl_node.bbox
                syllables.append({
                    "label": syl_node.label,
                    "text": syl_node.text or "",
                    "bbox": [sbbox.xmin, sbbox.ymin, sbbox.xmax, sbbox.ymax],
                })
            words.append({
                "label": word_node.label,
                "text": word_node.text or "",
                "bbox": [wbbox.xmin, wbbox.ymin, wbbox.xmax, wbbox.ymax],
                "source": "gt" if line_node.label in manifest else "fallback",
                "syllables": syllables,
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
    parser.add_argument("--recognition-model", default=None,
                        metavar="MODEL_ID_OR_PATH",
                        help="HuggingFace model ID or local path to a Kraken "
                             ".mlmodel file. Omit to run in stub mode "
                             "(empty text, pipeline still completes).")
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
        "--column-variance-threshold",
        type=float, default=0.5, metavar="FLOAT",
        help="Minimum fraction of x-start variance explained by a "
             "two-cluster split for the page to be treated as two-column. "
             "Higher values require tighter clusters (default: 0.5).",
    )
    parser.add_argument(
        "--prev-folio-state", metavar="PATH", default=None,
        help="JSON sidecar from the previous folio run. Provides post-77 "
             "continuation words for chants that span two folios.",
    )
    parser.add_argument(
        "--folio-state-out", metavar="PATH", default=None,
        help="Write folio state JSON to PATH after allocation. Pass this "
             "as --prev-folio-state on the next folio run.",
    )
    parser.add_argument(
        "--debug-ocr", action="store_true", default=False,
        help="Print per-fused-line OCR text and NW alignment detail to "
             "stdout for diagnosing misalignment.",
    )
    args = parser.parse_args()

    image_path = str(Path(args.image).expanduser().resolve())
    if not Path(image_path).exists():
        parser.error(f"Image not found: {image_path}")

    prev_state = (
        read_folio_state(args.prev_folio_state)
        if args.prev_folio_state
        else None
    )

    collection, manifest = run(
        image_path=image_path,
        folio=args.folio,
        source_id=args.source_id,
        csv_path=args.csv,
        recognition_model=args.recognition_model,
        device=args.device,
        line_offset=args.line_offset,
        column_variance_threshold=args.column_variance_threshold,
        prev_folio_state=prev_state,
        folio_state_out=args.folio_state_out,
        debug_ocr=args.debug_ocr,
    )
    _summarise(collection)

    if args.export_json:
        export_json(collection, image_path, manifest, args.export_json)


if __name__ == "__main__":
    main()

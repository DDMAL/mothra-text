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
import statistics
import sys
import tempfile
from pathlib import Path


def _find_tridis_model() -> str | None:
    """Return the local path to the Tridis model, or None if not installed.

    Searches the htrmopo cache directory (platform-appropriate via
    platformdirs) for the Tridis .mlmodel file. Works on any machine
    where the model has been installed with htrmopo regardless of the
    UUID-named subdirectory.
    """
    try:
        from platformdirs import user_data_dir
        htrmopo_dir = Path(user_data_dir("htrmopo"))
        pat = "*/Tridis_Medieval_EarlyModern.mlmodel"
        matches = list(htrmopo_dir.glob(pat))
        return str(matches[0]) if matches else None
    except Exception:
        return None


_DEFAULT_RECOGNITION_MODEL = _find_tridis_model()

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
    make_output_stem,
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

from htrflow.utils.geometry import Bbox as _HtBbox
from htrflow.volume.volume import ImageNode as _HtImageNode


class _HtrflowSplitNode(_HtImageNode):
    """Synthetic htrflow leaf node for one half of a spanning bbox.

    Inserted into page.children BEFORE KrakenRecognition so that OCR,
    word segmentation, and export all operate on the correct half-image
    with the correct half-bbox.  Does not wrap a Segment object.
    """

    def __init__(self, parent, label: str,
                 xmin: int, ymin: int, xmax: int, ymax: int,
                 image_crop):
        super().__init__(parent=parent, label=label)
        self._bbox_obj = _HtBbox(xmin, ymin, xmax, ymax)
        self._image = image_crop   # pre-fill cache; _load_image never called

    @property
    def bbox(self) -> _HtBbox:
        return self._bbox_obj

    def _load_image(self):
        return self._image         # fallback; normally pre-filled in __init__

    def asdict(self) -> dict:
        from dataclasses import asdict as _dc_asdict
        return super().asdict() | {
            "segmentation_label": "region",
            "segmentation_confidence": 1.0,
            "bbox": _dc_asdict(self._bbox_obj),
            "polygon": str(self.polygon),
        }


def _split_spanning_nodes_in_tree(
    page,
    split_x: float,
    min_span_fraction: float = 0.1,
) -> int:
    """Replace spanning-bbox nodes in page.children with half-crop pairs.

    Detects nodes where xmin < split_x < xmax (with >= min_span_fraction
    overhang on each side), crops the page image at split_x, and inserts
    two _HtrflowSplitNode objects in place of each spanning node.

    Calls page.relabel() after any replacements so that all labels are
    consistent before KrakenRecognition reads them.

    Returns the number of nodes replaced.
    """
    page_img = page.image
    new_children = []
    split_count = 0

    for node in list(page.children):
        xmin, xmax = node.bbox.xmin, node.bbox.xmax
        ymin, ymax = node.bbox.ymin, node.bbox.ymax
        width = xmax - xmin or 1
        sx = int(split_x)
        left_frac = (sx - xmin) / width
        right_frac = (xmax - sx) / width

        if (xmin < split_x < xmax
                and left_frac >= min_span_fraction
                and right_frac >= min_span_fraction):
            left_crop = page_img[ymin:ymax, xmin:sx]
            right_crop = page_img[ymin:ymax, sx:xmax]
            new_children.append(_HtrflowSplitNode(
                page, "region", xmin, ymin, sx, ymax, left_crop))
            new_children.append(_HtrflowSplitNode(
                page, "region", sx, ymin, xmax, ymax, right_crop))
            split_count += 1
            logger.debug(
                "Split spanning node %r at x=%d (%.0f|%.0f px)",
                node.label, sx,
                sx - xmin, xmax - sx,
            )
        else:
            new_children.append(node)

    if split_count:
        page.children = new_children
        page.relabel()
        logger.info(
            "  Split %d spanning bbox(es) at column boundary x=%d",
            split_count, int(split_x),
        )

    return split_count


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


def _music_overlap_ratio(node_bbox, music_box: list[float]) -> float:
    lx0, ly0 = node_bbox.xmin, node_bbox.ymin
    lx1, ly1 = node_bbox.xmax, node_bbox.ymax
    mx0, my0, mx1, my1 = music_box
    ix0, iy0 = max(lx0, mx0), max(ly0, my0)
    ix1, iy1 = min(lx1, mx1), min(ly1, my1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    line_area = (lx1 - lx0) * (ly1 - ly0)
    return intersection / line_area if line_area > 0 else 0.0


def run(
    image_path: str,
    folio: str | None = None,
    source_id: int | None = None,
    csv_path: str | None = None,
    segmentation_model: str | None = None,
    recognition_model: str | None = None,
    device: str = "cpu",
    column_bimodal_threshold: float = 0.5,
    prev_folio_state: "FolioState | None" = None,
    folio_state_out: str | None = None,
    debug_ocr: bool = False,
    column_count: int | None = None,
    ocr_only_mode: bool = False,
    mothra_json_path: str | None = None,
    padding: int = 15,
    music_boxes: list[list[float]] | None = None,
    music_overlap_threshold: float = 0.30,
) -> tuple[Collection, dict[str, str]]:
    """Run the PoC pipeline on one folio image.

    Args:
        image_path: Path to the folio JPEG/PNG/TIFF.
        folio: Folio string exactly as it appears in the CSV.  Required
            when ``csv_path`` or ``source_id`` is given.  In OCR-only
            mode (neither given) defaults to the image filename stem.
        source_id: Cantus source ID (fetches from cantusdatabase.org).
            Omit together with ``csv_path`` to run in OCR-only mode.
        csv_path: Path to a local Cantus-format CSV (alternative to
            source_id).  Omit together with ``source_id`` to run in
            OCR-only mode.
        recognition_model: HuggingFace model ID or local path to a
            Kraken ``.mlmodel`` file.  Pass ``None`` to run in explicit
            stub mode (all lines receive empty text, pipeline still
            completes).  The CLI exits with an error if no model is
            available and ``--stub-mode`` was not given.
        device: Kraken inference device (``"cpu"`` or ``"cuda"``).
        column_bimodal_threshold: Maximum ratio of the coverage-profile
            valley to the smaller column peak for the valley to be
            treated as a genuine inter-column gutter.  See cluster_columns().
        prev_folio_state: FolioState from the previous folio run.
            When provided, its remaining_words (post-77 continuation)
            are prepended to the flat text before alignment.  Ignored
            in OCR-only mode.
        folio_state_out: If given, write the folio state JSON to this
            path after allocation (for use as prev_folio_state on the
            next folio).  Ignored in OCR-only mode.
        debug_ocr: When True, print per-fused-line OCR text and NW
            alignment detail to stdout for diagnosis.  In OCR-only mode
            prints a startup banner and flags any ignored Cantus args.
        column_count: When 1 or 2, declares the folio column count and
            skips bimodal auto-detection.  Pass the appropriate
            column-specific model via ``segmentation_model``.  ``None``
            (default) preserves existing auto-detection behaviour.
        ocr_only_mode: When True, skip Cantus data loading and NW
            allocation entirely.  All lines fall back to OCR text for
            word and syllable segmentation.  Set automatically by
            ``main()`` when neither ``--csv`` nor ``--source-id`` is
            given.
        mothra_json_path: Path to a mothra annotation JSON. When
            provided, blacks out non-text image regions before line
            segmentation. Pass ``None`` (default) to skip masking —
            behavior is unchanged for CLI and ``run_chain.py`` callers.
        padding: Pixels added around each text bbox when masking
            (default 15). Only used when ``mothra_json_path`` is given.
        music_boxes: List of music-region bounding boxes
            ``[x0, y0, x1, y1]`` in absolute pixels. When provided, any
            line node whose bbox overlaps a music box by more than
            ``music_overlap_threshold`` is dropped **before** Stage 4
            (NW allocation), so it cannot consume a GT slot. Pass
            ``None`` (default) to skip this filter — behaviour is
            unchanged for CLI and ``run_chain.py`` callers.
        music_overlap_threshold: Minimum overlap ratio (intersection /
            line area) for a line to be considered overlapping a music
            box (default 0.30).

    Returns:
        Tuple of (collection, manifest) where collection has word-level
        nodes and manifest is the node-label → Cantus-text mapping
        (empty dict in OCR-only mode).
    """
    _original_image_path = image_path
    _mothra_tmp: str | None = None
    if mothra_json_path is not None:
        from PIL import Image as _PILImage
        from steps.mothra_mask import MothraImageMask as _MothraImageMask
        _img = _PILImage.open(image_path).convert("RGB")
        _masker = _MothraImageMask(mothra_json_path, padding_px=padding)
        _masked = _masker.apply(_img)
        logger.info(
            "Applied mothra mask: %d text bboxes, padding=%dpx",
            len(_masker._bboxes), padding,
        )
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as _tmp:
            _masked.save(_tmp.name)
            _mothra_tmp = _tmp.name
        image_path = _mothra_tmp
    try:
        collection = Collection([image_path])
        folio = folio or Path(_original_image_path).stem

        # Model selection: when --column-count is given the caller should
        # pass the appropriate fine-tuned model via --segmentation-model.
        # In future this block can insert hardwired per-column defaults
        # when segmentation_model is None.
        effective_model = segmentation_model

        # Stage 1: Kraken BLLA line segmentation
        logger.info("Stage 1: Kraken line segmentation")
        collection = KrakenSegmentation(
            device=device, model=effective_model
        ).run(collection)
        n_lines = sum(1 for _ in collection.active_leaves())
        logger.info("  %d line nodes", n_lines)

        # Stage 2: Column clustering — sort line nodes into reading order
        logger.info("Stage 2: Column clustering")
        page = next(iter(collection))
        sorted_labels, column_count, split_x = cluster_columns(
            line_nodes=list(page.children),
            page_width=page.image.shape[1],
            bimodal_threshold=column_bimodal_threshold,
            forced_column_count=column_count,
        )
        logger.info("  %d column(s) detected", column_count)

        # Span-fix: split any bboxes crossing the column divide before OCR
        if column_count >= 2 and split_x is not None:
            _split_spanning_nodes_in_tree(page, split_x)

        # Stage 3: Kraken HTR text recognition
        logger.info(
            "Stage 3: Kraken HTR recognition (model=%r)", recognition_model
        )
        collection = KrakenRecognition(
            model=recognition_model,
            device=device,
            allow_stub=(recognition_model is None),
        ).run(collection)

        # Pre-Stage-4: drop lines overlapping music regions before NW allocation
        if music_boxes:
            kept, dropped_nodes = [], []
            for node in page.children:
                if any(_music_overlap_ratio(node.bbox, mb) > music_overlap_threshold
                       for mb in music_boxes):
                    dropped_nodes.append(node)
                    logger.info(
                        "folio %s: pre-NW drop — line bbox [%d,%d,%d,%d]"
                        " overlaps music region",
                        folio or "?",
                        node.bbox.xmin, node.bbox.ymin,
                        node.bbox.xmax, node.bbox.ymax,
                    )
                else:
                    kept.append(node)
            page.children = kept
            collection._music_filter_dropped = [
                {"bbox": [n.bbox.xmin, n.bbox.ymin, n.bbox.xmax, n.bbox.ymax],
                 "text": n.text or ""}
                for n in dropped_nodes
            ]
        else:
            collection._music_filter_dropped = []

        # Stage 4: Line fusion + chant allocation (or OCR-only)
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

        # --- DEBUG OCR: print per-fused-line OCR transcripts ---
        if debug_ocr:
            print("\n=== DEBUG OCR: fused line transcripts ===")
            for f in fused_lines:
                print(f"  [{f.label}] {fused_ocr_texts[f.label]!r}")
            print()
        # --------------------------------------------------------

        if ocr_only_mode:
            logger.info("Stage 4: OCR-only — skipping Cantus alignment")
            if debug_ocr:
                print(
                    "[OCR-only mode] No CSV or source ID"
                    " — Cantus alignment skipped."
                )
                print(f"[OCR-only mode] Folio label: {folio}")
                _ignored_flags = [
                    (prev_folio_state is not None, "--prev-folio-state"),
                    (folio_state_out is not None, "--folio-state-out"),
                ]
                for cond, name in _ignored_flags:
                    if cond:
                        print(
                            f"[OCR-only mode] Ignoring {name}"
                            " (not applicable without Cantus data)"
                        )
                print()
            manifest = {}
        else:
            if csv_path:
                logger.info(
                    "Stage 4: NW allocation — loading local CSV from %s",
                    csv_path,
                )
                csv_rows = load_local_csv(csv_path)
            else:
                logger.info(
                    "Stage 4: NW allocation — fetching Cantus CSV"
                    " for source %d",
                    source_id,
                )
                csv_rows = fetch_cantus_csv(source_id)
            flat_text = build_flat_text_and_anchors(
                csv_rows, folio,
                prev_folio_state=prev_folio_state,
            )

            alloc_result = allocate_lines(
                flat_text=flat_text,
                sorted_labels=fused_sorted_labels,
                ocr_texts=fused_ocr_texts,
                column_count=column_count,
                left_column_count=left_column_count,
                snap_window=2,
                force_window=10,
                debug=debug_ocr,
                fused_lines=fused_lines,
                node_ocr=node_ocr,
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
                        nw_end = (
                            entry['pointer_start'] + entry['best_k_pre_snap']
                        )
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
            # --------------------------------------------------

            manifest = _defuse_manifest(alloc_result.manifest, fused_lines)
            for lbl, text in alloc_result.constituent_overrides.items():
                manifest[lbl] = text
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
                state = build_folio_state(
                    flat_text, alloc_result, source_id, folio
                )
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
    finally:
        if _mothra_tmp:
            Path(_mothra_tmp).unlink(missing_ok=True)


def _build_pipeline_payload(
    collection: Collection,
    image_path: str,
    manifest: dict[str, str],
    folio: str | None = None,
    mode: str = "cantus_aligned",
) -> dict:
    """Build the GUI-compatible JSON payload dict from a pipeline run."""
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
                "source": (
                    "gt" if manifest.get(line_node.label) else "fallback"
                ),
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
    return {
        "folio": folio or Path(image_path).stem,
        "mode": mode,
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "lines": lines,
    }


def export_json(
    collection: Collection,
    image_path: str,
    manifest: dict[str, str],
    out_path: str,
) -> None:
    """Write line polygons and word bboxes to a GUI-compatible JSON file."""
    payload = _build_pipeline_payload(collection, image_path, manifest)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Exported pipeline JSON to %s", out_path)


def _write_mei_json(payload: dict, out_path: str) -> None:
    """Convert a pipeline payload dict to MEI Text Alignment JSON and
    write it.
    """
    lines = payload.get("lines", [])
    y_centers = sorted(
        (line["bbox"][1] + line["bbox"][3]) / 2
        for line in lines
        if "bbox" in line
    )
    diffs = [
        y_centers[i + 1] - y_centers[i]
        for i in range(len(y_centers) - 1)
    ]
    med = statistics.quantiles(diffs, n=4)[2] if len(diffs) >= 2 else 0.0

    syl_boxes = []
    for line in lines:
        for word in line.get("words", []):
            for syl in word.get("syllables", []):
                text = (syl.get("text") or "").rstrip("-")
                if text:
                    bb = syl["bbox"]
                    syl_boxes.append({
                        "syl": text,
                        "ul": [int(bb[0]), int(bb[1])],
                        "lr": [int(bb[2]), int(bb[3])],
                    })

    result = {"median_line_spacing": med, "syl_boxes": syl_boxes}
    Path(out_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).expanduser().write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Exported MEI input JSON to %s (%d syllables)",
        out_path, len(syl_boxes),
    )


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
    parser.add_argument(
        "--folio", default=None, metavar="STR",
        help="Folio string as it appears in the CSV. Required when "
             "--csv or --source-id is given. In OCR-only mode (neither "
             "given) defaults to the image filename stem.",
    )
    csv_group = parser.add_mutually_exclusive_group(required=False)
    csv_group.add_argument(
        "--source-id", type=int, metavar="INT",
        help="Cantus source ID (fetches CSV from cantusdatabase.org). "
             "Omit together with --csv to run in OCR-only mode.",
    )
    csv_group.add_argument(
        "--csv", metavar="PATH",
        help="Path to a local Cantus-format CSV file. "
             "Omit together with --source-id to run in OCR-only mode.",
    )
    parser.add_argument("--segmentation-model", default=None,
                        metavar="PATH",
                        help="Local path to a custom Kraken BLLA segmentation "
                             "model (.mlmodel or .safetensors). Omit to use "
                             "Kraken's built-in default BLLA model.")
    parser.add_argument(
        "--column-count", type=int, choices=[1, 2], default=None,
        metavar="{1,2}",
        help="Declare the folio column count. When given, skips bimodal "
             "column auto-detection. Pass the appropriate column-specific "
             "fine-tuned model via --segmentation-model. Omit to use "
             "existing auto-detection behaviour.",
    )
    parser.add_argument(
        "--recognition-model",
        default=_DEFAULT_RECOGNITION_MODEL,
        metavar="PATH",
        help=(
            "Local path to a Kraken .mlmodel recognition model. "
            "Defaults to the Tridis model if installed via htrmopo "
            "(install: python -m htrmopo get 10.5281/zenodo.10788591). "
            "Use --stub-mode to skip recognition entirely."
        ),
    )
    parser.add_argument(
        "--stub-mode", action="store_true", default=False,
        help=(
            "Skip text recognition. All lines get empty text; "
            "the pipeline still runs and produces word/syllable "
            "geometry from ground truth. Takes precedence over "
            "--recognition-model."
        ),
    )
    parser.add_argument("--device", default="cpu",
                        help="Kraken inference device (default: cpu).")
    parser.add_argument(
        "--export-json", metavar="PATH", default=None,
        help="If given, write pipeline output JSON for the GUI to PATH.",
    )
    parser.add_argument(
        "--mei-json", metavar="PATH", default=None,
        help=(
            "If given, write MEI encoding Text Alignment JSON to PATH. "
            "Produces the same output as running "
            "scripts/convert_to_mei_input.py on the --export-json file."
        ),
    )
    parser.add_argument(
        "--column-bimodal-threshold",
        type=float, default=0.5, metavar="FLOAT",
        help="Maximum ratio of the coverage-profile valley to the smaller "
             "column peak for the valley to be treated as a genuine gutter. "
             "Lower values require a deeper valley (default: 0.5).",
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
    parser.add_argument(
        "--mothra-json", default=None, metavar="PATH",
        help="Path to a mothra annotation JSON for this folio. When provided, "
             "blacks out non-text image regions before line segmentation. "
             "Omit to skip masking (pipeline behaviour unchanged).",
    )
    parser.add_argument(
        "--padding", type=int, default=15, metavar="PX",
        help="Pixels added around each text bbox when masking (default 15). "
             "Only used when --mothra-json is given.",
    )
    parser.add_argument(
        "--skip-masking", action="store_true", default=False,
        help=(
            "Skip text-region masking even if --mothra-json is provided. "
            "Masking is also skipped automatically when --mothra-json is absent."
        ),
    )
    parser.add_argument(
        "--output-dir", metavar="PATH", default=None,
        help="Directory for auto-named MEI JSON output. Requires --source-id or "
             "--csv and --folio. Output is named {RISM-code}_{shelfmark}_{folio}.json "
             "(e.g. CH-E_611_001r.json). Overridden by --mei-json if both are given.",
    )
    args = parser.parse_args()

    ocr_only_mode = not args.csv and not args.source_id
    if not ocr_only_mode and not args.folio:
        parser.error(
            "--folio is required when --csv or --source-id is given"
        )

    image_path = str(Path(args.image).expanduser().resolve())
    if not Path(image_path).exists():
        parser.error(f"Image not found: {image_path}")

    mothra_json_for_run: str | None = None
    if args.mothra_json and not args.skip_masking:
        mothra_json_for_run = str(
            Path(args.mothra_json).expanduser().resolve()
        )
    elif args.skip_masking:
        logger.info("--skip-masking set; text-region masking disabled.")
    else:
        logger.info(
            "No --mothra-json provided; running without text-region masking."
        )

    if ocr_only_mode:
        logger.info(
            "OCR-only mode: no --csv or --source-id given; "
            "Cantus alignment will be skipped"
        )
        _cantus_flags = [
            (args.prev_folio_state, "--prev-folio-state"),
            (args.folio_state_out, "--folio-state-out"),
        ]
        for val, name in _cantus_flags:
            if val:
                logger.warning(
                    "%s is ignored in OCR-only mode", name
                )

    output_stem: str | None = None
    out_dir: Path | None = None
    if args.output_dir:
        if ocr_only_mode:
            parser.error(
                "--output-dir requires --source-id or --csv "
                "(not available in OCR-only mode)"
            )
        if not args.folio:
            parser.error("--output-dir requires --folio")
        early_rows = (
            fetch_cantus_csv(args.source_id)
            if args.source_id
            else load_local_csv(args.csv)
        )
        output_stem = make_output_stem(early_rows, args.folio)
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output stem: %s", output_stem)

    if args.stub_mode:
        recognition_model = None
    else:
        recognition_model = args.recognition_model
        if recognition_model is None:
            print(
                "Error: no recognition model found and --stub-mode was not given.\n"
                "Install the Tridis model: "
                "python -m htrmopo get 10.5281/zenodo.10788591\n"
                "Or pass a model via:       --recognition-model PATH\n"
                "Or skip recognition with:  --stub-mode",
                file=sys.stderr,
            )
            sys.exit(1)

    prev_state = (
        read_folio_state(args.prev_folio_state)
        if args.prev_folio_state
        else None
    )

    original_image_path = str(Path(args.image).expanduser().resolve())
    collection, manifest = run(
        image_path=image_path,
        folio=args.folio,
        source_id=args.source_id,
        csv_path=args.csv,
        segmentation_model=args.segmentation_model,
        recognition_model=recognition_model,
        device=args.device,
        column_bimodal_threshold=args.column_bimodal_threshold,
        prev_folio_state=prev_state,
        folio_state_out=args.folio_state_out,
        debug_ocr=args.debug_ocr,
        column_count=args.column_count,
        ocr_only_mode=ocr_only_mode,
        mothra_json_path=mothra_json_for_run,
        padding=args.padding,
    )
    _summarise(collection)

    effective_mei_json = args.mei_json or (
        str(out_dir / f"{output_stem}.json") if output_stem else None
    )
    effective_export_json = args.export_json

    if effective_export_json or effective_mei_json:
        resolved_folio = (
            output_stem or args.folio or Path(original_image_path).stem
        )
        _mode = "ocr_only" if ocr_only_mode else "cantus_aligned"
        payload = _build_pipeline_payload(
            collection, original_image_path, manifest,
            folio=resolved_folio, mode=_mode,
        )

        if effective_export_json:
            with open(effective_export_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(
                "Exported pipeline JSON to %s", effective_export_json
            )

        if effective_mei_json:
            _write_mei_json(payload, effective_mei_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Automated multi-folio chaining wrapper for the mothra-text pipeline.

Runs a sequence of consecutive folio images through run_pipeline.py in order,
automatically passing FolioState (post-77 continuation words) between runs so
you don't have to manage --prev-folio-state / --folio-state-out sidecar files
manually.

Usage
-----
python run_chain.py \\
    --images 006r.jpg 007v.jpg 008r.jpg \\
    --folios 006r 007v 008r \\
    --source-id 123672 \\
    --export-json ~/Downloads/006r.json ~/Downloads/007v.json ~/Downloads/008r.json

Add --folio-states-dir /tmp/states/ to keep intermediate FolioState JSON files
for post-run inspection. Add --debug-ocr for per-line OCR and NW alignment detail.
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path


def _find_tridis_model() -> "str | None":
    """Return the local path to the Tridis model, or None if not installed."""
    try:
        from platformdirs import user_data_dir
        htrmopo_dir = Path(user_data_dir("htrmopo"))
        matches = list(htrmopo_dir.glob("*/Tridis_Medieval_EarlyModern.mlmodel"))
        return str(matches[0]) if matches else None
    except Exception:
        return None


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    n = len(args.images)
    if len(args.folios) != n:
        parser.error(
            f"--images ({n}) and --folios ({len(args.folios)}) must have the same length"
        )
    if args.export_json is not None and len(args.export_json) != n:
        parser.error(
            f"--export-json ({len(args.export_json)}) must match --images ({n})"
        )
    if args.mei_json is not None and len(args.mei_json) != n:
        parser.error(
            f"--mei-json ({len(args.mei_json)}) must match --images ({n})"
        )
    if n < 2:
        parser.error(
            "At least 2 folios are required for chaining. "
            "Use run_pipeline.py directly for a single folio."
        )
    for p in args.images:
        if not Path(p).expanduser().exists():
            parser.error(f"Image not found: {p}")
    if args.csv and not Path(args.csv).expanduser().exists():
        parser.error(f"CSV not found: {args.csv}")


def _run_one(
    idx: int,
    total: int,
    image_path: str,
    folio: str,
    prev_state: "object | None",
    export_json_path: "str | None",
    mei_json_path: "str | None",
    folio_label: "str",
    folio_states_dir: "str | None",
    recognition_model: "str | None",
    args: argparse.Namespace,
    run: "callable",
    export_json: "callable",
    read_folio_state: "callable",
    build_pipeline_payload: "callable",
    write_mei_json: "callable",
) -> "object":
    logger = logging.getLogger(__name__)
    logger.info("Folio %d/%d: %s", idx + 1, total, folio)

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        collection, manifest = run(
            image_path=image_path,
            folio=folio,
            source_id=args.source_id,
            csv_path=args.csv,
            segmentation_model=args.segmentation_model,
            recognition_model=recognition_model,
            device=args.device,
            column_bimodal_threshold=args.column_bimodal_threshold,
            prev_folio_state=prev_state,
            folio_state_out=tmp_path,
            debug_ocr=args.debug_ocr,
            column_count=args.column_count,
        )
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    if export_json_path:
        Path(export_json_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        export_json(collection, image_path, manifest, str(Path(export_json_path).expanduser()))

    if mei_json_path:
        payload = build_pipeline_payload(
            collection, image_path, manifest,
            folio=folio_label, mode="cantus_aligned",
        )
        write_mei_json(payload, mei_json_path)

    next_state = read_folio_state(tmp_path)
    logger.info(
        "  FolioState: remaining_words=%d, fully_consumed=%s",
        len(next_state.remaining_words),
        next_state.fully_consumed,
    )

    if folio_states_dir:
        dest = Path(folio_states_dir) / f"state_{folio}.json"
        shutil.copy2(tmp_path, dest)
        logger.info("  FolioState saved to %s", dest)

    Path(tmp_path).unlink(missing_ok=True)
    return next_state


def main() -> None:
    tridis = _find_tridis_model()

    parser = argparse.ArgumentParser(
        description="Run the mothra-text pipeline across a chain of consecutive folios, "
                    "automatically passing FolioState between runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--images", nargs="+", required=True, metavar="PATH",
                        help="Ordered list of folio image paths.")
    parser.add_argument("--folios", nargs="+", required=True, metavar="STR",
                        help="Ordered list of folio identifiers as they appear in the CSV.")
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--source-id", type=int, metavar="INT",
                           help="Cantus source ID (fetches CSV from cantusdatabase.org).")
    csv_group.add_argument("--csv", metavar="PATH",
                           help="Path to a local Cantus-format CSV file.")
    parser.add_argument("--export-json", nargs="+", default=None, metavar="PATH",
                        help="One output JSON path per folio (same order as --images). "
                             "Parent directories are created automatically.")
    parser.add_argument("--folio-states-dir", default=None, metavar="PATH",
                        help="Directory to save intermediate FolioState JSON files after "
                             "each run (named state_{folio}.json). Useful for debugging "
                             "the chain. Omit to discard intermediate state files.")
    parser.add_argument("--segmentation-model", default=None, metavar="PATH",
                        help="Local path to a custom Kraken BLLA segmentation model "
                             "(.mlmodel or .safetensors). Omit to use Kraken's built-in "
                             "default BLLA model.")
    parser.add_argument(
        "--column-count", type=int, choices=[1, 2], default=None, metavar="{1,2}",
        help="Declare the folio column count. Skips bimodal column auto-detection.",
    )
    parser.add_argument("--recognition-model", default=tridis, metavar="PATH",
                        help="Local path to a Kraken .mlmodel recognition model. "
                             "Defaults to the Tridis model if installed via htrmopo.")
    parser.add_argument("--stub-mode", action="store_true", default=False,
                        help="Skip text recognition on all folios.")
    parser.add_argument("--device", default="cpu",
                        help="Kraken inference device (default: cpu).")
    parser.add_argument(
        "--column-bimodal-threshold", type=float, default=0.5, metavar="FLOAT",
        help="Coverage-profile valley/peak ratio for gutter detection (default: 0.5).",
    )
    parser.add_argument("--debug-ocr", action="store_true", default=False,
                        help="Print per-line OCR transcripts and NW alignment detail "
                             "for every folio.")
    parser.add_argument(
        "--output-dir", metavar="PATH", default=None,
        help="Directory for auto-named MEI JSON outputs. Each folio is written as "
             "{RISM-code}_{shelfmark}_{folio}.json (e.g. CH-E_611_001r.json). "
             "Overridden per-folio by --mei-json if both are given.",
    )
    parser.add_argument(
        "--mei-json", nargs="+", default=None, metavar="PATH",
        help="One MEI JSON output path per folio (same order as --images). "
             "Takes precedence over --output-dir when both are given.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_pipeline import run, export_json, _build_pipeline_payload, _write_mei_json  # noqa: E402
    from steps.nw_chant_allocator import read_folio_state  # noqa: E402
    from steps.gt_manifest import fetch_cantus_csv, load_local_csv, make_output_stem  # noqa: E402

    _validate_args(parser, args)

    if args.stub_mode:
        recognition_model = None
    else:
        recognition_model = args.recognition_model
        if recognition_model is None:
            print(
                "Warning: Tridis recognition model not found. Running in stub mode.\n"
                "Install with: python -m htrmopo get 10.5281/zenodo.7899855",
                file=sys.stderr,
            )

    if args.folio_states_dir:
        Path(args.folio_states_dir).expanduser().mkdir(parents=True, exist_ok=True)

    images = [str(Path(p).expanduser().resolve()) for p in args.images]
    n = len(images)
    export_paths = args.export_json or [None] * n

    # Build per-folio MEI JSON paths: explicit --mei-json takes precedence,
    # otherwise auto-name from --output-dir using the regularized stem.
    if args.mei_json:
        mei_paths = args.mei_json
        folio_labels = list(args.folios)
    elif args.output_dir:
        csv_rows = (
            fetch_cantus_csv(args.source_id)
            if args.source_id
            else load_local_csv(args.csv)
        )
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = [make_output_stem(csv_rows, f) for f in args.folios]
        mei_paths = [str(out_dir / f"{stem}.json") for stem in stems]
        folio_labels = stems
    else:
        mei_paths = [None] * n
        folio_labels = list(args.folios)

    logger = logging.getLogger(__name__)
    completed = 0
    prev_state = None

    for i, (image_path, folio, out_json, mei_path, folio_label) in enumerate(
        zip(images, args.folios, export_paths, mei_paths, folio_labels)
    ):
        try:
            prev_state = _run_one(
                idx=i,
                total=n,
                image_path=image_path,
                folio=folio,
                prev_state=prev_state,
                export_json_path=out_json,
                mei_json_path=mei_path,
                folio_label=folio_label,
                folio_states_dir=args.folio_states_dir,
                recognition_model=recognition_model,
                args=args,
                run=run,
                export_json=export_json,
                read_folio_state=read_folio_state,
                build_pipeline_payload=_build_pipeline_payload,
                write_mei_json=_write_mei_json,
            )
            completed += 1
        except Exception as exc:
            logger.error(
                "Chain aborted at folio %s (%d/%d): %s", folio, i + 1, n, exc
            )
            logger.error("Completed %d/%d folios before failure.", completed, n)
            sys.exit(1)

    logger.info("Done: %d/%d folios completed successfully.", completed, n)


if __name__ == "__main__":
    main()

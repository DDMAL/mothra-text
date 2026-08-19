"""Compare pipeline JSON outputs across baseline and mothra approaches.

Reads one or more pipeline export JSONs and prints a comparison table
showing line counts and word-source statistics.

Usage
-----
python scripts/compare_runs.py \\
    --label baseline ~/Downloads/DDMAL/baseline_1v.json \\
    --label masked   ~/Downloads/DDMAL/mothra_masked_1v.json \\
    --output ~/Downloads/DDMAL/mothra_comparison_report_2026-07-03.txt
"""

import argparse
import json
from pathlib import Path


def analyse(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = data.get("lines", [])
    n_lines = len(lines)
    gt_words = fallback_words = 0
    for line in lines:
        for word in line.get("words", []):
            if word.get("source") == "gt":
                gt_words += 1
            else:
                fallback_words += 1
    return {
        "folio": data.get("folio", path.stem),
        "lines": n_lines,
        "gt_words": gt_words,
        "fallback_words": fallback_words,
        "total_words": gt_words + fallback_words,
    }


def format_table(rows: list[dict], labels: list[str]) -> str:
    headers = ["Label", "Folio", "Lines", "GT words", "Fallback", "Total words"]
    col_w = [max(len(h), 10) for h in headers]
    for label, r in zip(labels, rows):
        col_w[0] = max(col_w[0], len(label))
        col_w[1] = max(col_w[1], len(r["folio"]))

    headers = ["Label", "Folio", "Lines", "GT words", "Fallback", "Total words"]
    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    lines_out = [header_line, sep]
    for label, r in zip(labels, rows):
        row = [
            label.ljust(col_w[0]),
            r["folio"].ljust(col_w[1]),
            str(r["lines"]).ljust(col_w[2]),
            str(r["gt_words"]).ljust(col_w[3]),
            str(r["fallback_words"]).ljust(col_w[4]),
            str(r["total_words"]).ljust(col_w[5]),
        ]
        lines_out.append("  ".join(row))

    return "\n".join(lines_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mothra pipeline run JSONs.",
    )
    parser.add_argument(
        "--label", nargs=2, action="append", metavar=("LABEL", "PATH"),
        required=True,
        help="Add a run: --label <label> <path>. Repeat for each run.",
    )
    parser.add_argument(
        "--output", metavar="PATH", default=None,
        help="Save the report to this file (in addition to printing).",
    )
    args = parser.parse_args()

    labels = []
    rows = []
    for label, path_str in args.label:
        p = Path(path_str).expanduser()
        if not p.exists():
            print(f"Warning: file not found: {p}")
            continue
        labels.append(label)
        rows.append(analyse(p))

    if not rows:
        print("No valid files to compare.")
        return

    table = format_table(rows, labels)
    print(table)

    if args.output:
        out = Path(args.output).expanduser()
        if out.exists():
            print(f"Error: output file already exists: {out}")
            print("Choose a new filename to avoid overwriting.")
            return
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()

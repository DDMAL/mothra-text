"""Debug column detection on a folder of folio images.

Runs BLLA segmentation followed by the current cluster_columns algorithm on
every image in a given folder, producing per-image annotated visualisations
and a printed/saved summary table.

Output is written to a ``debug/`` subdirectory inside the input folder so
all artefacts stay co-located with the images they describe.

Usage::

    python scripts/debug_column_detection.py \\
        --folder path/to/images \\
        [--expected 1|2] \\
        [--model path/to/custom_model]
"""

import argparse
import os
import sys

# Extend the path before importing project modules so this script can be run
# from any working directory without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from htrflow.volume.volume import Collection  # noqa: E402
from steps.column_clustering import cluster_columns  # noqa: E402
from steps.kraken_segmentation import KrakenSegmentation  # noqa: E402

matplotlib.use("Agg")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

LEFT_COLOR = "#e74c3c"   # red  — left / single column
RIGHT_COLOR = "#3498db"  # blue — right column


def _image_paths(folder):
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {folder}")
    return paths


def _coverage(line_nodes, page_width):
    cov = np.zeros(page_width, dtype=int)
    for nd in line_nodes:
        x0 = max(0, nd.bbox.xmin)
        x1 = min(page_width, nd.bbox.xmax)
        if x1 > x0:
            cov[x0:x1] += 1
    return cov


def _process(image_path, seg_step, expected, folder):
    out_dir = os.path.join(folder, "debug")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    print(f"  {stem}", flush=True)

    collection = Collection([image_path])
    collection = seg_step.run(collection)
    page = next(iter(collection))
    line_nodes = list(page.children)
    W = page.image.shape[1]

    if not line_nodes:
        print("    WARNING: no lines detected")
        return {
            "image": stem,
            "n_lines": 0,
            "pred": "n/a",
            "split_x": "n/a",
            "med_width": "n/a",
            "xmin_var": "n/a",
            "gutter_ratio": "n/a",
            "correct": "n/a",
        }

    sorted_labels, column_count, split_x = cluster_columns(line_nodes, W)

    widths = [nd.bbox.xmax - nd.bbox.xmin for nd in line_nodes]
    med_width_frac = float(np.median(widths)) / W
    xmin_var = float(np.var([nd.bbox.xmin for nd in line_nodes]))
    cov = _coverage(line_nodes, W)
    max_cov = int(cov.max()) or 1
    band_start = int(0.20 * W)
    band_end = int(0.80 * W)
    band_min = int(cov[band_start:band_end].min())
    gutter_ratio = band_min / max_cov

    correct = None
    if expected is not None:
        correct = column_count == expected

    # ---- visualisation ----
    img_rgb = cv2.cvtColor(page.image, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_rgb)
    status = ""
    if correct is not None:
        mark = "✓" if correct else "✗"
        status = f"  ({mark} expected={expected})"
    ax_img.set_title(
        f"{stem}  |  predicted={column_count} col{status}"
        f"  n_lines={len(line_nodes)}  med_width={med_width_frac:.2f}·W",
        fontsize=11,
    )
    ax_img.axis("off")

    for nd in line_nodes:
        in_right = split_x is not None and nd.bbox.xmin >= split_x
        color = RIGHT_COLOR if in_right else LEFT_COLOR
        rect = mpatches.Rectangle(
            (nd.bbox.xmin, nd.bbox.ymin),
            nd.bbox.xmax - nd.bbox.xmin,
            nd.bbox.ymax - nd.bbox.ymin,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
        )
        ax_img.add_patch(rect)

    handles = [
        mpatches.Patch(color=LEFT_COLOR, label="left / single col"),
        mpatches.Patch(color=RIGHT_COLOR, label="right col"),
    ]
    if split_x is not None:
        ax_img.axvline(
            x=split_x, color="limegreen", linewidth=2.5, linestyle="--"
        )
        handles.append(
            mpatches.Patch(color="limegreen", label=f"split_x={split_x:.0f}")
        )
    ax_img.legend(handles=handles, loc="upper right", fontsize=9)

    ax_cov = fig.add_subplot(gs[1])
    xs = np.arange(W)
    ax_cov.fill_between(xs, cov, alpha=0.4, color="steelblue")
    ax_cov.plot(xs, cov, color="steelblue", linewidth=0.8)
    ax_cov.axvline(
        x=band_start,
        color="orange",
        linewidth=1,
        linestyle=":",
        label="search band 20–80 %",
    )
    ax_cov.axvline(x=band_end, color="orange", linewidth=1, linestyle=":")
    if split_x is not None:
        ax_cov.axvline(
            x=split_x, color="limegreen", linewidth=2, linestyle="--"
        )
    ax_cov.set_xlim(0, W)
    ax_cov.set_ylabel("# lines covering x", fontsize=9)
    ax_cov.set_xlabel("x (px)", fontsize=9)
    ax_cov.set_title(
        f"Coverage profile  |  peak={max_cov}  band_min={band_min}"
        f"  gutter_ratio={gutter_ratio:.3f}  xmin_var={xmin_var:.0f}",
        fontsize=9,
    )
    ax_cov.legend(loc="upper right", fontsize=8)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_debug.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": stem,
        "n_lines": len(line_nodes),
        "pred": column_count,
        "split_x": f"{split_x:.0f}" if split_x is not None else "none",
        "med_width": f"{med_width_frac:.3f}",
        "xmin_var": f"{xmin_var:.0f}",
        "gutter_ratio": f"{gutter_ratio:.3f}",
        "correct": ("yes" if correct else "NO") if correct is not None else "n/a",
    }


def _print_table(rows, file=None):
    headers = [
        "image", "n_lines", "pred", "split_x",
        "med_width", "xmin_var", "gutter_ratio", "correct",
    ]
    widths = [
        max(len(h), max(len(str(r[h])) for r in rows)) + 2
        for h in headers
    ]
    sep = "-" * sum(widths)
    lines = [
        "".join(h.ljust(w) for h, w in zip(headers, widths)),
        sep,
        *(
            "".join(str(r[h]).ljust(w) for h, w in zip(headers, widths))
            for r in rows
        ),
    ]
    text = "\n".join(lines)
    print(text)
    if file:
        with open(file, "w") as fh:
            fh.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Debug column detection on a folder of folio images."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument(
        "--expected", type=int, choices=[1, 2], default=None
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Custom BLLA model (.mlmodel or .safetensors)",
    )
    args = parser.parse_args()

    folder = os.path.expanduser(args.folder)
    seg = KrakenSegmentation(device="cpu", model=args.model)
    paths = _image_paths(folder)
    print(f"Processing {len(paths)} image(s) from {folder}\n")

    rows = [_process(p, seg, args.expected, folder) for p in paths]

    print("\n" + "=" * 80)
    out_dir = os.path.join(folder, "debug")
    summary_path = os.path.join(out_dir, "summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    _print_table(rows, file=summary_path)
    print(f"\nDebug images + summary → {out_dir}")


if __name__ == "__main__":
    main()

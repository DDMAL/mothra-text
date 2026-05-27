"""Cantus-CSV-backed manifest for ground-truth word segmentation.

Builds a {node_label: text_fragment} manifest for one manuscript folio by:
  1. Fetching the Cantus CSV export for the manuscript source.
  2. Splitting each chant's text across its manuscript lines using the
     volpiano column's 7 line-break markers.
  3. Zipping the resulting ordered line list with the HTRflow node labels
     for that folio page.

The finished manifest is consumed by make_manifest_lookup(), which returns
the GroundTruthLookup callable expected by GroundTruthWordSegmentation.
"""

import csv
import io
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Type alias matching GroundTruthLookup in ground_truth_word_segmentation.py.
# Uses Any for the node parameter to avoid a hard htrflow import here.
GroundTruthLookup = Callable[[Any], Optional[str]]

_CANTUS_CSV_URL = "https://cantusdatabase.org/source/{source_id}/csv/"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip Cantus pipe separators and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r" \| ", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_by_volpiano(text: str, volpiano: str) -> list[str]:
    """Split a chant's text into per-manuscript-line fragments.

    Uses the volpiano column's structural-break markers to determine line
    boundaries:
      7   = line break
      77  = page break  (also terminates a line)
      777 = column break (also terminates a line)
      --- = word boundary within the volpiano
      --  = syllable boundary within a word

    Any run of one or more 7s is treated as a single line break. The number
    of '---'-separated word groups in each resulting volpiano segment gives
    the word count for that manuscript line; those words are taken from the
    chant's plain text in order.

    If a segment after a line-break marker begins without a '---' word
    boundary (i.e. it starts with '--' or directly with notes), its first
    group is a mid-word continuation of the last word of the previous line
    and is not counted as a new word for the current line.

    If volpiano is absent or empty, returns the full text as a single
    fragment. If text is empty, returns an empty list.

    Args:
        text:     Cleaned chant text (space-separated words).
        volpiano: Volpiano string from the Cantus CSV.

    Returns:
        Ordered list of text fragments, one per manuscript line.
    """
    if not text:
        return []
    if not volpiano:
        return [text]

    # Split on any run of 7s (line / page / column break)
    vp_segments = re.split(r"7+", volpiano)
    words = text.split()
    result: list[str] = []
    word_idx = 0

    for i, seg in enumerate(vp_segments):
        # A word group is a stretch between '---' separators that contains at
        # least one non-hyphen character (a real note or clef symbol).
        word_groups = [
            g for g in seg.split("---") if re.search(r"[a-zA-Z]", g)
        ]
        n = len(word_groups)
        if n == 0:
            continue

        # Detect mid-word continuation: when a line break falls inside a word,
        # the segment after '7' starts with '--' (syllable boundary) rather
        # than a note letter (new word) or '---' (word boundary). Strip any
        # leading clef digit first, then check. The first group is not a new
        # word — it belongs to the last word of the previous line.
        seg_stripped = re.sub(r"^[1-9]", "", seg)
        mid_word_start = (
            i > 0
            and bool(re.match(r"--[^-]", seg_stripped))
        )
        new_words = n - 1 if mid_word_start else n

        if new_words <= 0:
            continue
        chunk = words[word_idx:word_idx + new_words]
        if chunk:
            result.append(" ".join(chunk))
        word_idx += new_words

    # If the volpiano is shorter than the text (e.g. incomplete notation),
    # append any remaining words to the last produced fragment.
    if word_idx < len(words):
        tail = " ".join(words[word_idx:])
        if result:
            result[-1] += " " + tail
        else:
            result.append(tail)

    return result if result else [text]


# ---------------------------------------------------------------------------
# CSV fetch
# ---------------------------------------------------------------------------

def fetch_cantus_csv(source_id: int) -> list[dict]:
    """Download the Cantus CSV for a source and return it as a list of dicts.

    Fetches cantusdatabase.org/csv/{source_id}.  Raises urllib.error.HTTPError
    on non-200 responses.

    Args:
        source_id: Integer source ID as listed on cantusdatabase.org.

    Returns:
        List of row dicts (keys from the CSV header row).
    """
    url = _CANTUS_CSV_URL.format(source_id=source_id)
    logger.info("Fetching Cantus CSV from %s", url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "mothra-text/0.1"}
    )
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("utf-8-sig")  # handle BOM if present
    return list(csv.DictReader(io.StringIO(content)))


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def _label_sort_key(label: str) -> tuple:
    """Sort key that orders node labels by embedded integers numerically.

    Ensures region0_line2 < region0_line10 (numeric, not lexicographic).
    """
    parts = re.split(r"(\d+)", label)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def build_page_manifest(
    csv_rows: list[dict],
    folio: str,
    node_labels: list[str],
    line_offset: int = 0,
) -> dict[str, str]:
    """Build a node-label → text-fragment mapping for one folio page.

    Args:
        csv_rows:     All rows from a Cantus CSV (any folio); rows for other
                      folios are ignored.
        folio:        Folio string exactly as it appears in the Cantus CSV
                      (e.g. "006r").
        node_labels:  HTRflow node labels for the line nodes on this page, in
                      any order (sorted internally by reading order).
        line_offset:  Number of Cantus line fragments to skip before aligning
                      with node labels.  Use when the image is a crop that
                      starts partway through the folio (e.g. line_offset=2
                      to start alignment at the 3rd Cantus line).

    Returns:
        Dict mapping each node label to its Cantus text fragment.  Node
        labels for which no Cantus text is available are omitted; the
        make_manifest_lookup() callable will return None for them, triggering
        the recognition-output fallback.
    """
    # Filter to this folio, excluding rows not physically present (mode="*")
    folio_rows = [
        r for r in csv_rows
        if r.get("folio", "").strip() == folio
        and r.get("mode", "").strip() != "*"
    ]
    excluded = sum(
        1 for r in csv_rows
        if r.get("folio", "").strip() == folio
        and r.get("mode", "").strip() == "*"
    )
    if excluded:
        logger.info(
            "  Excluded %d mode='*' row(s) for folio %r (not physically present)",
            excluded, folio,
        )
    folio_rows.sort(key=lambda r: int(r.get("sequence") or 0))

    if not folio_rows:
        logger.warning("No Cantus rows found for folio %r", folio)
        return {}

    # Build ordered list of per-line text fragments for the whole folio
    line_texts: list[str] = []
    for row in folio_rows:
        # Prefer manuscript spelling (source spelling); fall back to
        # standardized spelling only when the manuscript field is empty.
        raw_text = (
            row.get("fulltext_ms") or row.get("fulltext_standardized") or ""
        ).strip()
        text = clean_text(raw_text)
        if not text:
            continue
        volpiano = (row.get("volpiano") or "").strip()
        fragments = split_by_volpiano(text, volpiano)
        line_texts.extend(fragments)

    if line_offset:
        logger.info(
            "Skipping first %d Cantus line(s) for folio %r (line_offset)",
            line_offset, folio,
        )
        line_texts = line_texts[line_offset:]

    sorted_labels = sorted(node_labels, key=_label_sort_key)

    if len(line_texts) != len(sorted_labels):
        logger.warning(
            "Cantus line count (%d) differs from node count (%d) for folio %r;"
            " extra %s will be ignored",
            len(line_texts),
            len(sorted_labels),
            folio,
            "Cantus lines"
            if len(line_texts) > len(sorted_labels)
            else "node labels",
        )

    return dict(zip(sorted_labels, line_texts))


# ---------------------------------------------------------------------------
# Lookup factory
# ---------------------------------------------------------------------------

def make_manifest_lookup(manifest: dict[str, str]) -> GroundTruthLookup:
    """Return a gt_lookup callable backed by a pre-built manifest dict.

    Args:
        manifest: Mapping from node label to Cantus text fragment, as
                  produced by build_page_manifest().

    Returns:
        Callable[[node], Optional[str]] suitable for passing to
        GroundTruthWordSegmentation.
    """
    return lambda node: manifest.get(node.label)


def load_local_csv(path: str | Path) -> list[dict]:
    """Load a local Cantus-format CSV file as a list of dicts."""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_manifest(path: str | Path) -> dict[str, str]:
    """Load a manifest JSON file saved by build_gt_manifest.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

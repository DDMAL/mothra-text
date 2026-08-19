"""NW-based chant text allocator for the mothra-text pipeline.

Sub-plans 4a, 4b, and 4c: flat text builder, volpiano anchor
extraction, NW-based line allocation, and folio-state persistence
for continuation chants.

Flattens all chant words for a folio into a single ordered sequence
(FlatTextData) and records volpiano-derived break positions as
Anchors. This replaces the one-to-one row→line approach of
build_page_manifest(), which fails when a chant starts mid-line
after the previous chant ends.

The NW alignment step (4b) will consume FlatTextData and map each
detected line to a word span using Needleman-Wunsch alignment
against OCR output.

FolioState (4c) carries post-77 continuation words across folio
runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Anchor:
    word_index: int   # index in flat_text.words after the break
    anchor_type: str  # within_chant_7 | page_break_77 | column_break_777


@dataclass
class ChantSpan:
    sequence: int    # CSV sequence number; 0 for prev-folio continuation
    start_word: int  # inclusive index into flat_text.words
    end_word: int    # exclusive index into flat_text.words


@dataclass
class MidWordBreak:
    # The split word is flat_text.words[anchor_word_index - 1].
    anchor_word_index: int  # index of first word on next line after the break
    syl_left: int           # syllables of split word on line BEFORE the break
    syl_right: int          # syllables of split word on line AFTER the break


@dataclass
class FlatTextData:
    words: list[str]
    anchors: list[Anchor]
    chant_spans: list[ChantSpan]
    mid_word_breaks: list[MidWordBreak] = field(default_factory=list)
    initial_pointer: int = 0
    continuation_words: list[str] = field(default_factory=list)
    has_continuation: bool = False  # True if continuation was prepended
    suffix_probe_words: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _folio_sort_key(folio: str) -> tuple[int, int]:
    """Return a sortable (num, side) key for a folio string like '006r'."""
    m = re.match(r"0*(\d+)([rv])", folio.strip().lower())
    if m:
        return (int(m.group(1)), 0 if m.group(2) == "r" else 1)
    return (0, 0)


def _classify_break(break_str: str) -> str:
    n = len(break_str)
    if n == 1:
        return "within_chant_7"
    if n == 2:
        return "page_break_77"
    return "column_break_777"


def _count_syl_groups(volpiano_group: str) -> int:
    """Count syllable groups (split by --) within a single ---group.

    Safe to split on '--' because a single ---group by construction
    contains no '---' sequences.
    """
    parts = [
        p for p in volpiano_group.split("--")
        if re.search(r"[a-zA-Z]", p)
    ]
    return max(1, len(parts))


def _split_word_at_syl_boundary(
    word: str, syl_left: int, syl_right: int
) -> tuple[str, str] | None:
    """Split word text at a volpiano-derived syllable boundary.

    Returns (left_fragment, right_fragment) where left_fragment contains
    the first syl_left syllables joined without hyphens, and right_fragment
    contains the remaining syl_right syllables.  Returns None when the word
    cannot be split cleanly — e.g. the linguistic syllable count disagrees
    with syl_left + syl_right, or syllabification raises LatinError.
    """
    import unicodedata as _ud
    try:
        from volpiano_display_utilities.latin_word_syllabification import (
            LatinError,
            split_word_by_syl_bounds,
            syllabify_word,
        )
    except ImportError:
        return None

    normalized = re.sub(
        r"[^a-zA-Z]", "",
        _ud.normalize("NFKD", word).encode("ascii", "ignore").decode(),
    ).lower()
    if not normalized:
        return None
    try:
        bounds = syllabify_word(normalized, return_string=False)
        syllables = split_word_by_syl_bounds(normalized, bounds)
    except LatinError:
        return None

    if len(syllables) < syl_left + syl_right:
        return None
    if syl_left <= 0 or syl_right <= 0:
        return None

    left = "".join(s.rstrip("-") for s in syllables[:syl_left])
    right = "".join(s.rstrip("-") for s in syllables[syl_left:])
    return left, right


def _parse_row_words_and_anchors(
    raw_text: str, volpiano: str,
) -> tuple[
    list[str],
    list[tuple[int, str]],
    list[str],
    list[tuple[int, int, int]],
]:
    """Parse one chant row's text+volpiano into words, anchors, and breaks.

    Also identifies post-77 continuation words for the next folio.

    Args:
        raw_text: this row's fulltext_ms/fulltext_standardized, NOT yet
            passed through clean_text() — this function cleans it
            internally (see the pipe-offset note below).
        volpiano: this row's volpiano field.

    Returns:
        (this_folio_words, raw_anchors,
         continuation_words, raw_mid_word_breaks)

        - this_folio_words: words on this folio (up to first 77, if any)
        - raw_anchors: [(word_offset, anchor_type), ...] where
          word_offset is the cumulative count AFTER the break
        - continuation_words: words after the first 77 break (belong
          to the next folio; no separate CSV row for these)
        - raw_mid_word_breaks: [(anchor_word_index, syl_left, syl_right), ...]
          for each within_chant_7 break that falls mid-word; syl_left and
          syl_right are the syllable counts on each side of the break

    Note on Cantus '|' phrase separators: clean_text() strips '|' entirely,
    but the volpiano field's own word-group structure still allocates a
    position for it, so anchor/mid-word-break indices computed from the
    volpiano land one position too high for every '|' before them. `|` is
    already its own whitespace-delimited token in the raw text, so counting
    it via a plain split (no cleaning) gives an exact position-by-position
    offset; every index this function returns is corrected by that offset
    before being handed back, so callers always get valid indices into the
    returned (cleaned) word list.
    """
    from steps.gt_manifest import clean_text  # noqa: PLC0415

    words = clean_text(raw_text).split() if raw_text else []
    if not words:
        return [], [], [], []
    if not volpiano:
        return words, [], [], []

    raw_tokens = raw_text.split() if raw_text else []
    pipe_offsets = [0] * (len(raw_tokens) + 1)
    running = 0
    for idx, tok in enumerate(raw_tokens):
        pipe_offsets[idx] = running
        if tok == "|":
            running += 1
    pipe_offsets[len(raw_tokens)] = running

    def _to_real_index(raw_idx: int) -> int:
        raw_idx = min(max(raw_idx, 0), len(raw_tokens))
        return raw_idx - pipe_offsets[raw_idx]

    # Split volpiano keeping breaks: ["seg0", "7+", "seg1", "7+", ...]
    parts = re.split(r"(7+)", volpiano)

    word_idx = 0
    raw_anchors: list[tuple[int, str]] = []
    raw_mid_word_breaks: list[tuple[int, int, int]] = []
    continuation_start: int | None = None
    seg_count = 0  # number of volpiano segments processed so far
    prev_seg = ""  # previous iteration's segment (used for syl_left)

    i = 0
    while i < len(parts):
        seg = parts[i]
        i += 1

        # Count new words contributed by this segment.
        # A word group is a stretch between '---' separators with
        # at least one letter (clef digits like '9' are excluded).
        word_groups = [
            g for g in seg.split("---")
            if re.search(r"[a-zA-Z]", g)
        ]
        n = len(word_groups)
        if n > 0:
            # When a line break falls inside a word, the following
            # segment starts with '--' (syllable, not word boundary).
            # Its first group is a mid-word continuation and does not
            # represent a new word.
            seg_stripped = re.sub(r"^[1-9]", "", seg)
            if seg_count > 0 and re.match(r"--[^-]", seg_stripped):
                n -= 1
                # Record syllable counts on each side of this mid-word break.
                # Left: syllables in the last ---group of the previous segment.
                prev_groups = [
                    g for g in prev_seg.split("---")
                    if re.search(r"[a-zA-Z]", g)
                ]
                last_group = prev_groups[-1] if prev_groups else ""
                syl_left = _count_syl_groups(last_group)
                # Right: syllables in the continuation before the first ---.
                continuation = seg_stripped.split("---")[0]
                continuation_body = re.sub(r"^--", "", continuation)
                syl_right = _count_syl_groups(continuation_body)
                # word_idx is anchor_word_index (before adding n).
                raw_mid_word_breaks.append((word_idx, syl_left, syl_right))
            word_idx += max(n, 0)
        prev_seg = seg
        seg_count += 1

        # Process the break that follows this segment (if any).
        if i < len(parts):
            break_str = parts[i]
            i += 1
            anchor_type = _classify_break(break_str)
            raw_anchors.append((word_idx, anchor_type))

            if anchor_type == "page_break_77" and continuation_start is None:
                continuation_start = word_idx

    # Correct every index from volpiano-native (pipe-inclusive) position
    # space into real (post-clean_text) word-space before using any of
    # them against `words`, which is already in real word-space.
    raw_anchors = [(_to_real_index(wi), at) for wi, at in raw_anchors]
    raw_mid_word_breaks = [
        (_to_real_index(wi), sl, sr) for (wi, sl, sr) in raw_mid_word_breaks
    ]
    if continuation_start is not None:
        continuation_start = _to_real_index(continuation_start)

    # Determine which words stay on this folio vs continue to next.
    if continuation_start is not None:
        this_folio_words = words[:continuation_start]
        continuation_words = words[continuation_start:]
        # Drop anchors/mid-word breaks at or after the continuation boundary
        # (they describe structure on the next folio, not this one).
        raw_anchors = [
            (wi, at) for wi, at in raw_anchors
            if wi <= continuation_start
        ]
        raw_mid_word_breaks = [
            (wi, sl, sr) for (wi, sl, sr) in raw_mid_word_breaks
            if wi < continuation_start
        ]
    else:
        this_folio_words = words
        continuation_words = []

    return (
        this_folio_words, raw_anchors, continuation_words, raw_mid_word_breaks
    )


# ---------------------------------------------------------------------------
# Public API — Sub-plan 4a
# ---------------------------------------------------------------------------

def build_flat_text_and_anchors(
    csv_rows: list[dict],
    folio: str,
    prev_folio_state: FolioState | None = None,
    infer_continuation: bool = True,
) -> FlatTextData:
    """Build flat word sequence and volpiano anchors for one folio.

    Args:
        csv_rows:           All rows from a Cantus CSV (any folio).
        folio:              Folio string to filter rows (e.g. "006r").
        prev_folio_state:   FolioState from the previous folio run. If
                            provided (even with an empty remaining_words -
                            e.g. the previous folio's chant fully
                            terminated there), it is authoritative and
                            infer_continuation is never consulted: an
                            explicit answer of "nothing to carry over"
                            is real information from the run itself, not
                            an absence of information.
        infer_continuation: When True (default) and prev_folio_state is
                            None (no chain information at all - e.g. a
                            standalone single-folio run), scan csv_rows
                            for the immediately preceding folio (by CSV
                            ordering) with a 77 break and prepend its
                            post-77 words. Handles the common case where
                            the previous folio was not run first.

    Returns:
        FlatTextData with words, anchors, chant_spans, initial_pointer,
        continuation_words (post-77 words for the next folio), and
        has_continuation (True when continuation words were prepended).
    """
    from steps.gt_manifest import clean_text  # noqa: PLC0415

    words: list[str] = []
    anchors: list[Anchor] = []
    chant_spans: list[ChantSpan] = []
    mid_word_breaks: list[MidWordBreak] = []
    continuation_words: list[str] = []
    has_continuation = False

    # Prepend continuation words carried from the previous folio's 77 break.
    # An explicitly-given prev_folio_state is authoritative even when its
    # remaining_words is empty (fully_consumed=True, i.e. the actual previous
    # folio's chant terminated with nothing left over) - that's real
    # information from the run itself, and must NOT fall through to the
    # infer_continuation CSV-guess below just because the list happens to be
    # empty. Without this, a folio whose true predecessor left zero
    # continuation (e.g. a phantom folio like "003r" whose chant ends there)
    # would still get the CSV-scan's stale guess from whichever earlier folio
    # happens to have a 77, even though the real chain already answered
    # "nothing to carry over" for this exact folio.
    if prev_folio_state is not None:
        if prev_folio_state.remaining_words:
            words.extend(prev_folio_state.remaining_words)
            chant_spans.append(ChantSpan(
                sequence=0,
                start_word=0,
                end_word=len(words),
            ))
            has_continuation = True
    elif infer_continuation:
        # Check only the immediately preceding folio for a 77 break.
        # Searching all preceding folios would incorrectly grab a stale
        # 77 from much earlier in the manuscript when intervening folios
        # have no 77 (or when the CSV has out-of-order folio entries).
        target_key = _folio_sort_key(folio)
        preceding_keys = [
            _folio_sort_key(r.get("folio", ""))
            for r in csv_rows
            if _folio_sort_key(r.get("folio", "")) < target_key
            and r.get("mode", "").strip() != "*"
        ]
        prev_77_rows = []
        if preceding_keys:
            prev_folio_key = max(preceding_keys)
            prev_77_rows = [
                r for r in csv_rows
                if _folio_sort_key(r.get("folio", "")) == prev_folio_key
                and "77" in (r.get("volpiano") or "")
                and r.get("mode", "").strip() != "*"
            ]
            prev_77_rows.sort(key=lambda r: int(r.get("sequence") or 0))
        if prev_77_rows:
            carry_row = prev_77_rows[-1]
            raw_carry = (
                carry_row.get("fulltext_ms")
                or carry_row.get("fulltext_standardized")
                or ""
            ).strip()
            _, _, carry_words, _ = _parse_row_words_and_anchors(
                raw_carry,
                (carry_row.get("volpiano") or "").strip(),
            )
            if carry_words:
                words.extend(carry_words)
                chant_spans.append(ChantSpan(
                    sequence=0,
                    start_word=0,
                    end_word=len(words),
                ))
                has_continuation = True

    # Filter rows to this folio; exclude mode="*" (not physically present).
    folio_rows = [
        r for r in csv_rows
        if r.get("folio", "").strip() == folio
        and r.get("mode", "").strip() != "*"
    ]
    folio_rows.sort(key=lambda r: int(r.get("sequence") or 0))

    no_volpiano_count = 0
    for row in folio_rows:
        raw_text = (
            row.get("fulltext_ms") or row.get("fulltext_standardized") or ""
        ).strip()
        text = clean_text(raw_text)
        if not text:
            continue

        volpiano = (row.get("volpiano") or "").strip()
        if not volpiano:
            no_volpiano_count += 1
        row_words, raw_anchors, row_continuation, row_mwbs = (
            _parse_row_words_and_anchors(raw_text, volpiano)
        )

        if not row_words and not row_continuation:
            continue

        span_start = len(words)
        seq = int(row.get("sequence") or 0)

        for word_offset, anchor_type in raw_anchors:
            anchors.append(Anchor(
                word_index=span_start + word_offset,
                anchor_type=anchor_type,
            ))

        for anchor_wi, syl_left, syl_right in row_mwbs:
            mid_word_breaks.append(MidWordBreak(
                anchor_word_index=span_start + anchor_wi,
                syl_left=syl_left,
                syl_right=syl_right,
            ))

        words.extend(row_words)
        chant_spans.append(ChantSpan(
            sequence=seq,
            start_word=span_start,
            end_word=len(words),
        ))

        if row_continuation:
            continuation_words = row_continuation
            # No more rows contribute to this folio after a 77 break.
            break

    if no_volpiano_count:
        logger.warning(
            "%d of %d chant row(s) on folio %r have no volpiano; "
            "NW alignment only for those rows (no line-break anchors available)",
            no_volpiano_count, len(folio_rows), folio,
        )

    initial_pointer = 0

    # When no continuation was found, try to build a suffix probe from the
    # immediately preceding folio's last chant row.  Used by allocate_lines
    # to assign CSV ground-truth text to pre-start lines via suffix NW.
    suffix_probe_words: list[str] = []
    if not has_continuation and infer_continuation:
        target_key = _folio_sort_key(folio)
        preceding_rows = [
            r for r in csv_rows
            if _folio_sort_key(r.get("folio", "")) < target_key
            and r.get("mode", "").strip() != "*"
        ]
        if preceding_rows:
            max_key = max(
                _folio_sort_key(r.get("folio", "")) for r in preceding_rows
            )
            last_folio_rows = sorted(
                [r for r in preceding_rows
                 if _folio_sort_key(r.get("folio", "")) == max_key],
                key=lambda r: int(r.get("sequence") or 0),
            )
            suffix_row = last_folio_rows[-1]
            raw_suffix = (
                suffix_row.get("fulltext_ms")
                or suffix_row.get("fulltext_standardized")
                or ""
            ).strip()
            if raw_suffix:
                suffix_probe_words, _, _, _ = _parse_row_words_and_anchors(
                    raw_suffix, ""
                )

    return FlatTextData(
        words=words,
        anchors=anchors,
        chant_spans=chant_spans,
        mid_word_breaks=mid_word_breaks,
        initial_pointer=initial_pointer,
        continuation_words=continuation_words,
        has_continuation=has_continuation,
        suffix_probe_words=suffix_probe_words,
    )


# ---------------------------------------------------------------------------
# Sub-plan 4b — NW allocation
# ---------------------------------------------------------------------------

@dataclass
class ValidationFlag:
    flag_type: str
    detail: str


@dataclass
class AllocationResult:
    manifest: dict[str, str]   # node_label → word-fragment string
    flags: list[ValidationFlag]
    text_pointer_end: int       # flat_text index after last consumed word
    debug_lines: list[dict] | None = None  # set when debug=True
    folio_start_line: int = 0  # L* index (0 when no pre-start region)
    constituent_overrides: dict[str, str] = field(default_factory=dict)


def _chant_starts_in_range(
    start: int,
    end: int,
    chant_spans: list[ChantSpan],
) -> bool:
    """Return True if any chant span starts in the interval (start, end].

    Guards force_window: if a new chant begins between text_pointer and
    the anchor, forcing overrides NW's evidence about which physical line
    that chant first appears on.
    """
    return any(start < span.start_word <= end for span in chant_spans)


def locate_first_chant_line(
    flat_text: FlatTextData,
    sorted_labels: list[str],
    ocr_texts: dict[str, str],
    aligner=None,
    n_probe_words: int = 8,
) -> tuple[int, float] | None:
    """Return (line_index, normalised_score) for the best NW-matching OCR line.

    Searches for the line in sorted_labels whose OCR text best matches the
    opening words of the first folio chant (the first ChantSpan with
    sequence > 0).  Used in no-volpiano mode to estimate where this folio's
    content actually begins on the page, independent of has_continuation.

    Works at line granularity.  When the first folio chant begins mid-line
    (the previous chant ends partway through a physical line), the returned
    index points to that mixed line — a known approximation.

    Args:
        flat_text:      Output of build_flat_text_and_anchors().
        sorted_labels:  Line node labels in reading order.
        ocr_texts:      {label: ocr_string} mapping.
        aligner:        Bio.Align.PairwiseAligner to reuse.  When None, a
                        default aligner with the same scoring params as
                        allocate_lines() is constructed internally.
        n_probe_words:  Number of words from the first folio chant to use
                        as the probe string (default 8).

    Returns:
        (line_index, normalised_score) for the best-matching line, or None
        when no folio chant span (sequence > 0) exists, no probe words are
        available, or every OCR line is empty.
    """
    from math import sqrt

    first_folio_span = next(
        (s for s in flat_text.chant_spans if s.sequence > 0), None
    )
    if first_folio_span is None:
        return None

    span_start = first_folio_span.start_word
    probe_words = flat_text.words[span_start:span_start + n_probe_words]
    if not probe_words:
        return None

    probe_text = " ".join(probe_words)

    if aligner is None:
        from Bio.Align import PairwiseAligner as _PA
        _al = _PA()
        _al.mode = "global"
        _al.match_score = 8.0
        _al.mismatch_score = -5.0
        _al.open_gap_score = -7.0
        _al.extend_gap_score = -3.0
        aligner = _al

    best_score = float("-inf")
    best_idx = 0
    any_ocr = False

    for i, label in enumerate(sorted_labels):
        ocr = (ocr_texts.get(label) or "").strip()
        if not ocr:
            continue
        any_ocr = True
        raw = aligner.score(ocr, probe_text)
        denom = sqrt(len(ocr) * len(probe_text))
        norm = raw / denom if denom > 0 else float("-inf")
        if norm > best_score:
            best_score = norm
            best_idx = i

    if not any_ocr:
        return None

    return best_idx, best_score


def allocate_lines(
    flat_text: FlatTextData,
    sorted_labels: list[str],
    ocr_texts: dict[str, str],
    column_count: int = 1,
    left_column_count: int = 0,
    search_window: int = 40,
    snap_window: int = 2,
    force_window: int = 10,
    match_score: float = 8.0,
    mismatch_score: float = -5.0,
    open_penalty: float = -7.0,
    extend_penalty: float = -3.0,
    debug: bool = False,
    locate_folio_start: bool = True,
    folio_start_n_probe: int = 8,
    folio_start_min_score: float = 0.0,
    pre_start_suffix_align: bool = True,
    pre_start_suffix_min_score: float = 0.0,
    fused_lines: list | None = None,
    node_ocr: dict | None = None,
    mixed_line_n_words: int = 3,
    mixed_line_min_score: float = 0.0,
) -> AllocationResult:
    """Align OCR text for each line against flat_text using NW alignment.

    For each label in sorted_labels, scores candidate word spans of
    lengths 1..search_window by normalising the NW alignment score by
    the geometric mean of the OCR string length and the candidate span
    length.  Optionally snaps the result to the nearest volpiano anchor
    within snap_window words.

    When model=None was used for KrakenRecognition, all OCR texts will
    be empty strings.  In stub mode each line is advanced to the next
    anchor position.  When no anchor is available (e.g. all chants on
    the folio lack volpiano), remaining words are distributed uniformly
    across remaining lines rather than assigning 1 word per line.

    Args:
        flat_text:          Output of build_flat_text_and_anchors().
        sorted_labels:      Line node labels in reading order (left
                            column first, then right column).
        ocr_texts:          {label: ocr_string} mapping.  Missing
                            labels or empty strings trigger stub mode.
        column_count:       1 or 2 (from cluster_columns()).
        left_column_count:  Number of labels in the left column.
                            When > 0 and column_count >= 2, pointer is
                            hard-reset to the column_break_777 anchor
                            at label index left_column_count.
        search_window:      Maximum words to look ahead per line.
        snap_window:        Anchor snap tolerance in words.  NW results
                            within this distance of the next volpiano
                            anchor are snapped to it silently; larger
                            differences emit nw_volpiano_disagreement.
        force_window:       Mid-chant force tolerance in words.  When
                            > 0, within_chant_7 anchors are forced even
                            beyond snap_window, unless a new chant span
                            starts between text_pointer and the anchor
                            (which would override NW's evidence about
                            which line that chant appears on).
                            Set to 0 to disable the feature entirely.
        match_score, mismatch_score, open_penalty, extend_penalty:
                            Bio.Align.PairwiseAligner scoring params.
                            Defaults are calibrated for medieval chant.
        debug:              When True, populate AllocationResult.
                            debug_lines with per-line OCR and NW
                            alignment detail.
        locate_folio_start: When True (default), attempt to locate
                            where this folio's first chant begins on
                            the page.  Only activates in no-volpiano
                            mode (flat_text.anchors empty) with at
                            least one non-empty OCR line.  Lines
                            before L* are treated as the previous
                            folio's bleeding continuation.
        folio_start_n_probe: Words from the first folio chant used as
                            the NW probe string (default 8).
        folio_start_min_score: Minimum normalised NW score to accept
                            the located line.  When no line reaches
                            this threshold a folio_start_not_located
                            flag is emitted and alignment reverts to
                            line 0 (default 0.0, i.e. net-positive).
        pre_start_suffix_align: When True (default), attempt to assign
                            CSV text to pre-start lines by aligning the
                            concatenated pre-start OCR against the
                            preceding folio's last chant (stored in
                            flat_text.suffix_probe_words).  Only
                            activates when has_continuation=False and
                            folio_start_line > 0.
        pre_start_suffix_min_score: Quality gate for the suffix
                            alignment score.  Below this threshold a
                            suffix_alignment_skipped flag is emitted
                            and pre-start lines fall back to ""
                            (default 0.0).
        fused_lines:        Optional list of FusedLine objects
                            (from fuse_colinear_segments).  Required
                            for mixed-line detection.  When None,
                            mixed-line detection is skipped entirely.
        node_ocr:           Per-segment OCR dict {label: text}, as
                            built before fusion.  Required alongside
                            fused_lines for mixed-line detection.
        mixed_line_n_words: Maximum number of folio chant words to
                            detect on fused_{L*-1}'s right constituents
                            (default 3).  Capped at 3 because beyond
                            that the NW main loop would already be
                            correct.
        mixed_line_min_score: NW quality gate for mixed-line detection.
                            Detection is skipped when the best score
                            falls below this threshold (default 0.0).

    Returns:
        AllocationResult with manifest, validation flags, and
        text_pointer_end.  debug_lines is populated when debug=True.
    """
    from math import sqrt

    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_penalty
    aligner.extend_gap_score = extend_penalty

    manifest: dict[str, str] = {}
    flags: list[ValidationFlag] = []
    text_pointer = flat_text.initial_pointer
    debug_lines_out: list[dict] | None = [] if debug else None

    # Sort anchors by word_index once for efficient lookup.
    sorted_anchors = sorted(flat_text.anchors, key=lambda a: a.word_index)

    # Index mid-word breaks by anchor_word_index for O(1) lookup.
    mwb_by_anchor: dict[int, MidWordBreak] = {
        m.anchor_word_index: m for m in flat_text.mid_word_breaks
    }
    # Right-side syllable fragment of a split word, carried to the next line.
    syllable_prefix: str | None = None

    # Pre-compute column-1 word ceiling for two-column folios.
    col_break_word: int | None = None
    if column_count >= 2 and left_column_count > 0:
        _cb = next(
            (a for a in sorted_anchors if a.anchor_type == "column_break_777"),
            None,
        )
        if _cb:
            col_break_word = _cb.word_index

    # Heuristic: lowercase first word without prepended continuation
    # suggests the folio may begin mid-chant.
    if (
        not flat_text.has_continuation
        and flat_text.words
        and flat_text.words[0][0:1].islower()
    ):
        flags.append(ValidationFlag(
            "continuation_missing",
            f"First word of folio is lowercase ({flat_text.words[0]!r}); "
            "folio may start mid-chant. If the preceding folio lacked "
            "volpiano, automatic continuation inference was skipped — "
            "run the preceding folio with --folio-state-out and pass "
            "the result via --prev-folio-state.",
        ))

    # No-volpiano folio-start location.
    # Find the first ChantSpan with sequence > 0 (the first real chant of
    # this folio, as opposed to the continuation span from the previous folio).
    first_folio_span = next(
        (s for s in flat_text.chant_spans if s.sequence > 0), None
    )
    folio_start_line = 0
    if (
        locate_folio_start
        and not flat_text.anchors
        and first_folio_span is not None
        and any((ocr_texts.get(lbl) or "").strip() for lbl in sorted_labels)
    ):
        loc = locate_first_chant_line(
            flat_text, sorted_labels, ocr_texts,
            aligner=aligner, n_probe_words=folio_start_n_probe,
        )
        if loc is not None:
            candidate_line, loc_score = loc
            if loc_score >= folio_start_min_score:
                folio_start_line = candidate_line
                if folio_start_line > 0:
                    flags.append(ValidationFlag(
                        "folio_start_detected",
                        f"First folio chant located at line index "
                        f"{folio_start_line} (score={loc_score:.3f}); "
                        f"{folio_start_line} pre-start line(s) assigned "
                        "to previous folio's continuation.",
                    ))
            else:
                flags.append(ValidationFlag(
                    "folio_start_not_located",
                    f"No OCR line matched the first folio chant with "
                    f"score >= {folio_start_min_score} "
                    f"(best={loc_score:.3f}); reverting to line-0 "
                    "alignment.",
                ))

    # Pre-start suffix alignment: when folio_start_line > 0 and no continuation
    # words were carried forward, align the concatenated pre-start OCR against
    # the preceding folio's last chant to find which suffix of that chant falls
    # on this page.  Produces _suffix_words (words to distribute to pre-start
    # lines) and _suffix_ptr (position within _suffix_words).
    _suffix_words: list[str] = []
    _suffix_ptr: int = 0
    if (
        folio_start_line > 0
        and not flat_text.has_continuation
        and flat_text.suffix_probe_words
        and pre_start_suffix_align
    ):
        pre_start_ocr = " ".join(
            (ocr_texts.get(lbl) or "").strip()
            for lbl in sorted_labels[:folio_start_line]
            if (ocr_texts.get(lbl) or "").strip()
        )
        if pre_start_ocr:
            from Bio.Align import PairwiseAligner as _SGA
            _sg = _SGA()
            _sg.mode = "global"
            _sg.match_score = match_score
            _sg.mismatch_score = mismatch_score
            _sg.open_gap_score = open_penalty
            _sg.extend_gap_score = extend_penalty
            _sg.open_left_insertion_score = 0.0
            _sg.extend_left_insertion_score = 0.0
            _best_k, _best_norm = 0, float("-inf")
            for _k in range(len(flat_text.suffix_probe_words)):
                _cand = " ".join(flat_text.suffix_probe_words[_k:])
                if not _cand:
                    continue
                _raw = _sg.score(pre_start_ocr, _cand)
                _den = sqrt(len(pre_start_ocr) * len(_cand))
                _nm = _raw / _den if _den > 0 else float("-inf")
                if _nm > _best_norm:
                    _best_norm, _best_k = _nm, _k
            if _best_norm >= pre_start_suffix_min_score:
                _suffix_words = flat_text.suffix_probe_words[_best_k:]
                flags.append(ValidationFlag(
                    "suffix_alignment_detected",
                    f"Pre-start lines aligned to preceding folio's last chant "
                    f"starting at word {_best_k} (score={_best_norm:.3f}); "
                    f"{len(_suffix_words)} word(s) available for "
                    f"{folio_start_line} pre-start line(s).",
                ))
            else:
                flags.append(ValidationFlag(
                    "suffix_alignment_skipped",
                    f"Pre-start suffix alignment score {_best_norm:.3f} below "
                    f"threshold {pre_start_suffix_min_score}; pre-start lines "
                    "assigned empty.",
                ))

    # Mixed-line state: updated inside the pre-start loop for the last
    # pre-start line; used at the folio-region hard-reset.
    constituent_overrides: dict[str, str] = {}
    _mixed_word_skip: int = 0

    for label_idx, label in enumerate(sorted_labels):
        # Hard-reset at column boundary (two-column folios only).
        if (
            column_count >= 2
            and left_column_count > 0
            and label_idx == left_column_count
        ):
            col_break = next(
                (a for a in sorted_anchors if a.anchor_type == "column_break_777"),
                None,
            )
            if col_break:
                text_pointer = col_break.word_index
            else:
                flags.append(ValidationFlag(
                    "column_count_uncertain",
                    f"Expected column_break_777 anchor near word "
                    f"{text_pointer} but none found",
                ))

        # Pre-start region [0, folio_start_line): lines that belong to the
        # previous folio's bleeding continuation, not this folio's chants.
        if folio_start_line > 0 and label_idx < folio_start_line:
            _pre_ptr_start = text_pointer
            ocr = (ocr_texts.get(label) or "").strip()
            _pre_best_norm: float | None = None

            # Mixed-line detection: on the last pre-start line, check if
            # fused_{L*-1}'s rightmost constituents contain opening words
            # of this folio's first chant.  Runs before force-snap so
            # _suffix_ptr reflects exactly what this line will receive.
            if (
                label_idx == folio_start_line - 1
                and fused_lines is not None
                and node_ocr is not None
                and not flat_text.anchors
                and first_folio_span is not None
            ):
                _fused_prev = next(
                    (f for f in fused_lines if f.label == label), None
                )
                if (
                    _fused_prev is not None
                    and len(_fused_prev.constituent_labels) > 1
                ):
                    if _suffix_words:
                        _ml_suf = list(_suffix_words[_suffix_ptr:])
                    elif flat_text.has_continuation:
                        _ml_suf = flat_text.words[
                            text_pointer:first_folio_span.start_word
                        ]
                    else:
                        _ml_suf = []
                    _prev_lbls = _fused_prev.constituent_labels
                    _prev_ws = _fused_prev.constituent_widths
                    _fstart = first_folio_span.start_word
                    # Joint grid search over (split index k, word count n).
                    # k is where in this fused line's constituents the
                    # previous folio's suffix ends and this folio's first
                    # chant begins; n is how many of this folio's first
                    # words we're testing for. Neither can be fixed
                    # independently — a larger n needs a smaller k (fewer
                    # right-hand constituents to hold more words) and vice
                    # versa — so every (k, n) pair in range is scored via NW
                    # against the right-hand OCR and the best-scoring pair
                    # wins. Both axes are capped at mixed_line_n_words to
                    # keep this bounded (see steps/README.md's Mixed-line
                    # detection section for the feature-level description).
                    _n_try = min(
                        mixed_line_n_words, len(_prev_lbls) - 1
                    )
                    _best_ml: float = float("-inf")
                    _best_ml_k: int | None = None
                    _best_ml_n: int = 0
                    for _mk in range(
                        len(_prev_lbls) - _n_try, len(_prev_lbls)
                    ):
                        _right_ocr = " ".join(
                            (node_ocr.get(lbl) or "").strip()
                            for lbl in _prev_lbls[_mk:]
                            if (node_ocr.get(lbl) or "").strip()
                        )
                        if not _right_ocr:
                            continue
                        for _mn in range(1, mixed_line_n_words + 1):
                            _fw = flat_text.words[
                                _fstart: _fstart + _mn
                            ]
                            if not _fw:
                                break
                            _mc = " ".join(_fw)
                            _mr = aligner.score(_right_ocr, _mc)
                            _md = sqrt(
                                len(_right_ocr) * len(_mc)
                            )
                            _ms = (
                                _mr / _md if _md > 0
                                else float("-inf")
                            )
                            if _ms > _best_ml:
                                _best_ml = _ms
                                _best_ml_k = _mk
                                _best_ml_n = _mn
                    # Require a real suffix (_ml_suf) in addition to the
                    # score/k checks: without prior-folio words to assign
                    # to the left-hand constituents, there is nothing to
                    # split "left vs. right" — the override would just be
                    # this folio's words duplicated onto an empty left side.
                    if (
                        _best_ml >= mixed_line_min_score
                        and _best_ml_k is not None
                        and _ml_suf
                    ):
                        _llbls = _prev_lbls[:_best_ml_k]
                        _rlbls = _prev_lbls[_best_ml_k:]
                        _lws = _prev_ws[:_best_ml_k]
                        _rws = _prev_ws[_best_ml_k:]
                        # Left: suffix words across left constituents
                        _ltw = sum(_lws) or 1
                        _li2 = 0
                        for _li, (_ll, _lw) in enumerate(
                            zip(_llbls, _lws)
                        ):
                            if _li == len(_llbls) - 1:
                                constituent_overrides[_ll] = (
                                    " ".join(_ml_suf[_li2:])
                                )
                            else:
                                _lc = max(
                                    0,
                                    round(len(_ml_suf) * _lw / _ltw),
                                )
                                constituent_overrides[_ll] = (
                                    " ".join(_ml_suf[_li2:_li2 + _lc])
                                )
                                _li2 += _lc
                        # Right: first N folio words across right
                        _rfws = flat_text.words[
                            _fstart: _fstart + _best_ml_n
                        ]
                        _rtw = sum(_rws) or 1
                        _ri2 = 0
                        for _ri, (_rl, _rw) in enumerate(
                            zip(_rlbls, _rws)
                        ):
                            if _ri == len(_rlbls) - 1:
                                constituent_overrides[_rl] = (
                                    " ".join(_rfws[_ri2:])
                                )
                            else:
                                _rc = max(
                                    0,
                                    round(len(_rfws) * _rw / _rtw),
                                )
                                constituent_overrides[_rl] = (
                                    " ".join(_rfws[_ri2:_ri2 + _rc])
                                )
                                _ri2 += _rc
                        _mixed_word_skip = _best_ml_n
                        flags.append(ValidationFlag(
                            "mixed_start_detected",
                            f"{_best_ml_n} folio word(s) detected on "
                            f"pre-start line {label} at constituent "
                            f"index {_best_ml_k} "
                            f"(score={_best_ml:.3f}); "
                            "moved to that line.",
                        ))

            if flat_text.has_continuation:
                # Continuation words are prepended in flat_text.words[0:start].
                _pre_limit = first_folio_span.start_word
                _pre_cands = flat_text.words[text_pointer:_pre_limit]

                if not _pre_cands:
                    consumed = 0
                elif not ocr:
                    # Distribute remaining continuation words uniformly.
                    _rem_lines = folio_start_line - label_idx
                    _rem_words = _pre_limit - text_pointer
                    consumed = max(
                        1, _rem_words // max(_rem_lines, 1)
                    )
                else:
                    _pbest_norm = float("-inf")
                    _pbest_k = 1
                    for _pk in range(1, len(_pre_cands) + 1):
                        _pwin = " ".join(_pre_cands[:_pk])
                        _praw = aligner.score(ocr, _pwin)
                        _pden = sqrt(len(ocr) * len(_pwin))
                        _pnm = (
                            _praw / _pden if _pden > 0 else float("-inf")
                        )
                        if _pnm > _pbest_norm:
                            _pbest_norm = _pnm
                            _pbest_k = _pk
                    consumed = _pbest_k
                    _pre_best_norm = _pbest_norm

                # Force-snap: last pre-start line consumes all remaining
                # continuation words (mirrors the col_break_word force-close).
                if label_idx == folio_start_line - 1:
                    consumed = _pre_limit - text_pointer

                consumed = max(consumed, 0)
                manifest[label] = " ".join(
                    flat_text.words[text_pointer:text_pointer + consumed]
                )
                _is_forced = label_idx == folio_start_line - 1
                if debug and debug_lines_out is not None:
                    debug_lines_out.append({
                        "label": label,
                        "ocr": ocr,
                        "pointer_start": _pre_ptr_start,
                        "pointer_end": text_pointer + consumed,
                        "consumed": consumed,
                        "assigned": manifest[label],
                        "best_k_pre_snap": consumed,
                        "best_norm": _pre_best_norm,
                        "anchor_word": None,
                        "anchor_type": None,
                        "snapped": False,
                        "forced": _is_forced,
                        "alignment": "(pre-start continuation)",
                    })
                text_pointer += consumed

            elif _suffix_words:
                # Suffix alignment: distribute words from the preceding folio's
                # last chant across this pre-start line.  Uses _suffix_ptr to
                # track position within _suffix_words independently of
                # text_pointer (which stays at 0 until the hard-reset at L*).
                _suf_cands = _suffix_words[_suffix_ptr:]
                _suf_consumed: int
                _is_forced = label_idx == folio_start_line - 1

                if not _suf_cands:
                    _suf_consumed = 0
                elif not ocr:
                    _rem_lines = folio_start_line - label_idx
                    _rem_suf = len(_suf_cands)
                    _suf_consumed = max(1, _rem_suf // max(_rem_lines, 1))
                else:
                    _pbest_norm = float("-inf")
                    _pbest_k = 1
                    for _pk in range(1, len(_suf_cands) + 1):
                        _pwin = " ".join(_suf_cands[:_pk])
                        _praw = aligner.score(ocr, _pwin)
                        _pden = sqrt(len(ocr) * len(_pwin))
                        _pnm = (
                            _praw / _pden if _pden > 0 else float("-inf")
                        )
                        if _pnm > _pbest_norm:
                            _pbest_norm = _pnm
                            _pbest_k = _pk
                    _suf_consumed = _pbest_k
                    _pre_best_norm = _pbest_norm

                if _is_forced:
                    _suf_consumed = len(_suf_cands)

                _suf_consumed = max(_suf_consumed, 0)
                manifest[label] = " ".join(
                    _suffix_words[_suffix_ptr:_suffix_ptr + _suf_consumed]
                )
                if debug and debug_lines_out is not None:
                    debug_lines_out.append({
                        "label": label,
                        "ocr": ocr,
                        "pointer_start": _suffix_ptr,
                        "pointer_end": _suffix_ptr + _suf_consumed,
                        "consumed": _suf_consumed,
                        "assigned": manifest[label],
                        "best_k_pre_snap": _suf_consumed,
                        "best_norm": _pre_best_norm,
                        "anchor_word": None,
                        "anchor_type": None,
                        "snapped": False,
                        "forced": _is_forced,
                        "alignment": "(pre-start suffix alignment)",
                    })
                _suffix_ptr += _suf_consumed
                # text_pointer is intentionally not advanced here; it remains
                # at flat_text.initial_pointer until the hard-reset at L*.

            else:
                # No CSV data for this pre-start line.
                manifest[label] = ""
                if debug and debug_lines_out is not None:
                    debug_lines_out.append({
                        "label": label,
                        "ocr": ocr,
                        "pointer_start": _pre_ptr_start,
                        "pointer_end": _pre_ptr_start,
                        "consumed": 0,
                        "assigned": "",
                        "best_k_pre_snap": 0,
                        "best_norm": None,
                        "anchor_word": None,
                        "anchor_type": None,
                        "snapped": False,
                        "forced": False,
                        "alignment": "(pre-start continuation)",
                    })

            continue

        # Hard-reset when entering the folio region (zero-distance after
        # force-snap consumed all continuation words on last pre-start line).
        if folio_start_line > 0 and label_idx == folio_start_line:
            text_pointer = first_folio_span.start_word + _mixed_word_skip

        col1 = col_break_word is not None and label_idx < left_column_count
        word_limit = col_break_word if col1 else text_pointer + search_window
        candidate_words = flat_text.words[text_pointer: word_limit]
        if not candidate_words:
            manifest[label] = ""
            if debug and debug_lines_out is not None:
                debug_lines_out.append({
                    "label": label,
                    "ocr": "",
                    "pointer_start": text_pointer,
                    "pointer_end": text_pointer,
                    "consumed": 0,
                    "assigned": "",
                    "best_k_pre_snap": 0,
                    "best_norm": None,
                    "anchor_word": None,
                    "anchor_type": None,
                    "snapped": False,
                    "alignment": "(no words left)",
                })
            continue

        ocr = (ocr_texts.get(label) or "").strip()

        # Debug-only tracking variables (no effect on allocation logic).
        _dbg_ptr_start = text_pointer
        _dbg_best_k_pre_snap = 0
        _dbg_best_norm: float | None = None
        _dbg_next_snap: Anchor | None = None
        _dbg_snapped = False
        _dbg_forced = False
        _dbg_alignment = ""

        if not ocr:
            # Stub mode: advance to the next anchor of any type.
            next_anchor = next(
                (a for a in sorted_anchors if a.word_index > text_pointer),
                None,
            )
            if next_anchor is not None:
                consumed = next_anchor.word_index - text_pointer
            else:
                # No anchor available (all chants on this folio lack volpiano).
                # Distribute remaining words uniformly across remaining lines
                # so the manifest is populated rather than assigning 1 word
                # per line (the previous fallback).
                remaining_lines = len(sorted_labels) - label_idx
                remaining_words_count = len(flat_text.words) - text_pointer
                consumed = max(1, remaining_words_count // max(remaining_lines, 1))
        else:
            # Find word count maximising NW score / geometric mean of lengths.
            best_norm = float("-inf")
            best_k = 1
            for k in range(1, len(candidate_words) + 1):
                window = " ".join(candidate_words[:k])
                raw = aligner.score(ocr, window)
                denom = sqrt(len(ocr) * len(window))
                norm = raw / denom if denom > 0 else float("-inf")
                if norm > best_norm:
                    best_norm = norm
                    best_k = k
            consumed = best_k
            _dbg_best_k_pre_snap = best_k
            _dbg_best_norm = best_norm

            # Snap to the next within_chant_7 or page_break_77 anchor.
            next_snap = next(
                (a for a in sorted_anchors
                 if a.word_index > text_pointer
                 and a.anchor_type in ("within_chant_7", "page_break_77")
                 and (not col1 or a.word_index <= col_break_word)),
                None,
            )
            _dbg_next_snap = next_snap
            if next_snap:
                raw_end = text_pointer + consumed
                diff = abs(raw_end - next_snap.word_index)
                if diff <= snap_window:
                    consumed = next_snap.word_index - text_pointer
                    _dbg_snapped = True
                    if next_snap.anchor_type == "page_break_77" and diff > 0:
                        flags.append(ValidationFlag(
                            "page_break_77_mismatch",
                            f"Line {label}: snapped to page_break_77 at "
                            f"word {next_snap.word_index} (NW was {raw_end})",
                        ))
                elif (
                    force_window > 0
                    and diff <= force_window
                    and next_snap.anchor_type == "within_chant_7"
                    and not _chant_starts_in_range(
                        text_pointer,
                        next_snap.word_index,
                        flat_text.chant_spans,
                    )
                ):
                    consumed = next_snap.word_index - text_pointer
                    _dbg_forced = True
                    flags.append(ValidationFlag(
                        "forced_mid_chant_snap",
                        f"Line {label}: forced to within_chant_7 anchor"
                        f" at word {next_snap.word_index}"
                        f" (NW was {raw_end}, diff={diff})",
                    ))
                else:
                    flag_type = (
                        "page_break_77_mismatch"
                        if next_snap.anchor_type == "page_break_77"
                        else "nw_volpiano_disagreement"
                    )
                    flags.append(ValidationFlag(
                        flag_type,
                        f"Line {label}: NW end at word {raw_end}, "
                        f"anchor at {next_snap.word_index} (diff={diff})",
                    ))

            if debug:
                try:
                    final_window = " ".join(
                        candidate_words[:max(consumed, 1)]
                    )
                    _aligns = aligner.align(ocr, final_window)
                    _dbg_alignment = str(next(iter(_aligns)))
                except Exception:
                    _dbg_alignment = "(alignment unavailable)"

        consumed = max(consumed, 0)
        if col1:
            consumed = min(consumed, max(0, col_break_word - text_pointer))

        # Force the last col1 line to close at col_break_word so that any
        # word the NW left in the gap triggers the MWB lookup below.
        if (col_break_word is not None
                and label_idx == left_column_count - 1
                and text_pointer + consumed < col_break_word):
            consumed = col_break_word - text_pointer

        # Assemble this line's word list, optionally prepending a syllable
        # fragment carried over from a mid-word break on the previous line.
        assigned_words = list(
            flat_text.words[text_pointer: text_pointer + consumed]
        )
        if syllable_prefix is not None:
            assigned_words = [syllable_prefix] + assigned_words
            syllable_prefix = None
        manifest_text = " ".join(assigned_words)

        # Check whether the next pointer lands on a mid-word break and, if so,
        # split the last word of this line at the volpiano syllable boundary.
        new_pointer = text_pointer + consumed
        mwb = mwb_by_anchor.get(new_pointer)
        if mwb is not None and new_pointer > 0:
            split_word = flat_text.words[new_pointer - 1]
            split_result = _split_word_at_syl_boundary(
                split_word, mwb.syl_left, mwb.syl_right
            )
            if split_result is not None:
                left_frag, right_frag = split_result
                parts_m = manifest_text.split()
                parts_m[-1] = left_frag
                manifest_text = " ".join(parts_m)
                syllable_prefix = right_frag
            else:
                flags.append(ValidationFlag(
                    "mid_word_split_skipped",
                    f"Line {label}: could not split {split_word!r} "
                    f"({mwb.syl_left}+{mwb.syl_right} syllables); "
                    "word left intact on this line",
                ))

        manifest[label] = manifest_text
        text_pointer = new_pointer

        if debug and debug_lines_out is not None:
            debug_lines_out.append({
                "label": label,
                "ocr": ocr,
                "pointer_start": _dbg_ptr_start,
                "pointer_end": text_pointer,
                "consumed": consumed,
                "assigned": manifest[label],
                "best_k_pre_snap": _dbg_best_k_pre_snap,
                "best_norm": _dbg_best_norm,
                "anchor_word": (
                    _dbg_next_snap.word_index if _dbg_next_snap else None
                ),
                "anchor_type": (
                    _dbg_next_snap.anchor_type if _dbg_next_snap else None
                ),
                "snapped": _dbg_snapped,
                "forced": _dbg_forced,
                "alignment": _dbg_alignment,
            })

    if text_pointer < len(flat_text.words):
        flags.append(ValidationFlag(
            "line_count_mismatch",
            f"Allocated up to word {text_pointer}/{len(flat_text.words)}; "
            f"{len(flat_text.words) - text_pointer} word(s) unassigned",
        ))

    return AllocationResult(
        manifest=manifest,
        flags=flags,
        text_pointer_end=text_pointer,
        debug_lines=debug_lines_out,
        folio_start_line=folio_start_line,
        constituent_overrides=constituent_overrides,
    )


# ---------------------------------------------------------------------------
# Sub-plan 4c — Folio state persistence
# ---------------------------------------------------------------------------

@dataclass
class FolioState:
    """Carries post-77 continuation words and metadata across folio runs.

    When the last chant row on a folio has a 77 (page break) in its
    volpiano, the words after the 77 belong to the next folio but have
    no separate CSV row.  FolioState captures those words so the next
    folio run can prepend them via
    build_flat_text_and_anchors(prev_folio_state=...).

    Also captures the unconsumed flat_text tail when the NW allocator
    didn't reach the end (fewer detected lines than expected).
    """

    source_id: int | None
    folio: str
    last_chant_sequence: int    # sequence of last chant row on this folio
    remaining_words: list[str]  # words for next folio (post-77 or unconsumed)
    fully_consumed: bool        # True when remaining_words is empty


def build_folio_state(
    flat_text: FlatTextData,
    result: AllocationResult,
    source_id: int | None,
    folio: str,
) -> FolioState:
    """Build a FolioState from the allocation result for the current folio.

    Two sources for remaining_words (in priority order):
    1. flat_text.continuation_words — words after a 77 break in the
       last row; physically on the next folio, no separate CSV row.
    2. flat_text.words[result.text_pointer_end:] — words not consumed
       by the allocator (fewer lines detected than expected).

    Args:
        flat_text: FlatTextData returned by build_flat_text_and_anchors().
        result:    AllocationResult returned by allocate_lines().
        source_id: Cantus source ID (passed through for next folio run).
        folio:     Current folio string (for provenance).

    Returns:
        FolioState ready to be passed as prev_folio_state on next folio.
    """
    remaining = (
        flat_text.continuation_words
        or flat_text.words[result.text_pointer_end:]
    )
    last_span = next(
        (s for s in reversed(flat_text.chant_spans)
         if s.start_word <= result.text_pointer_end),
        flat_text.chant_spans[-1] if flat_text.chant_spans else None,
    )
    return FolioState(
        source_id=source_id,
        folio=folio,
        last_chant_sequence=last_span.sequence if last_span else 0,
        remaining_words=list(remaining),
        fully_consumed=(len(remaining) == 0),
    )


def write_folio_state(state: FolioState, path: str) -> None:
    """Serialise a FolioState to a JSON file."""
    import dataclasses
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(state), f, indent=2, ensure_ascii=False)


def read_folio_state(path: str) -> FolioState:
    """Deserialise FolioState from a JSON file by write_folio_state()."""
    import json

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return FolioState(**d)

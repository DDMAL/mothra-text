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

import re
from dataclasses import dataclass, field


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
class FlatTextData:
    words: list[str]
    anchors: list[Anchor]
    chant_spans: list[ChantSpan]
    initial_pointer: int = 0
    continuation_words: list[str] = field(default_factory=list)
    has_continuation: bool = False  # True if continuation was prepended


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


def _parse_row_words_and_anchors(
    text: str, volpiano: str
) -> tuple[list[str], list[tuple[int, str]], list[str]]:
    """Parse one chant row's text+volpiano into words and anchors.

    Also identifies post-77 continuation words for the next folio.

    Returns:
        (this_folio_words, raw_anchors, continuation_words)

        - this_folio_words: words on this folio (up to first 77, if any)
        - raw_anchors: [(word_offset, anchor_type), ...] where
          word_offset is the cumulative count AFTER the break
        - continuation_words: words after the first 77 break (belong
          to the next folio; no separate CSV row for these)
    """
    words = text.split() if text else []
    if not words:
        return [], [], []
    if not volpiano:
        return words, [], []

    # Split volpiano keeping breaks: ["seg0", "7+", "seg1", "7+", ...]
    parts = re.split(r"(7+)", volpiano)

    word_idx = 0
    raw_anchors: list[tuple[int, str]] = []
    continuation_start: int | None = None
    seg_count = 0  # number of volpiano segments processed so far

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
            word_idx += max(n, 0)
        seg_count += 1

        # Process the break that follows this segment (if any).
        if i < len(parts):
            break_str = parts[i]
            i += 1
            anchor_type = _classify_break(break_str)
            raw_anchors.append((word_idx, anchor_type))

            if anchor_type == "page_break_77" and continuation_start is None:
                continuation_start = word_idx

    # Determine which words stay on this folio vs continue to next.
    if continuation_start is not None:
        this_folio_words = words[:continuation_start]
        continuation_words = words[continuation_start:]
        # Drop anchors at or after the continuation boundary
        # (they describe structure on the next folio, not this one).
        raw_anchors = [
            (wi, at) for wi, at in raw_anchors
            if wi <= continuation_start
        ]
    else:
        this_folio_words = words
        continuation_words = []

    return this_folio_words, raw_anchors, continuation_words


# ---------------------------------------------------------------------------
# Public API — Sub-plan 4a
# ---------------------------------------------------------------------------

def build_flat_text_and_anchors(
    csv_rows: list[dict],
    folio: str,
    line_offset: int = 0,
    prev_folio_state: FolioState | None = None,
    infer_continuation: bool = True,
) -> FlatTextData:
    """Build flat word sequence and volpiano anchors for one folio.

    Args:
        csv_rows:           All rows from a Cantus CSV (any folio).
        folio:              Folio string to filter rows (e.g. "006r").
        line_offset:        Number of within_chant_7 breaks to skip
                            before alignment starts. Use when the image
                            crop begins partway into the folio.
        prev_folio_state:   FolioState from the previous folio run.
                            If provided, its remaining_words (post-77
                            continuation) are prepended to flat_text
                            before this folio's rows. Takes priority
                            over infer_continuation.
        infer_continuation: When True (default) and prev_folio_state
                            is None, scan csv_rows for the last row
                            from any preceding folio with a 77 break
                            and prepend its post-77 words. Handles the
                            common case where the previous folio was
                            not run first.

    Returns:
        FlatTextData with words, anchors, chant_spans, initial_pointer,
        continuation_words (post-77 words for the next folio), and
        has_continuation (True when continuation words were prepended).
    """
    from steps.gt_manifest import clean_text  # noqa: PLC0415

    words: list[str] = []
    anchors: list[Anchor] = []
    chant_spans: list[ChantSpan] = []
    continuation_words: list[str] = []
    has_continuation = False

    # Prepend continuation words carried from the previous folio's 77 break.
    if prev_folio_state is not None and prev_folio_state.remaining_words:
        words.extend(prev_folio_state.remaining_words)
        chant_spans.append(ChantSpan(
            sequence=0,
            start_word=0,
            end_word=len(words),
        ))
        has_continuation = True
    elif infer_continuation:
        # Scan all CSV rows for the last row from a preceding folio
        # with a 77 break — its post-77 words belong to this folio.
        target_key = _folio_sort_key(folio)
        prev_77_rows = [
            r for r in csv_rows
            if _folio_sort_key(r.get("folio", "")) < target_key
            and "77" in (r.get("volpiano") or "")
            and r.get("mode", "").strip() != "*"
        ]
        if prev_77_rows:
            prev_77_rows.sort(key=lambda r: (
                _folio_sort_key(r.get("folio", "")),
                int(r.get("sequence") or 0),
            ))
            carry_row = prev_77_rows[-1]
            raw_carry = (
                carry_row.get("fulltext_ms")
                or carry_row.get("fulltext_standardized")
                or ""
            ).strip()
            _, _, carry_words = _parse_row_words_and_anchors(
                clean_text(raw_carry),
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

    for row in folio_rows:
        raw_text = (
            row.get("fulltext_ms") or row.get("fulltext_standardized") or ""
        ).strip()
        text = clean_text(raw_text)
        if not text:
            continue

        volpiano = (row.get("volpiano") or "").strip()
        row_words, raw_anchors, row_continuation = (
            _parse_row_words_and_anchors(text, volpiano)
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

    # Advance initial_pointer past the first line_offset within_chant_7 breaks.
    initial_pointer = 0
    if line_offset > 0:
        within_anchors = [
            a for a in anchors if a.anchor_type == "within_chant_7"
        ]
        if line_offset <= len(within_anchors):
            initial_pointer = within_anchors[line_offset - 1].word_index

    return FlatTextData(
        words=words,
        anchors=anchors,
        chant_spans=chant_spans,
        initial_pointer=initial_pointer,
        continuation_words=continuation_words,
        has_continuation=has_continuation,
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


def allocate_lines(
    flat_text: FlatTextData,
    sorted_labels: list[str],
    ocr_texts: dict[str, str],
    column_count: int = 1,
    left_column_count: int = 0,
    search_window: int = 40,
    snap_window: int = 1,
    match_score: float = 8.0,
    mismatch_score: float = -5.0,
    open_penalty: float = -7.0,
    extend_penalty: float = -3.0,
    debug: bool = False,
) -> AllocationResult:
    """Align OCR text for each line against flat_text using NW alignment.

    For each label in sorted_labels, scores candidate word spans of
    lengths 1..search_window by normalising the NW alignment score by
    the geometric mean of the OCR string length and the candidate span
    length.  Optionally snaps the result to the nearest volpiano anchor
    within snap_window words.

    When model=None was used for KrakenRecognition, all OCR texts will
    be empty strings.  In stub mode each line is advanced to the next
    anchor position so the pipeline can complete end-to-end.

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
        match_score, mismatch_score, open_penalty, extend_penalty:
                            Bio.Align.PairwiseAligner scoring params.
                            Defaults are calibrated for medieval chant.
        debug:              When True, populate AllocationResult.
                            debug_lines with per-line OCR and NW
                            alignment detail.

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
            "consider providing --prev-folio-state "
            "if this folio starts mid-chant",
        ))

    for label_idx, label in enumerate(sorted_labels):
        # Hard-reset at column boundary (two-column folios only).
        if (
            column_count >= 2
            and left_column_count > 0
            and label_idx == left_column_count
        ):
            col_break = next(
                (a for a in sorted_anchors
                 if a.anchor_type == "column_break_777"
                 and a.word_index >= max(0, text_pointer - search_window)),
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

        candidate_words = flat_text.words[
            text_pointer: text_pointer + search_window
        ]
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
        _dbg_alignment = ""

        if not ocr:
            # Stub mode: advance to the next anchor of any type.
            next_anchor = next(
                (a for a in sorted_anchors if a.word_index > text_pointer),
                None,
            )
            consumed = (
                (next_anchor.word_index - text_pointer)
                if next_anchor else 1
            )
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
                 and a.anchor_type in ("within_chant_7", "page_break_77")),
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
                elif diff > snap_window:
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
        manifest[label] = " ".join(
            flat_text.words[text_pointer: text_pointer + consumed]
        )
        text_pointer += consumed

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

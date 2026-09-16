"""Extract sheet text and bind boundary calls to traced runs.

docling reads the page once -- to build a text-exclusion mask for tracing and
to surface leftover, unplaceable text -- and again per traced run, on a band
deskewed along that run's own direction (issue #8: crop + deskew is the
change that recovers rotated labels; a plain axis-aligned crop and full-page
OCR each under-read them on their own). A parsed call binds to the run
nearest its OCR box, gated by distance so an off-gate read -- a monument tag,
a curve-table cell, the title block -- lands in ``unbound_text`` and is never
force-bound.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from vectorjuju.tracing import Run

_REFERENCE_DPI = 200.0

# Physically derived from the synthetic fixture (synthetic_plat.py): a label
# sits FONT_SIZE=7.5pt tall, offset 5pt perpendicular from its line -- worst
# case an OCR box centre is ~8.7pt (~24px @200dpi) from the line. 60px leaves
# headroom for OCR box imprecision while staying far short of the curve
# table, which sits well outside any parcel edge's vicinity.
_BIND_RADIUS_PX = 60.0

# Carried from prototypes/diagonal_call_labels.py:_PUNCT. Order matters: NFKC
# folds the masculine ordinal indicator to a letter "o" and splits the double
# prime into two primes, so nothing after it can undo either (commit 4785f9d).
_PUNCT = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "′": "'",
    "″": '"',
    "º": "°",
    "–": "-",
    "—": "-",
}

# Quadrant is [NS]/[EW] character classes, never "[E W]" -- the space-capture
# bug from docs/prior-art/00-recon.md:229 that let a bearing match with a
# blank quadrant. Minutes and seconds are each optional (a whole-degree
# bearing like "N 45° E" must parse); ranges are validated after the match,
# not in the pattern. The trailing mark on each numeric group is optional so
# a dropped mark still parses; a present-but-wrong mark is flagged in
# parse_call rather than accepted silently.
_BEARING_RE = re.compile(
    r"([NS])\s*(\d{1,3})\s*(°)?\s*"
    r"(?:(\d{1,2})\s*(')?\s*"
    r"(?:(\d{1,2}(?:\.\d+)?)\s*(\")?\s*)?"
    r")?"
    r"([EW])"
)
# Distance is terminated by a lookahead (whitespace or end of string), not a
# \b after the foot mark -- docs/prior-art/00-recon.md:229 found \b backwards
# ("100.00'E" matched, "100.00'" did not). The mark itself tolerates the
# corrupted forms #7/#8 actually measured (foot misread as inch, or dropped
# to a stray asterisk); anything other than a clean "'" is a suspect token.
_DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(['\"*])?(?=\s|$)")
_CURVE_REF_RE = re.compile(r"^C(\d{1,3})$")


def normalize_ocr(raw: str) -> str:
    """Fold OCR punctuation confusables and collapse whitespace, upper-cased."""
    for k, v in _PUNCT.items():
        raw = raw.replace(k, v)
    raw = unicodedata.normalize("NFKC", raw)
    return re.sub(r"\s+", " ", raw).strip().upper()


@dataclass(frozen=True)
class ParsedCall:
    """One parsed boundary call: a bearing+distance line, or a bare curve reference.

    ``radius``/``arc_length`` are deliberately absent: those come from the
    fitted arc or curve table (issues #18, #21), never from a call's own OCR
    text -- see docs/prior-art/HANDOFF.md's arc-length-conflated-with-distance
    finding for why the two must stay separate fields, not a shared one.
    """

    raw_text: str
    kind: str  # "bearing_distance" | "curve_ref"
    bearing_deg: float | None  # azimuth, 0-360
    distance_ft: float | None
    curve_id: str | None
    suspect_tokens: tuple[str, ...]


def _azimuth(q1: str, q2: str, angle_deg: float) -> float:
    if q1 == "N":
        az = angle_deg if q2 == "E" else 360.0 - angle_deg
    else:
        az = 180.0 - angle_deg if q2 == "E" else 180.0 + angle_deg
    return az % 360.0


def parse_call(raw: str) -> ParsedCall | None:
    """Parse one OCR read into a bearing/distance call or a bare curve
    reference; ``None`` for anything else -- a non-call, a curve-table cell,
    distractor text. A complete call needs a bearing *and* a distance, or a
    bare ``C<n>``; that single rule rejects every curve-table cell for free
    (a radius cell has no bearing, a chord-bearing cell has no distance,
    ``PARCEL 5``/``LOT 12`` have neither) without special-casing any of them.
    """
    text = normalize_ocr(raw)

    curve = _CURVE_REF_RE.fullmatch(text)
    if curve:
        return ParsedCall(
            raw_text=raw,
            kind="curve_ref",
            bearing_deg=None,
            distance_ft=None,
            curve_id=f"C{curve.group(1)}",
            suspect_tokens=(),
        )

    bearing_match = _BEARING_RE.search(text)
    if not bearing_match:
        return None
    q1, deg, deg_mark, minutes, min_mark, seconds, sec_mark, q2 = bearing_match.groups()
    degrees = int(deg)
    minutes_val = int(minutes) if minutes else 0
    seconds_val = float(seconds) if seconds else 0.0
    if not (0 <= degrees <= 90 and 0 <= minutes_val < 60 and 0 <= seconds_val < 60):
        return None

    distance_match = _DISTANCE_RE.search(text, bearing_match.end())
    if not distance_match:
        return None

    angle = degrees + minutes_val / 60.0 + seconds_val / 3600.0
    suspects = []
    if deg_mark != "°":
        suspects.append("degree-mark")
    if minutes and min_mark != "'":
        suspects.append("minute-mark")
    if seconds and sec_mark != '"':
        suspects.append("second-mark")
    if distance_match.group(2) != "'":
        suspects.append("foot-mark")

    return ParsedCall(
        raw_text=raw,
        kind="bearing_distance",
        bearing_deg=_azimuth(q1, q2, angle),
        distance_ft=float(distance_match.group(1)),
        curve_id=None,
        suspect_tokens=tuple(suspects),
    )


@dataclass(frozen=True, eq=False)
class TextItem:
    """One OCR read: a page-pass full-sheet text, or a crop-pass read along one traced run."""

    text: str
    box_px: tuple[float, float, float, float]  # x0, y0, x1, y1 -- raster px, y down
    source: str  # "page" | "crop"


@dataclass(frozen=True, eq=False)
class BoundCall:
    """A parsed call bound to the run its text sits nearest to."""

    call: ParsedCall
    run: Run
    item: TextItem
    distance_px: float


def _scaled(px_at_reference_dpi: float, dpi: float) -> float:
    return px_at_reference_dpi * dpi / _REFERENCE_DPI


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


@lru_cache(maxsize=1)
def _converter():
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def _convert_path(path: Path):
    try:
        return _converter().convert(str(path)).document
    except Exception:  # noqa: BLE001 -- one unreadable page or crop must not kill the run
        return None


def _page_items(image: Image.Image) -> list[TextItem]:
    """Full-page OCR: the text-exclusion-mask source and the catch-all for
    leftover text (monuments, curve table, title block) that never binds."""
    page_h = image.size[1]
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        image.save(path, format="PNG")
        doc = _convert_path(path)
    if doc is None:
        return []
    items = []
    for t in doc.texts:
        if not t.prov or not t.text.strip():
            continue
        b = t.prov[0].bbox
        top, bottom = max(b.t, b.b), min(b.t, b.b)  # docling: bottom-left origin
        items.append(TextItem(text=t.text, box_px=(b.l, page_h - top, b.r, page_h - bottom), source="page"))
    return items


def warp_band(
    image: Image.Image,
    seg: tuple[float, float, float, float],
    center: tuple[float, float],
    half_len: float,
    half_h: float,
) -> Image.Image:
    """Deskew an upright band centred on ``center`` with the source x axis
    along ``seg``. No mirroring: the linear part is a rotation.

    Ported from prototypes/diagonal_call_labels.py -- issue #8's measured fix
    for rotated labels: plain axis-aligned crops and full-page OCR each
    under-read them; deskewing along the traced direction does not.
    """
    x0, y0, x1, y1 = seg
    seg_len = math.hypot(x1 - x0, y1 - y0)
    u = ((x1 - x0) / seg_len, (y1 - y0) / seg_len) if seg_len else (1.0, 0.0)
    n = (-u[1], u[0])
    w, h = round(2 * half_len), round(2 * half_h)
    src = [
        (center[0] - half_len * u[0] - half_h * n[0], center[1] - half_len * u[1] - half_h * n[1]),
        (center[0] + half_len * u[0] - half_h * n[0], center[1] + half_len * u[1] - half_h * n[1]),
        (center[0] - half_len * u[0] + half_h * n[0], center[1] - half_len * u[1] + half_h * n[1]),
    ]
    dst = [(0.0, 0.0), (float(w), 0.0), (0.0, float(h))]
    m = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    warped = cv2.warpAffine(np.asarray(image), m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(warped)


def _ocr_band(band: Image.Image) -> str:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "band.png"
        band.save(path, format="PNG")
        doc = _convert_path(path)
    if doc is None:
        return ""
    return " ".join(t.text for t in doc.texts)


def _crop_item(image: Image.Image, run: Run, dpi: float) -> TextItem | None:
    """OCR a deskewed band along ``run``; ``None`` if nothing on it parses as a call."""
    pts = run.points_px
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return None
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    # ponytail: band centred on the run's own midpoint, matching the
    # synthetic fixture's label placement; an off-centre real-plat label
    # needs a locator pass to recentre the band, not just a wider one.
    half_len = max(length * 0.4, _scaled(20.0, dpi))
    # A curve-ref label sits at the arc's bulge, not on the chord: measured
    # on the fixture's own C2 (radius 240ft over a 137ft chord), the offset
    # from the run's own endpoint-to-endpoint chord reaches ~51px @200dpi --
    # sagitta plus the label's own draw offset and text height. 70px covers
    # that with headroom for a straight call's much smaller offset too.
    half_h = _scaled(70.0, dpi)
    band = warp_band(image, (float(x0), float(y0), float(x1), float(y1)), center, half_len, half_h)

    text = _ocr_band(band)
    if parse_call(text) is None:
        # A traced run has no arrowhead; try the band's own 180-degree twin
        # and keep it only if it actually parses -- prefer a read that
        # succeeds over one with merely more characters (prototypes/
        # diagonal_call_labels.py's alnum-count tie-break is a measured
        # crutch: commit 50e54ed shows it flips on stray OCR periods).
        rotated_text = _ocr_band(band.rotate(180))
        if parse_call(rotated_text) is not None:
            text = rotated_text

    if parse_call(text) is None:
        return None
    return TextItem(text=text, box_px=_bbox(pts), source="crop")


def text_mask(items: Sequence[TextItem], shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask, true where OCR found text -- feeds ``trace_runs``'s
    ``exclude_mask`` so text ink is not mistaken for boundary linework."""
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for item in items:
        x0, y0, x1, y1 = item.box_px
        xi0, yi0 = max(0, int(x0)), max(0, int(y0))
        xi1, yi1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if xi1 > xi0 and yi1 > yi0:
            mask[yi0:yi1, xi0:xi1] = True
    return mask


def extract_text(image: Image.Image, runs: Sequence[Run], *, dpi: float = 200.0) -> list[TextItem]:
    """Page-pass items plus one crop-pass item per run, deskewed along that
    run's own direction. Every run is cropped, not just runs near a page-pass
    read: full-page OCR missed the short curve-ref labels (``C1``/``C2``) on
    every medium in issues #7 and #8, so a page-driven crop loop would too.
    """
    items = _page_items(image)
    for run in runs:
        item = _crop_item(image, run, dpi)
        if item is not None:
            items.append(item)
    return items


def _point_run_distance(point: tuple[float, float], run: Run) -> float:
    """Min distance from ``point`` to any segment of ``run``'s polyline."""
    px, py = point
    best = math.inf
    pts = run.points_px
    for (ax, ay), (bx, by) in pairwise(pts):
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
        qx, qy = ax + t * abx, ay + t * aby
        best = min(best, math.hypot(px - qx, py - qy))
    return best


def bind_calls(
    runs: Sequence[Run], items: Sequence[TextItem], *, dpi: float = 200.0
) -> tuple[list[BoundCall], list[TextItem]]:
    """Bind each parsed call to its nearest run within the gate radius.

    One call per run, one run per call: candidates are sorted by distance and
    assigned greedily by that order, so a run already claimed by a closer
    call cannot also take a farther one. Cheap at this scale (tens of runs
    and items); a real k-d tree is the upgrade docs/prior-art/
    20-implementation-plan.md names for a denser sheet.

    Everything that fails to parse as a call, or has no run within the gate,
    comes back in the second list -- never force-bound.
    """
    gate = _scaled(_BIND_RADIUS_PX, dpi)
    candidates: list[tuple[float, int, int, ParsedCall]] = []
    for item_index, item in enumerate(items):
        call = parse_call(item.text)
        if call is None:
            continue
        center = _center(item.box_px)
        for run_index, run in enumerate(runs):
            distance = _point_run_distance(center, run)
            if distance <= gate:
                candidates.append((distance, run_index, item_index, call))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    claimed_runs: set[int] = set()
    claimed_items: set[int] = set()
    bound: list[BoundCall] = []
    for distance, run_index, item_index, call in candidates:
        if run_index in claimed_runs or item_index in claimed_items:
            continue
        claimed_runs.add(run_index)
        claimed_items.add(item_index)
        bound.append(BoundCall(call=call, run=runs[run_index], item=items[item_index], distance_px=distance))

    unbound = [item for index, item in enumerate(items) if index not in claimed_items]
    return bound, unbound

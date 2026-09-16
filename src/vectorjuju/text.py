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

import logging
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
from scipy.optimize import linear_sum_assignment

from vectorjuju.tracing import Run

_logger = logging.getLogger(__name__)

_REFERENCE_DPI = 200.0

# Physically derived from the synthetic fixture (synthetic_plat.py): a label
# sits FONT_SIZE=7.5pt tall, offset 5pt perpendicular from its line -- worst
# case an OCR box centre is ~8.7pt (~24px @200dpi) from the line. 60px leaves
# headroom for OCR box imprecision while staying far short of the curve
# table, which sits well outside any parcel edge's vicinity.
_BIND_RADIUS_PX = 60.0

# Bounds on extract_text()'s crop-pass fan-out and per-crop allocation.
# _MAX_CROP_SIDE_PX caps warpAffine()'s output buffer regardless of a run's
# span or an oversized dpi; _MAX_RUNS_FOR_TEXT caps total OCR conversions
# (1-2 per run) a call to extract_text() will start. ponytail: flat caps, not
# a budget derived from image size or a timeout -- a sheet that legitimately
# needs more than this needs batched/async OCR, not a bigger constant.
_MAX_CROP_SIDE_PX = 4000.0
_MAX_RUNS_FOR_TEXT = 4000

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
# fullmatch, not a tolerant search: a curve ref must be the read's entire
# text, the same reject-rather-than-guess stance the bearing parser takes on
# a corrupted mark. A ref merged with adjacent OCR text (a stray monument
# tag, table-row noise) fails safe into unbound_text rather than guessing
# which part of the string is the real reference.
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

    ``ParsedCall.raw_text`` is the call as it should be transcribed: unit
    marks the parser positioned itself are written in their canonical form
    (a ``'`` misread as ``"`` or ``*`` becomes ``'``), and OCR noise glued
    onto the call's boundary -- tolerated above so a real read is not thrown
    away -- is not carried along. Each correction is recorded in
    ``suspect_tokens``; the untouched read stays on the ``TextItem``.
    """
    text = normalize_ocr(raw)

    curve = _CURVE_REF_RE.fullmatch(text)
    if curve:
        return ParsedCall(
            raw_text=f"C{curve.group(1)}",
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
    angle = degrees + minutes_val / 60.0 + seconds_val / 3600.0
    # A quadrant bearing tops out at exactly 90 degrees; validating each
    # field in isolation (degrees<=90, minutes<60, ...) lets 90deg30' through
    # even though the combined angle is past the quadrant boundary.
    if not (0 <= minutes_val < 60 and 0 <= seconds_val < 60 and angle <= 90):
        return None

    # The distance must immediately follow the bearing (only whitespace
    # between): .search() would otherwise be free to skip a glued/garbled
    # distance -- the exact shape docs/prior-art/00-recon.md:229's \b bug
    # left behind -- and match a stray digit further into the string
    # instead (e.g. the "1" in a neighbouring "C1").
    after_bearing = text[bearing_match.end() :]
    gap = len(after_bearing) - len(after_bearing.lstrip())
    distance_match = _DISTANCE_RE.match(text, bearing_match.end() + gap)
    if not distance_match:
        return None

    # Reject a call glued to a separate, whitespace-delimited word -- an
    # annotation like "NOTE: N 45° E 100' TYPICAL" -- but tolerate noise
    # merged directly onto the call with no gap at all (measured: OCR
    # occasionally misreads a stray mark right before the quadrant letter as
    # an extra character, e.g. "WN 35°09'59\" E 107.65'"). Whitespace
    # adjacent to the boundary is what marks a residual as its own token;
    # normalize_ocr already strips the read's own outer whitespace, so a
    # non-empty residual with no adjacent space is glued-on noise, not text.
    before, after = text[: bearing_match.start()], text[distance_match.end() :]
    if (before and before[-1].isspace()) or (after and after[0].isspace()):
        return None

    suspects = []
    if deg_mark != "°":
        suspects.append("degree-mark")
    if minutes and min_mark != "'":
        suspects.append("minute-mark")
    if seconds and sec_mark != '"':
        suspects.append("second-mark")
    if distance_match.group(2) != "'":
        suspects.append("foot-mark")

    canonical = f"{q1} {deg}°"
    if minutes is not None:
        canonical += f"{minutes}'"
    if seconds is not None:
        canonical += f'{seconds}"'
    canonical += f" {q2} {distance_match.group(1)}'"

    return ParsedCall(
        raw_text=canonical,
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
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrAutoOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption

    # scale=1.0: OCR inputs are already pre-scaled by _ocr_image, so docling
    # must not resample them a second time; it still resolves the engine the
    # same way (ocrmac on macOS, RapidOCR elsewhere).
    options = PdfPipelineOptions(ocr_options=OcrAutoOptions(scale=1.0))
    return DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)})


def _convert_path(path: Path):
    # One unreadable page or crop must not kill the run, but the failure must
    # stay observable: logged here rather than silently folded into "no text
    # found" (an unavailable model asset or OCR init error looks identical to
    # a blank page otherwise).
    try:
        return _converter().convert(str(path)).document
    except Exception:
        _logger.warning("docling conversion failed for %s", path, exc_info=True)
        return None


# OCR input scale. The fixture's 7.5 pt labels are ~21 px tall at the
# reference 200 dpi, and both engines misread them at native resolution
# (RapidOCR dropped a decimal point or a digit on Linux; ocrmac's reads are
# stable but lossy), and at 2x RapidOCR still misses the short curve-ref
# labels. Handing the engines a LANCZOS 3x pre-scale instead of docling's own
# default 3.0x resample keeps the reads correct on both, and pixel-wise it is
# no more than the default it replaces. Capped so a large sheet cannot blow
# up the OCR buffer; a page already over the cap is scaled less.
_OCR_UPSCALE = 3.0
_OCR_MAX_PIXELS = 40_000_000


def _ocr_image(image: Image.Image) -> tuple[Image.Image, float]:
    """The image handed to OCR, and the factor its pixels were scaled by."""
    factor = min(_OCR_UPSCALE, math.sqrt(_OCR_MAX_PIXELS / (image.width * image.height)))
    if factor <= 1.0:
        return image, 1.0
    return image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS), factor


def _doc_items(doc, height: int, source: str, factor: float = 1.0) -> list[TextItem]:
    """One TextItem per docling text region, boxes back in the caller's pixels.

    ``height`` is the OCR image's own height and ``factor`` the scale
    ``_ocr_image`` applied, so boxes come back in the image the caller has.
    """
    items = []
    for t in doc.texts:
        if not t.prov or not t.text.strip():
            continue
        b = t.prov[0].bbox
        top, bottom = max(b.t, b.b), min(b.t, b.b)  # docling: bottom-left origin
        items.append(
            TextItem(
                text=t.text,
                box_px=(b.l / factor, (height - top) / factor, b.r / factor, (height - bottom) / factor),
                source=source,
            )
        )
    return items


def page_items(image: Image.Image) -> list[TextItem]:
    """Full-page OCR: the text-exclusion-mask source and the catch-all for
    leftover text (monuments, curve table, title block) that never binds.

    Pass the result to both ``text_mask()`` (before tracing) and
    ``extract_text(..., page_items=...)`` (after) to run this page-pass once
    instead of once per call.
    """
    ocr, factor = _ocr_image(image)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        ocr.save(path, format="PNG")
        doc = _convert_path(path)
    if doc is None:
        return []
    return _doc_items(doc, ocr.size[1], "page", factor)


_page_items = page_items  # internal alias: extract_text's page_items kwarg shadows the module-level name


def warp_band(
    image: Image.Image,
    seg: tuple[float, float, float, float],
    center: tuple[float, float],
    half_len: float,
    half_h: float,
) -> tuple[Image.Image, np.ndarray]:
    """Deskew an upright band centred on ``center`` with the source x axis
    along ``seg``. No mirroring: the linear part is a rotation.

    Returns the band and the affine matrix ``m`` mapping ``image`` pixel
    coordinates to band pixel coordinates (``cv2.invertAffineTransform(m)``
    maps back), so a caller can place an OCR box found in the band back onto
    the source image.

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
    return Image.fromarray(warped), m


def _ocr_band(band: Image.Image) -> tuple[str, tuple[float, float, float, float] | None]:
    """OCR one band; returns its text and the band-local bbox spanning every
    OCR'd text region that contributed to it (``None`` if none did)."""
    ocr, factor = _ocr_image(band)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "band.png"
        ocr.save(path, format="PNG")
        doc = _convert_path(path)
    if doc is None:
        return "", None
    items = _doc_items(doc, ocr.size[1], "crop", factor)
    text = " ".join(item.text for item in items)
    if not items:
        return text, None
    x0 = min(item.box_px[0] for item in items)
    y0 = min(item.box_px[1] for item in items)
    x1 = max(item.box_px[2] for item in items)
    y1 = max(item.box_px[3] for item in items)
    return text, (x0, y0, x1, y1)


def _band_box_to_image(
    box: tuple[float, float, float, float],
    band_size: tuple[int, int],
    m: np.ndarray,
    *,
    rotated: bool,
) -> tuple[float, float, float, float]:
    """Map a band-local OCR box back onto the source image ``warp_band`` cropped
    it from, undoing the 180-degree retry rotation (if any) and the affine."""
    x0, y0, x1, y1 = box
    if rotated:
        w, h = band_size
        x0, x1 = w - x1, w - x0
        y0, y1 = h - y1, h - y0
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    m_inv = cv2.invertAffineTransform(m.astype(np.float64))
    mapped = corners @ m_inv[:, :2].T + m_inv[:, 2]
    return _bbox(mapped)


def _crop_item(image: Image.Image, run: Run, dpi: float) -> TextItem | None:
    """OCR a deskewed band along ``run``; ``None`` if nothing on it parses as a call."""
    pts = run.points_px
    if len(pts) < 2:
        return None  # trace_runs never emits this, but nothing here should assume it
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return None
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    # ponytail: band centred on the run's own midpoint, matching the
    # synthetic fixture's label placement; an off-centre real-plat label
    # needs a locator pass to recentre the band, not just a wider one.
    half_len = min(max(length * 0.4, _scaled(20.0, dpi)), _MAX_CROP_SIDE_PX / 2)
    # ponytail: a curve-ref label sits at the arc's bulge, not on the chord,
    # so its offset from the run's own endpoint-to-endpoint chord grows with
    # the curve's sagitta -- unboundedly as radius approaches chord/2. 110px
    # covers the fixture's own C2 (radius 240ft/chord 137ft, offset ~51px)
    # with headroom down to roughly a 100ft-radius curve on a similar chord;
    # a materially tighter curve on a long chord can still clip. Once curve
    # classification lands (issue #18), size this from the run's actual
    # fitted geometry instead of a flat constant.
    half_h = min(_scaled(110.0, dpi), _MAX_CROP_SIDE_PX / 2)
    band, m = warp_band(image, (float(x0), float(y0), float(x1), float(y1)), center, half_len, half_h)

    text, box_local = _ocr_band(band)
    rotated = False
    if parse_call(text) is None:
        # A traced run has no arrowhead; try the band's own 180-degree twin
        # and keep it only if it actually parses -- prefer a read that
        # succeeds over one with merely more characters (prototypes/
        # diagonal_call_labels.py's alnum-count tie-break is a measured
        # crutch: commit 50e54ed shows it flips on stray OCR periods).
        rotated_text, rotated_box_local = _ocr_band(band.rotate(180))
        if parse_call(rotated_text) is not None:
            text, box_local, rotated = rotated_text, rotated_box_local, True

    if parse_call(text) is None:
        return None
    # box_local is only ever None when text is "" (no OCR'd region
    # contributed to it), and an empty text never survives parse_call above.
    box_px = _band_box_to_image(box_local, band.size, m, rotated=rotated)
    return TextItem(text=text, box_px=box_px, source="crop")


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


def extract_text(
    image: Image.Image,
    runs: Sequence[Run],
    *,
    dpi: float = 200.0,
    page_items: Sequence[TextItem] | None = None,
) -> list[TextItem]:
    """Page-pass items plus one crop-pass item per run, deskewed along that
    run's own direction. Every run is cropped, not just runs near a page-pass
    read: full-page OCR missed the short curve-ref labels (``C1``/``C2``) on
    every medium in issues #7 and #8, so a page-driven crop loop would too.

    ``page_items`` reuses an already-computed page pass -- e.g. the one a
    caller ran to build ``text_mask()``'s exclusion mask before tracing --
    instead of running docling over the full page a second time. Pass
    ``vectorjuju.text.page_items(image)``'s own result straight through.
    """
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"extract_text requires a finite dpi > 0, got {dpi!r}")
    if len(runs) > _MAX_RUNS_FOR_TEXT:
        raise ValueError(f"extract_text requires len(runs) <= {_MAX_RUNS_FOR_TEXT}, got {len(runs)}")
    items = list(page_items) if page_items is not None else _page_items(image)
    for run in runs:
        item = _crop_item(image, run, dpi)
        if item is not None:
            items.append(item)
    return items


def _point_run_distance(point: tuple[float, float], run: Run) -> float:
    """Min distance from ``point`` to any segment of ``run``'s polyline.

    ``run.points_px`` needs at least 2 points to have a segment at all --
    every ``Run`` `trace_runs` emits does. A shorter run returns
    ``math.inf``, which safely excludes it from binding rather than raising.
    """
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


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """True when two OCR boxes share area -- the same physical text read twice."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def bind_calls(
    runs: Sequence[Run], items: Sequence[TextItem], *, dpi: float = 200.0
) -> tuple[list[BoundCall], list[TextItem]]:
    """Bind each parsed call to a run within the gate radius, minimising total
    bind distance over the whole sheet at once -- not nearest-first greedy,
    which lets one call steal a run a farther call needed more (docs/
    prior-art/20-implementation-plan.md's "assignment, not greedy" binding
    requirement). One call per run, one run per call. Duplicate reads of one
    call -- the page pass and a band crop both reading the same label -- are
    collapsed first (same parsed call, overlapping boxes), so a call cannot
    claim two runs.

    Everything that fails to parse as a call, or has no run within the gate,
    comes back in the second list -- never force-bound.
    """
    gate = _scaled(_BIND_RADIUS_PX, dpi)
    callable_items = [(index, call) for index, item in enumerate(items) if (call := parse_call(item.text))]
    if not callable_items or not runs:
        return [], list(items)

    # One physical label is routinely read twice -- the full-page pass and the
    # deskewed band crop that exists to recover rotated text -- and binding
    # both reads lets one call claim two runs, stranding a different, real
    # call's read (or reporting one call twice). Collapse reads of the same
    # parsed call whose boxes overlap; the first read wins (page items precede
    # crop items), and a collapsed duplicate is not a second call, so it stays
    # out of ``unbound_text`` too. Every earlier read stays a peer, merged ones
    # included, so a chain of mutually overlapping reads (page -> crop -> crop)
    # collapses whole: a read overlapping only a merged duplicate is still the
    # same physical label, and the first read remains the representative.
    distinct: list[tuple[int, ParsedCall]] = []
    boxes_by_call: dict[tuple[object, ...], list[int]] = {}
    merged_item_indices: set[int] = set()
    for item_index, call in callable_items:
        key = (call.kind, call.bearing_deg, call.distance_ft, call.curve_id)
        peers = boxes_by_call.setdefault(key, [])
        duplicate = any(_boxes_overlap(items[item_index].box_px, items[peer].box_px) for peer in peers)
        peers.append(item_index)
        if duplicate:
            merged_item_indices.add(item_index)
            continue
        distinct.append((item_index, call))
    callable_items = distinct

    distances = np.array(
        [
            [_point_run_distance(_center(items[item_index].box_px), run) for run in runs]
            for item_index, _ in callable_items
        ]
    )
    # A penalty cell must cost more than a whole fully in-gate assignment can
    # total -- at most min(rows, cols) cells, each at most `gate` -- not
    # merely more than the largest pairwise distance. Otherwise the solver can
    # buy cheap rows elsewhere with one penalty cell, beat a complete valid
    # matching on total cost, and the gate check below then drops that row even
    # though a real match existed. The `finite` fallback keeps an all-infinite
    # matrix (no finite candidate anywhere) from raising on an empty max.
    finite = distances[np.isfinite(distances)]
    largest = float(finite.max()) if finite.size else 0.0
    penalty = max(largest, gate) * min(distances.shape) + 1.0
    cost = np.where(distances <= gate, distances, penalty)
    row_index, col_index = linear_sum_assignment(cost)

    bound: list[BoundCall] = []
    bound_item_indices: set[int] = set()
    for row, col in zip(row_index, col_index, strict=True):
        distance = float(distances[row, col])
        if distance > gate:
            continue  # every candidate run for this item was out of gate
        item_index, call = callable_items[row]
        bound.append(BoundCall(call=call, run=runs[col], item=items[item_index], distance_px=distance))
        bound_item_indices.add(item_index)

    # row_index is sorted ascending (scipy guarantees this), and rows follow
    # callable_items' -- hence items' -- own order, so this needs no extra
    # sort to stay deterministic (A8).
    unbound = [
        item for index, item in enumerate(items) if index not in bound_item_indices and index not in merged_item_indices
    ]
    return bound, unbound

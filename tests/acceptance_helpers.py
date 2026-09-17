"""Helpers shared by the run-level gates over the synthetic plat.

The full acceptance suite (issue #22) builds its geometry comparisons on
these; keep them free of tracing and DXF concerns. ``write_overlay`` is the
one exception: it reads the written DXF (the way a CAD consumer does) to draw
the artifact a human reviews.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ezdxf.recover import readfile
from PIL import Image, ImageDraw

from vectorjuju.curves import CircleFit
from vectorjuju.pipeline import cad_to_px
from vectorjuju.synthetic_plat import SCALE_PT_PER_FT

# Overlay colours: the DXF's own layers, drawn over the input raster.
LINE_COLOUR = (0, 0, 220)
CURVE_COLOUR = (200, 0, 0)
LABEL_COLOUR = (0, 140, 0)

# Carried from prototypes/diagonal_call_labels.py: OCR folds these before any
# comparison, in this order (NFKC splits a double prime into two primes).
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
_MARKS = "°'\""


def pt_to_px(pt: tuple[float, float], *, dpi: float = 200.0, img_height: int) -> tuple[float, float]:
    """Page points (PDF, bottom-left origin) -> raster px (top-left origin)."""
    x, y = pt
    return x * dpi / 72.0, img_height - y * dpi / 72.0


def planted_fpp(dpi: float = 200.0) -> float:
    """The fixture's feet-per-pixel: one foot is ``SCALE_PT_PER_FT`` page points."""
    return 72.0 / (SCALE_PT_PER_FT * dpi)


def planted_fpp_m(dpi: float = 200.0) -> float:
    """The same calibration in metres per pixel, the metre run's oracle."""
    return planted_fpp(dpi) * 1200.0 / 3937.0


def point_segment_distance(points: np.ndarray, start: Iterable[float], end: Iterable[float]) -> np.ndarray:
    """Per-point distance from ``points`` to segment ``start``-``end``."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    edge = b - a
    length_sq = float(edge @ edge)
    if length_sq == 0:
        return np.hypot(*(np.asarray(points, float) - a).T)
    t = np.clip(((np.asarray(points, float) - a) @ edge) / length_sq, 0.0, 1.0)
    nearest = a + t[:, None] * edge
    return np.hypot(*(np.asarray(points, float) - nearest).T)


def quad_distance(point: Iterable[float], quad: Iterable[Iterable[float]]) -> float:
    """Distance from ``point`` to a planted text quad; zero inside the quad."""
    corners = [np.asarray(corner, float) for corner in quad]
    inside = False
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1], strict=True):
        if (y0 > point[1]) != (y1 > point[1]) and point[0] < (x1 - x0) * (point[1] - y0) / (y1 - y0) + x0:
            inside = not inside
    if inside:
        return 0.0
    return min(
        point_segment_distance(np.array([point]), a, b)[0]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True)
    )


def normalize_text(text: str) -> str:
    for old, new in _PUNCT.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().upper()


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", text)


def _mark_places(text: str) -> list[tuple[int, str]]:
    """Each mark with the alphanumerics before it: its place in the text, so a
    mark read against the wrong number cannot hide behind an equal count."""
    places: list[tuple[int, str]] = []
    alnum = 0
    for char in text:
        if char in _MARKS:
            places.append((alnum, char))
        elif char.isalnum():
            alnum += 1
    return places


def _marks_in_place(got: str, want: str) -> bool:
    """Every mark got is read where want has it: marks may be lost, but never
    invented, duplicated, or swapped between numbers."""
    remaining = iter(_mark_places(want))
    return all(place in remaining for place in _mark_places(got))


def label_passes(recovered: str, truth: str) -> bool:
    """The suite's label-fidelity rule, criteria from the issue-13 prototype.

    ``exact`` and ``normalized`` pass outright. Otherwise the read must be
    ``text_only`` -- every digit and letter right, decimals right -- *and* gain
    no unit mark: the read may have lost marks but not invented, duplicated,
    or moved one, so ``N 12'34°56" E`` for truth ``N 12°34'56" E`` fails on
    the swapped degree and minute. A ``200.16'`` read as ``200.16"`` is 12x
    out and fails, which is exactly what the bare ``text_only`` verdict would
    wave through.
    """
    got, want = normalize_text(recovered), normalize_text(truth)
    if not recovered.strip():
        return False
    if got == want:
        return True
    if _alnum(got) != _alnum(want) or got.count(".") != want.count("."):
        return False
    return _marks_in_place(got, want)


def to_px_points(points: Iterable[Iterable[float]], *, fpp: float, img_height: int) -> np.ndarray:
    """Sidecar/DXF CAD points -> raster px, the inverse of the writer's transform."""
    return np.array([cad_to_px((float(x), float(y)), fpp=fpp, img_height=img_height) for x, y in points])


def arc_points(
    start: float, end: float, centre: tuple[float, float], radius: float, count: int = 64
) -> list[tuple[float, float]]:
    """Sample a circular sweeping arc from ``start`` to ``end`` degrees (CCW)."""
    sweep = (end - start) % 360.0
    return [
        (
            centre[0] + radius * math.cos(math.radians(angle)),
            centre[1] + radius * math.sin(math.radians(angle)),
        )
        for angle in (start + sweep * i / (count - 1) for i in range(count))
    ]


def write_overlay(page: Image.Image, dxf: str | Path, sidecar: dict, out: str | Path) -> Path:
    """Draw the input raster with the written DXF geometry on top.

    One artifact per end-to-end run, for human review: boundary lines blue,
    true arcs (as the DXF stores them, not the traced polyline) red, and every
    label insertion a small cross. Pillow only -- the DXF is *read* through
    ezdxf the way the mechanical gates read it, and drawn with ``ImageDraw``;
    no drawing add-on, no matplotlib, no golden-image comparison.
    """
    out = Path(out)
    fpp = float(sidecar["scale"]["value"])
    image = page.convert("RGB")
    painter = ImageDraw.Draw(image)
    doc, _auditor = readfile(dxf)
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind == "LWPOLYLINE":
            cad = [point[:2] for point in entity.get_points("xy")]
            if entity.closed:
                cad.append(cad[0])
            points = to_px_points(cad, fpp=fpp, img_height=image.size[1])
            painter.line([tuple(point) for point in points], fill=LINE_COLOUR, width=2)
        elif kind == "ARC":
            cad = arc_points(
                entity.dxf.start_angle,
                entity.dxf.end_angle,
                (entity.dxf.center[0], entity.dxf.center[1]),
                entity.dxf.radius,
            )
            points = to_px_points(cad, fpp=fpp, img_height=image.size[1])
            painter.line([tuple(point) for point in points], fill=CURVE_COLOUR, width=2)
        elif kind == "SPLINE":
            points = to_px_points([point[:2] for point in entity.fit_points], fpp=fpp, img_height=image.size[1])
            painter.line([tuple(point) for point in points], fill=CURVE_COLOUR, width=2)
        elif kind == "MTEXT":
            x, y = cad_to_px((entity.dxf.insert[0], entity.dxf.insert[1]), fpp=fpp, img_height=image.size[1])
            painter.line([(x - 5, y), (x + 5, y)], fill=LABEL_COLOUR, width=2)
            painter.line([(x, y - 5), (x, y + 5)], fill=LABEL_COLOUR, width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format="PNG")
    return out


def arc_midpoint(points: np.ndarray, fit: CircleFit) -> np.ndarray:
    """Point halfway along the fitted sweep from the run's start to its end."""
    centre = np.asarray(fit.center_px)
    start = math.atan2(points[0][1] - centre[1], points[0][0] - centre[0])
    end = math.atan2(points[-1][1] - centre[1], points[-1][0] - centre[0])
    if fit.clockwise:  # increasing raster angle turns clockwise on the page
        while end <= start:
            end += 2 * math.pi
    else:
        while end >= start:
            end -= 2 * math.pi
    angle = (start + end) / 2
    return centre + fit.radius_px * np.array([math.cos(angle), math.sin(angle)])

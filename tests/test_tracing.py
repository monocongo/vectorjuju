"""Unit gates for tracing: the straight half of U3, on rasters drawn in the test.

Nothing here needs OCR or the sheet fixture; each test draws the one property it
pins, so a failure names the property rather than a fixture drift.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vectorjuju.tracing import trace_runs

DPI = 200.0


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (width, height), 255)
    return image, ImageDraw.Draw(image)


def test_thick_stroke_traces_one_centreline():
    image, draw = _canvas(120, 60)
    draw.line((10, 30, 110, 30), fill=0, width=4)

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    points = runs[0].points_px
    assert np.all(np.abs(points[:, 1] - 30.0) <= 1.5)  # one centreline, not two edges
    assert np.ptp(points[:, 0]) >= 95


def test_l_corner_splits_into_two_straight_runs():
    image, draw = _canvas(120, 120)
    draw.line((30, 20, 30, 90), fill=0, width=5)
    draw.line((30, 90, 100, 90), fill=0, width=5)

    runs = trace_runs(image, DPI)

    assert len(runs) == 2
    for run in runs:
        points = run.points_px
        assert np.ptp(points[:, 0]) < 3 or np.ptp(points[:, 1]) < 3


def test_glyphs_yield_no_primitives():
    image, draw = _canvas(160, 60)
    draw.text((5, 5), "ABC 12", font=ImageFont.load_default(size=20), fill=0)

    assert trace_runs(image, DPI) == []


def test_dashed_line_merges_into_one_run():
    image, draw = _canvas(220, 40)
    for k in range(10):
        x = 10 + k * 20
        draw.line((x, 20, x + 15, 20), fill=0, width=4)

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    assert np.ptp(runs[0].points_px[:, 0]) >= 180  # spans the gaps


def test_run_count_is_dpi_invariant():
    def square(dpi: float) -> Image.Image:
        scale = dpi / DPI
        image, draw = _canvas(round(200 * scale), round(200 * scale))
        margin = round(40 * scale)
        draw.rectangle(
            (margin, margin, image.width - margin, image.height - margin),
            outline=0,
            width=max(2, round(4 * scale)),
        )
        return image

    assert len(trace_runs(square(DPI), DPI)) == 4
    assert len(trace_runs(square(DPI / 2), DPI / 2)) == 4

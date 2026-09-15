"""Unit gates for tracing: the straight half of U3, on rasters drawn in the test.

Nothing here needs OCR or the sheet fixture; each test draws the one property it
pins, so a failure names the property rather than a fixture drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from vectorjuju.tracing import trace_runs

DPI = 200.0


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (width, height), 255)
    return image, ImageDraw.Draw(image)


@pytest.mark.parametrize("width", [3, 4])
def test_thick_stroke_traces_one_centreline(width: int):
    image, draw = _canvas(120, 60)
    draw.line((10, 30, 110, 30), fill=0, width=width)

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
    corner = np.array([30.0, 90.0])
    for run in runs:
        points = run.points_px
        assert np.ptp(points[:, 0]) < 3 or np.ptp(points[:, 1]) < 3
        assert min(np.hypot(*(points[0] - corner)), np.hypot(*(points[-1] - corner))) <= 4.0
        assert np.sum(np.hypot(*np.diff(points, axis=0).T)) >= 60.0  # not truncated at the corner


def test_small_arc_stays_one_run():
    radius, scale = 40, 8
    size = 2 * radius + 40
    big = Image.new("L", (size * scale, size * scale), 255)
    ImageDraw.Draw(big).arc(
        (
            (size / 2 - radius) * scale,
            (size / 2 - radius) * scale,
            (size / 2 + radius) * scale,
            (size / 2 + radius) * scale,
        ),
        0,
        270,
        fill=0,
        width=4 * scale,
    )
    image = big.resize((size, size), Image.LANCZOS)

    assert len(trace_runs(image, DPI)) == 1


def test_short_thick_run_survives():
    image, draw = _canvas(80, 80)
    draw.line((40, 25, 40, 50), fill=0, width=5)  # 25 px long, 5 px wide

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    assert np.ptp(runs[0].points_px[:, 1]) >= 18


def test_glyphs_yield_no_primitives():
    image, draw = _canvas(160, 60)
    draw.text((5, 5), "ABC 12", font=ImageFont.load_default(size=20), fill=0)

    assert trace_runs(image, DPI) == []


def test_dashed_line_merges_into_one_run():
    image, draw = _canvas(240, 40)
    for k in range(10):
        x = 10 + k * 20
        draw.line((x, 20, x + 10, 20), fill=0, width=4)

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    assert np.ptp(runs[0].points_px[:, 0]) >= 170  # spans the gaps


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

    for dpi in (DPI / 2, DPI, 2 * DPI):
        assert len(trace_runs(square(dpi), dpi)) == 4


def test_exclude_mask_drops_masked_stroke():
    image, draw = _canvas(240, 60)
    draw.line((10, 15, 230, 15), fill=0, width=5)
    draw.line((10, 45, 230, 45), fill=0, width=5)
    mask = np.zeros((60, 240), bool)
    mask[35:56, :] = True

    runs = trace_runs(image, DPI, exclude_mask=mask)

    assert len(runs) == 1
    assert np.median(runs[0].points_px[:, 1]) < 30  # the surviving line, not the masked one


def test_invalid_inputs_are_rejected():
    image, draw = _canvas(40, 40)
    draw.line((5, 20, 35, 20), fill=0, width=4)

    with pytest.raises(ValueError):
        trace_runs(image, dpi=0)
    with pytest.raises(ValueError):
        trace_runs(np.zeros((40, 40, 3), np.uint8), DPI)
    with pytest.raises(ValueError):
        trace_runs(np.zeros((40, 40), np.float64), DPI)
    with pytest.raises(ValueError):
        trace_runs(image, DPI, exclude_mask=np.zeros((1, 1), bool))

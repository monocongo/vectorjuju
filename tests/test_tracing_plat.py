"""A7 at the run level, on the synthetic plat fixture.

The full acceptance suite reads A7 off a DXF; until `convert()` emits one (issue
#22), the same properties are asserted over traced runs in raster px: one run
per planted straight edge, reaching its endpoints, with no second parallel run
and nothing on the right-of-way distractors.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectorjuju.synthetic_plat import PAGE_H, RENDER_DPI, generate_sheet
from vectorjuju.tracing import Run, trace_runs


@pytest.fixture(scope="module")
def plat(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Run], dict]:
    out = tmp_path_factory.mktemp("plat")
    truth = generate_sheet(out)
    return trace_runs(Image.open(Path(out) / "sheet.tif"), RENDER_DPI), truth


def to_px(point: list[float]) -> np.ndarray:
    scale = RENDER_DPI / 72.0
    return np.array([point[0] * scale, (PAGE_H - point[1]) * scale])


def densify(points: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Two-point runs still need interior samples for the distance gates."""
    chunks = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step)), endpoint=False) for a, b in pairwise(points)]
    return np.vstack([*chunks, points[-1:]])


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.hypot(*(points - (a + t[:, None] * ab)).T)


def edge_hits(
    runs: list[Run], a: np.ndarray, b: np.ndarray, *, tolerance: float = 3.0, endpoint_px: float = 8.0
) -> list[tuple[np.ndarray, bool]]:
    """Runs substantially inside the tolerance band of ab, as (points, entire interior inside).

    A run counts as a hit once half its interior samples are in the band, so a
    second entity that only partly parallels the edge is still counted; the
    boolean reports whether that run stays inside for its whole interior.
    """
    hits = []
    for run in runs:
        points = densify(run.points_px)
        distance = point_segment_distance(points, a, b)
        to_endpoint = np.minimum(np.hypot(*(points - a).T), np.hypot(*(points - b).T))
        interior = to_endpoint > endpoint_px  # the 8 px corner-splitting slop
        if interior.sum() < 3:
            continue
        in_band = distance[interior] <= tolerance
        if in_band.mean() >= 0.5:
            hits.append((points, bool(in_band.all())))
    return hits


def test_exactly_one_run_per_straight_edge(plat: tuple[list[Run], dict]):
    runs, truth = plat
    straight = [segment for segment in truth["segments"] if segment["kind"] == "straight"]
    assert len(straight) == 4  # the fixture plants four straight edges

    for segment in straight:
        a, b = to_px(segment["start_pt"]), to_px(segment["end_pt"])
        hits = edge_hits(runs, a, b)
        assert len(hits) == 1, f"{segment['id']}: {len(hits)} runs within 3 px"
        points, inside = hits[0]
        assert inside, f"{segment['id']}: run leaves the 3 px band"
        for end in (points[0], points[-1]):
            assert min(np.hypot(*(end - a)), np.hypot(*(end - b))) <= 8.0


def test_no_primitive_on_offset_lines(plat: tuple[list[Run], dict]):
    runs, truth = plat
    assert runs
    assert truth["offset_lines"]

    for line in truth["offset_lines"]:
        a, b = to_px(line["start_pt"]), to_px(line["end_pt"])
        for run in runs:
            points = densify(run.points_px)
            distance = point_segment_distance(points, a, b)
            to_endpoint = np.minimum(np.hypot(*(points - a).T), np.hypot(*(points - b).T))
            interior = to_endpoint > 8.0  # the same endpoint slop A7 allows the boundary runs
            if interior.sum() == 0:
                continue
            assert np.all(distance[interior] > 3.0), f"{line['id']} traced as geometry"

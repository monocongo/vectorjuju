"""Run-level gates on the synthetic plat fixture: A7 plus the curve half of U3/A4.

The full acceptance suite reads A7 and A4 off a DXF; until `convert()` emits
one (issue #22), the same properties are asserted over traced runs in raster
px: one run per planted straight edge, reaching its endpoints, with no second
parallel run and nothing on the right-of-way distractors; the two planted arcs
classify as circles with the right radius, endpoints, and bulge; the curve
table never traces as geometry.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from acceptance_helpers import arc_midpoint
from PIL import Image

from vectorjuju.curves import classify
from vectorjuju.synthetic_plat import PAGE_H, RENDER_DPI, SCALE_PT_PER_FT, generate_sheet
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


def endpoint_reach(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Worst end distance of the better start/end assignment to (a, b)."""
    forward = max(np.hypot(*(points[0] - a)), np.hypot(*(points[-1] - b)))
    backward = max(np.hypot(*(points[0] - b)), np.hypot(*(points[-1] - a)))
    return float(min(forward, backward))


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


def test_planted_arcs_classify_as_circles(plat: tuple[list[Run], dict]):
    """A4 at run level: the two planted arcs become circle fits and nothing else does.

    Radius, endpoint, and bulge gates match the acceptance suite's; the 4 px
    endpoint check is the fitted circle against the traced run's own ends here,
    because A4's 4 px against the planted corners needs the arc/line endpoint
    joining that lands with entity assembly in issue #22 (the traced ends sit
    inside A7's 8 px corner-splitting slop).
    """
    runs, truth = plat
    classified = [classify(run.points_px, RENDER_DPI) for run in runs]
    curves = [(index, fit) for index, (kind, fit) in enumerate(classified) if kind == "curve"]

    assert len(curves) == 2  # the fixture plants two arcs
    assert all(fit is not None for _, fit in curves)  # so the writer emits zero SPLINE

    radii_ft = {curve["id"]: curve["radius_ft"] for curve in truth["curves"]}
    anchors = {label["segment_id"]: label["anchor_pt"] for label in truth["labels"] if label["kind"] == "curve_ref"}
    matched = []
    for segment in (segment for segment in truth["segments"] if segment["kind"] == "curve"):
        radius_px = radii_ft[segment["curve_id"]] * SCALE_PT_PER_FT * RENDER_DPI / 72.0
        # Match on the fitted radius, not endpoint proximity: the radius is
        # distinctive while the traced ends sit inside the corner-splitting slop.
        hits = [(index, fit) for index, fit in curves if abs(fit.radius_px - radius_px) / radius_px <= 0.03]
        assert len(hits) == 1, f"{segment['id']}: {len(hits)} curve runs fit its radius"
        index, fit = hits[0]
        assert fit is not None
        matched.append(index)

        a, b = to_px(segment["start_pt"]), to_px(segment["end_pt"])
        assert endpoint_reach(runs[index].points_px, a, b) <= 8.0, segment["id"]
        for end in (runs[index].points_px[0], runs[index].points_px[-1]):
            centre = np.asarray(fit.center_px)
            assert abs(float(np.hypot(*(end - centre))) - fit.radius_px) <= 4.0
        anchor = to_px(anchors[segment["id"]])
        assert np.hypot(*(arc_midpoint(runs[index].points_px, fit) - anchor)) <= 8.0
    assert len(set(matched)) == 2  # each arc claims its own run


def test_no_run_follows_the_curve_table_border(plat: tuple[list[Run], dict]):
    """No traced run touches the table: its 0.7 pt borders and cell text stay out.

    The 0.7 pt border is already under the tracer's stroke-width gate, so this
    is the canary for that rejection: a border run that ever got through would
    land inside or within 3 px of the table box. The DXF-level gate lands with
    issue #22.
    """
    runs, truth = plat
    x0, y0, x1, y1 = truth["curve_table_bbox_pt"]
    corners = [to_px([x, y]) for x in (x0, x1) for y in (y0, y1)]
    left, right = min(corner[0] for corner in corners), max(corner[0] for corner in corners)
    top, bottom = min(corner[1] for corner in corners), max(corner[1] for corner in corners)

    for run in runs:
        points = densify(run.points_px)
        dx = np.maximum(np.maximum(left - points[:, 0], points[:, 0] - right), 0.0)
        dy = np.maximum(np.maximum(top - points[:, 1], points[:, 1] - bottom), 0.0)
        assert np.all(np.hypot(dx, dy) > 3.0), "a traced run reaches the curve table"

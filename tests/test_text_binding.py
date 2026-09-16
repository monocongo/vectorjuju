"""A2 and A3 at the run level, plus A4's binding half, on the synthetic plat
fixture.

The full acceptance suite reads these off a JSON sidecar; until `convert()`
writes one (issue #22), the same properties are asserted over `BoundCall`'s
own run and parsed-call fields -- the pattern test_tracing_plat.py set for
A7. Curve-ref binding (`test_each_planted_curve_ref_binds_exactly_once`) is
A4's assertion, not A2/A3's -- covered here early since it needs no fitted
arc, only a run to bind to. This is the OCR path (real docling, real crops),
so it is marked ``acceptance`` and stays local until #22 stands up the CI
acceptance job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectorjuju.calibrate import calibrate_scale
from vectorjuju.synthetic_plat import PAGE_H, RENDER_DPI, generate_sheet
from vectorjuju.text import BoundCall, TextItem, bind_calls, extract_text, normalize_ocr, page_items
from vectorjuju.tracing import Run, trace_runs

pytestmark = pytest.mark.acceptance


def test_page_items_converts_docling_bottom_left_boxes_to_top_left_px() -> None:
    """`page_items()` and the crop pass share `_doc_items()`'s docling
    bottom-left -> package top-left conversion, but the crop pass then maps
    its box back through a band-local affine transform, which could mask a
    flip bug in the shared conversion itself. Pin that conversion directly
    with page-pass text at a known pixel position."""
    from PIL import ImageDraw, ImageFont

    width, height = 600, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((50, 150), "TOP", fill="black", font=font)
    draw.text((50, 600), "BOTTOM", fill="black", font=font)

    items = {item.text.strip().upper(): item for item in page_items(image)}

    assert "TOP" in items
    assert "BOTTOM" in items
    top_box, bottom_box = items["TOP"].box_px, items["BOTTOM"].box_px
    for box in (top_box, bottom_box):
        assert box[1] < box[3]  # y0 < y1: top-left origin, y down
    assert top_box[1] < height / 2  # drawn near the top -> small y
    assert bottom_box[1] > height / 2  # drawn near the bottom -> large y


def to_px(point: list[float]) -> np.ndarray:
    scale = RENDER_DPI / 72.0
    return np.array([point[0] * scale, (PAGE_H - point[1]) * scale])


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(((point - a) @ ab) / (ab @ ab), 0.0, 1.0))
    return float(np.hypot(*(point - (a + t * ab))))


def run_hugs_edge(run: Run, a: np.ndarray, b: np.ndarray, *, tolerance: float = 3.0) -> bool:
    """True if every point of ``run`` sits within ``tolerance`` of segment ab --
    the same band test_tracing_plat.py's edge_hits searches for, at the level
    of one already-identified run."""
    return all(point_segment_distance(p, a, b) <= tolerance for p in run.points_px)


@pytest.fixture(scope="module")
def bound_and_unbound(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[BoundCall], list[TextItem], dict]:
    out = tmp_path_factory.mktemp("plat")
    truth = generate_sheet(out)
    image = Image.open(Path(out) / "sheet.tif")
    # One page pass, reused below for extraction instead of re-run (text.py's
    # module docstring). Not fed to trace_runs's exclude_mask here: this
    # fixture's labels are rotated to align with -- and nearly span -- their
    # own boundary edge (issue #8), so a page-level (non-deskewed) OCR box
    # for one is an axis-aligned bound on a diagonal run of glyphs, wide and
    # tall enough to blank most of the edge it sits beside. That masking
    # path needs its own fix (oversized exclusion regions for rotated text,
    # not something introduced here); tracked separately rather than folded
    # into this pass.
    page = page_items(image)
    runs = trace_runs(image, RENDER_DPI)
    items = extract_text(image, runs, dpi=RENDER_DPI, page_items=page)
    bound, unbound = bind_calls(runs, items, dpi=RENDER_DPI)
    return bound, unbound, truth


def test_each_planted_straight_call_binds_exactly_once(
    bound_and_unbound: tuple[list[BoundCall], list[TextItem], dict],
) -> None:
    bound, _unbound, truth = bound_and_unbound
    straight = [s for s in truth["segments"] if s["kind"] == "straight"]
    assert len(straight) == 4  # the fixture plants four straight edges

    for segment in straight:
        a, b = to_px(segment["start_pt"]), to_px(segment["end_pt"])
        matches = [bc for bc in bound if run_hugs_edge(bc.run, a, b)]
        assert len(matches) == 1, f"{segment['id']}: {len(matches)} bound calls on this edge"
        call = matches[0].call
        assert call.kind == "bearing_distance"
        # Digit/letter recovery is near-exact post-deskew (issue #8); only
        # unit-mark punctuation is corrupted, so the parsed *values* -- not
        # necessarily raw_text -- should match the planted call closely.
        label = next(lab for lab in truth["labels"] if lab["segment_id"] == segment["id"])
        planted_bearing, planted_distance = label["text"].rsplit(None, 1)
        assert abs(call.distance_ft - float(planted_distance.rstrip("'"))) <= 0.1
        assert planted_bearing[:1] in ("N", "S")  # sanity: the label really is a bearing call


def test_each_planted_curve_ref_binds_exactly_once(
    bound_and_unbound: tuple[list[BoundCall], list[TextItem], dict],
) -> None:
    bound, _unbound, truth = bound_and_unbound
    curve_refs = [lab["text"] for lab in truth["labels"] if lab["kind"] == "curve_ref"]
    assert set(curve_refs) == {"C1", "C2"}

    curve_calls = [bc for bc in bound if bc.call.kind == "curve_ref"]
    bound_ids = [bc.call.curve_id for bc in curve_calls]
    assert sorted(bound_ids) == sorted(curve_refs)  # each ref binds exactly once, none extra


def test_calibration_recovers_the_planted_scale_from_traced_runs(
    bound_and_unbound: tuple[list[BoundCall], list[TextItem], dict],
) -> None:
    """A1's scale half at the run level: the parsed calls and the traced runs
    they bound to recover the planted feet-per-pixel inside the 1 % budget.
    The full gate reads the same number off the sidecar once `convert()`
    writes one (issue #22)."""
    bound, _unbound, truth = bound_and_unbound
    planted = 72.0 / (truth["scale_pt_per_ft"] * RENDER_DPI)

    scale = calibrate_scale(bound, dpi=RENDER_DPI)

    assert scale.method == "ransac"
    assert abs(scale.value - planted) <= 0.01 * planted


def test_no_distractor_or_table_cell_text_binds(
    bound_and_unbound: tuple[list[BoundCall], list[TextItem], dict],
) -> None:
    bound, _unbound, truth = bound_and_unbound
    banned = set(truth["distractor_text"])
    for row in truth["curve_table_cells"][1:]:  # skip the header row
        for cell in row:
            # C1/C2 table cells are textually identical to the planted
            # on-curve labels (docs/acceptance-suite.md A3) -- distinguishing
            # them needs the fitted-arc insertion check (A4, issue #18/#21).
            if cell not in ("C1", "C2"):
                banned.add(cell)
    banned_normalized = {normalize_ocr(text) for text in banned}

    for bc in bound:
        assert normalize_ocr(bc.call.raw_text) not in banned_normalized


def test_nothing_binds_near_the_offset_line_distractors(
    bound_and_unbound: tuple[list[BoundCall], list[TextItem], dict],
) -> None:
    bound, _unbound, truth = bound_and_unbound
    assert truth["offset_lines"]

    for line in truth["offset_lines"]:
        a, b = to_px(line["start_pt"]), to_px(line["end_pt"])
        for bc in bound:
            for p in bc.run.points_px:
                assert point_segment_distance(p, a, b) > 3.0, f"{line['id']} has a bound call nearby"

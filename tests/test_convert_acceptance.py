"""A1-A11: the full ``convert()`` path over PDF, TIFF, and EXIF-rotated JPG.

Gate definitions and tolerances live in ``docs/acceptance-suite.md``. Every
fixture comes from ``conftest.py``: one generated sheet per session, one
``convert()`` per medium, and a repeat run per medium for determinism.
"""

from __future__ import annotations

import json
import math
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pytest
from acceptance_helpers import (
    arc_points,
    label_passes,
    normalize_text,
    planted_fpp,
    planted_fpp_m,
    point_segment_distance,
    pt_to_px,
    quad_distance,
    to_px_points,
)
from conftest import MEDIA, Conversion, SyntheticSheet
from ezdxf.recover import readfile

from vectorjuju.calibrate import ScaleCalibrationError
from vectorjuju.convert import UnsupportedInputError, convert, load_raster
from vectorjuju.export import LAYERS
from vectorjuju.synthetic_plat import RENDER_DPI, SCALE_PT_PER_FT

pytestmark = pytest.mark.acceptance

BOUNDARY_TYPES = {"LWPOLYLINE", "ARC", "SPLINE"}


def _boundary_entities(doc) -> list:
    """Boundary entities in document order -- the order ``write_outputs`` adds them."""
    return [entity for entity in doc.modelspace() if entity.dxftype() in BOUNDARY_TYPES]


def _entity_px(entry: dict, conversion: Conversion) -> np.ndarray:
    return to_px_points(
        entry["points"], fpp=float(conversion.sidecar["scale"]["value"]), img_height=conversion.img_height
    )


def _segment_px(segment: dict, conversion: Conversion) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(pt_to_px(segment["start_pt"], dpi=RENDER_DPI, img_height=conversion.img_height)),
        np.array(pt_to_px(segment["end_pt"], dpi=RENDER_DPI, img_height=conversion.img_height)),
    )


def _label_px(label: dict, conversion: Conversion) -> np.ndarray:
    return np.array(pt_to_px(label["anchor_pt"], dpi=RENDER_DPI, img_height=conversion.img_height))


def _quad_px(label: dict, conversion: Conversion) -> list[np.ndarray]:
    return [np.array(pt_to_px(point, dpi=RENDER_DPI, img_height=conversion.img_height)) for point in label["quad_pt"]]


def _straight_labels(sheet: SyntheticSheet) -> list[dict]:
    return [label for label in sheet.truth["labels"] if label["kind"] == "straight_call"]


def _curve_labels(sheet: SyntheticSheet) -> list[dict]:
    return [label for label in sheet.truth["labels"] if label["kind"] == "curve_ref"]


def _segment_of(sheet: SyntheticSheet, label: dict) -> dict:
    return next(segment for segment in sheet.truth["segments"] if segment["id"] == label["segment_id"])


def _matched_entity(conversion: Conversion, text: str) -> dict:
    """The single sidecar entity whose label reads ``text`` with fidelity."""
    matches = [entry for entry in conversion.labeled if label_passes(entry["label"]["raw_text"], text)]
    assert len(matches) == 1, (
        f"{text!r}: {len(matches)} sidecar entities match; "
        f"bound={[entry['label']['raw_text'] for entry in conversion.labeled]!r} "
        f"unbound={[item['raw_text'] for item in conversion.sidecar['unbound_text']]!r}"
    )
    return matches[0]


def _dxf_index(conversion: Conversion, entry: dict) -> int:
    """The DXF boundary entity's index for a sidecar entity: 1:1 in order."""
    return conversion.entities.index(entry)


def _mtext_for(doc, text: str):
    matches = [entity for entity in doc.modelspace() if entity.dxftype() == "MTEXT" and entity.text == text]
    assert len(matches) == 1, f"{text!r}: {len(matches)} MTEXT entities match"
    return matches[0]


def _fold180(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def _raster_height(sheet: SyntheticSheet, medium: str = "pdf") -> int:
    return load_raster(sheet.media(medium), dpi=RENDER_DPI).size[1]


# --- A1: scale ---------------------------------------------------------------


def test_a1_planted_scale_is_recovered_from_the_calls(conversion: Conversion) -> None:
    scale = conversion.sidecar["scale"]
    assert scale["method"] == "ransac"
    assert abs(scale["value"] - planted_fpp()) / planted_fpp() <= 0.01


def test_a1_scale_override_skips_calibration_and_is_recorded(
    synthetic_sheet: SyntheticSheet, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out = convert(
        synthetic_sheet.media("pdf"), out=tmp_path_factory.mktemp("override") / "sheet.dxf", scale=planted_fpp()
    )
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["scale"] == {"value": planted_fpp(), "method": "override"}


# --- A2: every planted straight call binds exactly once ----------------------


def test_a2_each_straight_call_binds_exactly_once_within_the_run(
    conversion: Conversion, synthetic_sheet: SyntheticSheet
) -> None:
    for label in _straight_labels(synthetic_sheet):
        entry = _matched_entity(conversion, label["text"])
        assert entry["type"] == "line"

        segment = _segment_of(synthetic_sheet, label)
        start, end = _segment_px(segment, conversion)
        points = _entity_px(entry, conversion)

        assert len(points) >= 2
        interior = points[1:-1]
        if len(interior):
            assert point_segment_distance(interior, start, end).max() <= 3.0, label["id"]
        for extreme in (points[0], points[-1]):
            assert min(float(np.hypot(*(extreme - start))), float(np.hypot(*(extreme - end)))) <= 8.0, label["id"]


# --- A3: distractors never bind ----------------------------------------------


def test_a3_no_distractor_or_table_text_ever_binds(conversion: Conversion, synthetic_sheet: SyntheticSheet) -> None:
    truth = synthetic_sheet.truth
    for line in truth["offset_lines"]:
        offset_start = np.array(pt_to_px(line["start_pt"], dpi=RENDER_DPI, img_height=conversion.img_height))
        offset_end = np.array(pt_to_px(line["end_pt"], dpi=RENDER_DPI, img_height=conversion.img_height))
        for entry in conversion.labeled:
            assert point_segment_distance(_entity_px(entry, conversion), offset_start, offset_end).min() > 3.0, line[
                "id"
            ]

    # The C1/C2 table cells are textually identical to the on-curve labels; A4's
    # insertion check is what proves which one was read.
    cells = {cell for row in truth["curve_table_cells"] for cell in row} - {"C1", "C2"}
    banned = {normalize_text(text) for text in truth["distractor_text"]} | {normalize_text(cell) for cell in cells}
    for entry in conversion.labeled:
        assert normalize_text(entry["label"]["raw_text"]) not in banned


# --- A4: exactly the planted arcs, bound to their curve refs -----------------


def test_a4_true_arcs_with_endpoints_radius_and_bulge(conversion: Conversion, synthetic_sheet: SyntheticSheet) -> None:
    doc, auditor = readfile(conversion.out)
    assert auditor.errors == []

    arcs = [entity for entity in doc.modelspace() if entity.dxftype() == "ARC"]
    assert len(arcs) == 2
    assert all(entity.dxf.layer == "BOUNDARY_CURVE" for entity in arcs)
    assert [entity for entity in doc.modelspace() if entity.dxftype() == "SPLINE"] == []

    boundaries = _boundary_entities(doc)
    assert len(boundaries) == len(conversion.entities)

    for label in _curve_labels(synthetic_sheet):
        entry = _matched_entity(conversion, label["text"])
        assert entry["type"] == "curve"
        arc = boundaries[_dxf_index(conversion, entry)]
        assert arc.dxftype() == "ARC"

        segment = _segment_of(synthetic_sheet, label)
        planted_radius_px = (
            next(curve["radius_ft"] for curve in synthetic_sheet.truth["curves"] if curve["id"] == segment["curve_id"])
            * SCALE_PT_PER_FT
            * RENDER_DPI
            / 72.0
        )
        assert (
            abs(arc.dxf.radius / float(conversion.sidecar["scale"]["value"]) - planted_radius_px) / planted_radius_px
            <= 0.03
        ), label["id"]

        ends = to_px_points(
            [tuple(arc.start_point)[:2], tuple(arc.end_point)[:2]],
            fpp=float(conversion.sidecar["scale"]["value"]),
            img_height=conversion.img_height,
        )
        start, end = _segment_px(segment, conversion)
        reach = min(
            float(np.hypot(*(ends[0] - start))) + float(np.hypot(*(ends[1] - end))),
            float(np.hypot(*(ends[0] - end))) + float(np.hypot(*(ends[1] - start))),
        )
        assert reach <= 8.0, label["id"]  # 4 px per endpoint in the better orientation

        # The ARC's mid-sweep point must sit on the bulge side the planted
        # label anchors, which is what tells an arc from its complement.
        sweep = (arc.dxf.end_angle - arc.dxf.start_angle) % 360.0
        middle_angle = math.radians(arc.dxf.start_angle + sweep / 2.0)
        middle = (
            arc.dxf.center[0] + arc.dxf.radius * math.cos(middle_angle),
            arc.dxf.center[1] + arc.dxf.radius * math.sin(middle_angle),
        )
        middle_px = to_px_points(
            [middle], fpp=float(conversion.sidecar["scale"]["value"]), img_height=conversion.img_height
        )[0]
        assert float(np.hypot(*(middle_px - _label_px(label, conversion)))) <= 8.0, label["id"]

        mtext = _mtext_for(doc, entry["label"]["raw_text"])
        insertion = to_px_points(
            [tuple(mtext.dxf.insert)[:2]],
            fpp=float(conversion.sidecar["scale"]["value"]),
            img_height=conversion.img_height,
        )[0]
        assert quad_distance(insertion, _quad_px(label, conversion)) <= 20.0, label["id"]


# --- A5: units ---------------------------------------------------------------


def test_a5_default_run_is_us_survey_feet(conversion: Conversion) -> None:
    doc, _auditor = readfile(conversion.out)
    assert doc.header["$INSUNITS"] == 21
    assert doc.units == 21
    assert conversion.sidecar["units"] == "us-survey-foot"


@pytest.fixture(scope="session")
def international(synthetic_sheet: SyntheticSheet, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = convert(
        synthetic_sheet.media("pdf"),
        out=tmp_path_factory.mktemp("international") / "sheet.dxf",
        units="international-foot",
    )
    return out, json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def metre(synthetic_sheet: SyntheticSheet, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = convert(synthetic_sheet.media("pdf"), out=tmp_path_factory.mktemp("metre") / "sheet.dxf", units="metre")
    return out, json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def test_a5_international_feet_header_and_scale(international: tuple[Path, dict]) -> None:
    out, sidecar = international
    doc, _auditor = readfile(out)
    assert doc.header["$INSUNITS"] == 2
    assert sidecar["units"] == "international-foot"
    # 2 ppm from the US survey foot: below the 1 % budget, told apart by the header.
    assert abs(sidecar["scale"]["value"] - planted_fpp()) / planted_fpp() <= 0.01


def test_a5_metre_header_scale_and_coordinates(metre: tuple[Path, dict], synthetic_sheet: SyntheticSheet) -> None:
    out, sidecar = metre
    doc, _auditor = readfile(out)
    assert doc.header["$INSUNITS"] == 6
    assert sidecar["units"] == "metre"
    assert abs(sidecar["scale"]["value"] - planted_fpp_m()) / planted_fpp_m() <= 0.01

    # Every CAD coordinate is metres: invert with the metre fpp and the
    # geometry lands back on the planted raster.
    img_height = _raster_height(synthetic_sheet)
    label = _straight_labels(synthetic_sheet)[0]
    segment = _segment_of(synthetic_sheet, label)
    start, end = (
        np.array(pt_to_px(segment["start_pt"], dpi=RENDER_DPI, img_height=img_height)),
        np.array(pt_to_px(segment["end_pt"], dpi=RENDER_DPI, img_height=img_height)),
    )
    matches = [
        entry
        for entry in sidecar["entities"]
        if entry["label"] and label_passes(entry["label"]["raw_text"], label["text"])
    ]
    assert len(matches) == 1
    points = to_px_points(matches[0]["points"], fpp=float(sidecar["scale"]["value"]), img_height=img_height)
    interior = points[1:-1]
    if len(interior):
        assert point_segment_distance(interior, start, end).max() <= 3.0


# --- A6: label rotation and insertion -----------------------------------------


def test_a6_label_rotation_and_insertion(conversion: Conversion, synthetic_sheet: SyntheticSheet) -> None:
    doc, _auditor = readfile(conversion.out)
    mtexts = [entity for entity in doc.modelspace() if entity.dxftype() == "MTEXT"]
    assert mtexts
    for mtext in mtexts:
        assert -90.0 < mtext.dxf.rotation <= 90.0

    for label in _straight_labels(synthetic_sheet):
        entry = _matched_entity(conversion, label["text"])
        mtext = _mtext_for(doc, entry["label"]["raw_text"])
        assert abs(_fold180(mtext.dxf.rotation - label["rotation_deg"])) <= 5.0, label["id"]

        insertion = to_px_points(
            [tuple(mtext.dxf.insert)[:2]],
            fpp=float(conversion.sidecar["scale"]["value"]),
            img_height=conversion.img_height,
        )[0]
        assert quad_distance(insertion, _quad_px(label, conversion)) <= 20.0, label["id"]


# --- A7: one run per planted straight edge, no doubles ------------------------


def test_a7_one_run_per_straight_edge_within_reach(conversion: Conversion, synthetic_sheet: SyntheticSheet) -> None:
    doc, _auditor = readfile(conversion.out)
    polylines = [entity for entity in doc.modelspace() if entity.dxftype() == "LWPOLYLINE"]
    assert polylines

    for segment in (item for item in synthetic_sheet.truth["segments"] if item["kind"] == "straight"):
        start, end = _segment_px(segment, conversion)
        hits = []
        for entity in polylines:
            points = to_px_points(
                [point[:2] for point in entity.get_points("xy")],
                fpp=float(conversion.sidecar["scale"]["value"]),
                img_height=conversion.img_height,
            )
            if point_segment_distance(points, start, end).max() <= 3.0:
                hits.append(points)
        assert len(hits) == 1, f"{segment['id']}: {len(hits)} polylines run within 3 px"
        points = hits[0]
        for extreme in (points[0], points[-1]):
            assert min(float(np.hypot(*(extreme - start))), float(np.hypot(*(extreme - end)))) <= 8.0, segment["id"]


# --- A8: determinism ----------------------------------------------------------


def test_a8_two_runs_are_byte_identical(conversion: Conversion) -> None:
    assert conversion.out.read_bytes() == conversion.rerun_out.read_bytes()
    assert conversion.out.with_suffix(".json").read_bytes() == conversion.rerun_out.with_suffix(".json").read_bytes()


# --- A9: the media agree ------------------------------------------------------


def _polyline_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    """Max vertex-to-polyline distance, both ways."""

    def one_way(points: np.ndarray, other: np.ndarray) -> float:
        return max(
            min(float(point_segment_distance(np.array([point]), a, b)[0]) for a, b in pairwise(other))
            for point in points
        )

    return max(one_way(first, second), one_way(second, first))


def test_a9_matching_entities_agree_across_media(
    conversions: dict[str, Conversion], synthetic_sheet: SyntheticSheet
) -> None:
    text_sets = {
        medium: {normalize_text(entry["label"]["raw_text"]) for entry in run.labeled}
        for medium, run in conversions.items()
    }
    assert text_sets["pdf"] == text_sets["tif"] == text_sets["jpg"]

    for medium_a, medium_b in combinations(MEDIA, 2):
        run_a, run_b = conversions[medium_a], conversions[medium_b]
        for segment in (item for item in synthetic_sheet.truth["segments"] if item["kind"] == "straight"):
            label = next(item for item in _straight_labels(synthetic_sheet) if item["segment_id"] == segment["id"])
            points_a = _entity_px(_matched_entity(run_a, label["text"]), run_a)
            points_b = _entity_px(_matched_entity(run_b, label["text"]), run_b)
            assert _polyline_hausdorff(points_a, points_b) <= 3.0, f"{segment['id']} {medium_a}/{medium_b}"

        for label in _curve_labels(synthetic_sheet):
            entry_a, entry_b = _matched_entity(run_a, label["text"]), _matched_entity(run_b, label["text"])
            fpp_a, fpp_b = float(run_a.sidecar["scale"]["value"]), float(run_b.sidecar["scale"]["value"])
            arc_a = _arc_in(run_a.out, _dxf_index(run_a, entry_a))
            arc_b = _arc_in(run_b.out, _dxf_index(run_b, entry_b))
            centres = [
                to_px_points([(arc.dxf.center[0], arc.dxf.center[1])], fpp=fpp, img_height=run.img_height)[0]
                for arc, fpp, run in ((arc_a, fpp_a, run_a), (arc_b, fpp_b, run_b))
            ]
            assert float(np.hypot(*(centres[0] - centres[1]))) <= 3.0, f"{label['id']} {medium_a}/{medium_b}"
            radii = (arc_a.dxf.radius / fpp_a, arc_b.dxf.radius / fpp_b)
            assert abs(radii[0] - radii[1]) <= 3.0, f"{label['id']} {medium_a}/{medium_b}"


def _arc_in(out: Path, index: int):
    doc, _auditor = readfile(out)
    entity = _boundary_entities(doc)[index]
    assert entity.dxftype() == "ARC"
    return entity


# --- A10: failure paths -------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "sheet.txt",
        "corrupt.pdf",
        "corrupt.tif",
        "empty.pdf",
        "two_page.pdf",
        "two_frame.tif",
        "zero_page.pdf",
        "mislabeled.jpg",
    ],
)
def test_a10_unsupported_inputs_raise_and_write_nothing(bad_inputs: Path, tmp_path: Path, name: str) -> None:
    out = tmp_path / "fresh" / "sheet.dxf"
    with pytest.raises(UnsupportedInputError):
        convert(bad_inputs / name, out=out)
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


def test_a10_uncalibratable_page_raises_and_writes_nothing(bad_inputs: Path, tmp_path: Path) -> None:
    out = tmp_path / "fresh" / "sheet.dxf"
    with pytest.raises(ScaleCalibrationError):
        convert(bad_inputs / "text_only.pdf", out=out)
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


# --- A11: sidecar and DXF are one document ------------------------------------


def test_a11_every_sidecar_entity_maps_onto_the_dxf(conversion: Conversion) -> None:
    doc, auditor = readfile(conversion.out)
    assert auditor.errors == []

    used = {entity.dxf.layer for entity in doc.modelspace()}
    assert used <= set(LAYERS)
    assert not used & {"0", "Defpoints"}

    boundaries = _boundary_entities(doc)
    assert len(boundaries) == len(conversion.entities)
    fpp = float(conversion.sidecar["scale"]["value"])
    for entity, entry in zip(boundaries, conversion.entities, strict=True):
        assert entity.dxf.layer == ("BOUNDARY_LINE" if entry["type"] == "line" else "BOUNDARY_CURVE")
        if entity.dxftype() == "LWPOLYLINE":
            points = [list(map(float, point[:2])) for point in entity.get_points("xy")]
            assert points == (entry["points"][:-1] if entity.closed else entry["points"])
        elif entity.dxftype() == "SPLINE":
            points = [list(map(float, point[:2])) for point in entity.fit_points]
            assert points == (entry["points"][:-1] if entity.closed else entry["points"])
        else:
            assert entity.dxftype() == "ARC"
            run_px = _entity_px(entry, conversion)
            sweep = (entity.dxf.end_angle - entity.dxf.start_angle) % 360.0
            fitted = arc_points(
                entity.dxf.start_angle,
                entity.dxf.start_angle + sweep,
                (entity.dxf.center[0], entity.dxf.center[1]),
                entity.dxf.radius,
            )
            fitted_px = to_px_points(fitted, fpp=fpp, img_height=conversion.img_height)
            assert _polyline_hausdorff(run_px, fitted_px) <= 3.0
            if entry["label"] is not None:
                assert entity.dxf.radius == pytest.approx(entry["label"]["radius"], rel=1e-9)

    texts = [entity.text for entity in doc.modelspace() if entity.dxftype() == "MTEXT"]
    assert texts == [entry["label"]["raw_text"] for entry in conversion.labeled]

"""U4, A8, A11: the DXF and sidecar writer contract, without OCR or tracing.

Runs are planted directly, so a failure names the writer property rather than
fixture drift; the acceptance suite re-runs these gates over traced media.
"""

from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from ezdxf.recover import readfile

from vectorjuju.calibrate import Scale
from vectorjuju.convert import px_to_cad
from vectorjuju.curves import classify
from vectorjuju.export import LAYERS, write_outputs
from vectorjuju.text import BoundCall, ParsedCall, TextItem
from vectorjuju.tracing import Run

DPI = 200
IMG_HEIGHT = 1600
FPP = 0.4237
SCALE = Scale(value=FPP, method="override")
CHAR_HEIGHT = 7.5 / 72.0 * DPI * FPP


def _line_run() -> Run:
    return Run(points_px=np.array([[100.0, 100.0], [150.0, 120.0], [200.0, 140.0]]))


def _arc_run(centre=(500.0, 500.0), radius=300.0, start_deg=20.0, end_deg=60.0) -> Run:
    angles = np.radians(np.linspace(start_deg, end_deg, 49))
    return Run(points_px=np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)]))


def _full_circle_run() -> Run:
    angles = np.radians(np.linspace(0.0, 360.0, 73))[:-1]
    points = np.column_stack([500.0 + 300.0 * np.cos(angles), 500.0 + 300.0 * np.sin(angles)])
    return Run(points_px=np.vstack([points, points[0]]))  # exact closure


def _spline_run() -> Run:
    """A bowed run the circle fit rejects: 21 points on a parabola, the shape a
    traced run takes when it bends but is not one arc."""
    x = np.linspace(0.0, 100.0, 21)
    y = 40.0 - 40.0 * ((x / 50.0) - 1.0) ** 2
    return Run(points_px=np.column_stack([x, y]))


def _degenerate_closed_run() -> Run:
    """The only shape that closes and still classifies as straight: the tracer's
    minimum extent means a real closed loop bows past the line gate, so the
    contract's closed-LWPOLYLINE clause is unreachable from traced input."""
    return Run(points_px=np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]))


def _bound(run: Run, text: str, box: tuple[float, float, float, float], **call) -> BoundCall:
    parsed = ParsedCall(
        raw_text=text,
        kind=call.get("kind", "bearing_distance"),
        bearing_deg=call.get("bearing"),
        distance_ft=call.get("distance"),
        curve_id=call.get("curve_id"),
        suspect_tokens=(),
    )
    return BoundCall(call=parsed, run=run, item=TextItem(text=text, box_px=box, source="page"), distance_px=0.0)


def _planted() -> tuple[list[Run], list[BoundCall], list[TextItem]]:
    line, arc, spline, circle, closed = (
        _line_run(),
        _arc_run(),
        _spline_run(),
        _full_circle_run(),
        _degenerate_closed_run(),
    )
    bound = [
        _bound(line, "N 21°48'05\" E 100.00'", (140.0, 80.0, 260.0, 100.0), bearing=21.8, distance=100.0),
        _bound(arc, "C1", (520.0, 300.0, 560.0, 320.0), kind="curve_ref", curve_id="C1"),
    ]
    unbound = [TextItem(text="PARCEL 5", box_px=(900.0, 1400.0, 1000.0, 1420.0), source="page")]
    return [line, arc, spline, circle, closed], bound, unbound


def _write(directory: Path, runs, bound, unbound, *, scale=SCALE, **kwargs) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    out = write_outputs(directory / "sheet.dxf", runs, bound, unbound, scale=scale, img_height=IMG_HEIGHT, **kwargs)
    return out, json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def _cad(point) -> tuple[float, float]:
    x, y = px_to_cad(tuple(point), fpp=FPP, img_height=IMG_HEIGHT)
    return float(x), float(y)


def test_reopens_clean_on_the_contract_dxf(tmp_path):
    runs, bound, unbound = _planted()
    out, sidecar = _write(tmp_path, runs, bound, unbound)
    doc, auditor = readfile(out)

    assert out == tmp_path / "sheet.dxf"
    assert auditor.errors == []
    assert doc.header["$ACADVER"] == "AC1015"
    assert doc.header["$INSUNITS"] == 21
    assert doc.units == 21

    assert set(LAYERS) <= {layer.dxf.name for layer in doc.layers}
    used = {entity.dxf.layer for entity in doc.modelspace()}
    assert used <= set(LAYERS)
    assert not used & {"0", "Defpoints"}

    types = [entity.dxftype() for entity in doc.modelspace()]
    assert types == ["LWPOLYLINE", "ARC", "SPLINE", "SPLINE", "LWPOLYLINE", "MTEXT", "MTEXT"]
    assert all(entity.xdata is None for entity in doc.modelspace())  # no XDATA, so no APPID referenced
    assert "ACAD" in {appid.dxf.name for appid in doc.appids}

    # A11: sidecar entities[] line up index for index with the boundary
    # entities, and nothing else shares their layers.
    boundaries = [entity for entity in doc.modelspace() if entity.dxf.layer in {"BOUNDARY_LINE", "BOUNDARY_CURVE"}]
    assert len(boundaries) == len(sidecar["entities"])
    for entity, entry in zip(boundaries, sidecar["entities"], strict=True):
        assert entry["type"] == ("line" if entity.dxftype() == "LWPOLYLINE" else "curve")
        assert entity.dxf.layer == ("BOUNDARY_LINE" if entry["type"] == "line" else "BOUNDARY_CURVE")
        if entity.dxftype() == "LWPOLYLINE":
            points = [list(map(float, point[:2])) for point in entity.get_points("xy")]
            assert points == (entry["points"][:-1] if entity.closed else entry["points"])
        elif entity.dxftype() == "SPLINE":
            points = [list(map(float, point[:2])) for point in entity.fit_points]
            assert points == (entry["points"][:-1] if entity.closed else entry["points"])
    mtexts = [entity for entity in doc.modelspace() if entity.dxftype() == "MTEXT"]
    assert len(mtexts) == sum(1 for entry in sidecar["entities"] if entry["label"])  # unbound text is sidecar-only


def test_sidecar_matches_the_settled_schema(tmp_path):
    runs, bound, unbound = _planted()
    _, sidecar = _write(tmp_path, runs, bound, unbound)

    assert set(sidecar) == {"units", "scale", "dpi", "entities", "unbound_text"}
    assert sidecar["units"] == "us-survey-foot"
    assert sidecar["scale"] == {"value": FPP, "method": "override"}
    assert sidecar["dpi"] == DPI
    for run, entry in zip(runs, sidecar["entities"], strict=True):
        assert set(entry) == {"type", "points", "label"}
        assert entry["type"] in {"line", "curve"}
        assert [[*_cad(point)] for point in run.points_px] == entry["points"]
        if entry["label"] is not None:
            assert set(entry["label"]) == {"raw_text", "bearing", "distance", "radius"}

    line_label = sidecar["entities"][0]["label"]
    assert line_label == {"raw_text": "N 21°48'05\" E 100.00'", "bearing": 21.8, "distance": 100.0, "radius": None}
    _, fit = classify(runs[1].points_px, DPI)
    assert sidecar["entities"][1]["label"]["radius"] == pytest.approx(fit.radius_px * FPP)
    assert sidecar["entities"][1]["label"]["bearing"] is None

    item = unbound[0]
    assert sidecar["unbound_text"] == [
        {
            "raw_text": "PARCEL 5",
            "insertion": list(_cad(((item.box_px[0] + item.box_px[2]) / 2, (item.box_px[1] + item.box_px[3]) / 2))),
        }
    ]


def test_arc_is_the_fitted_circle_in_cad_units(tmp_path):
    run = _arc_run()
    out, _ = _write(tmp_path, [run], [], [])
    doc, _ = readfile(out)
    (arc,) = doc.modelspace()

    _, fit = classify(run.points_px, DPI)
    assert tuple(arc.dxf.center)[:2] == pytest.approx(_cad(fit.center_px))
    assert arc.dxf.radius == pytest.approx(fit.radius_px * FPP)
    assert (arc.dxf.end_angle - arc.dxf.start_angle) % 360.0 == pytest.approx(
        40.0
    )  # the traced sweep, not its complement
    ends = np.array(sorted([tuple(arc.start_point)[:2], tuple(arc.end_point)[:2]]), dtype=float)
    want = np.array(sorted([_cad(run.points_px[0]), _cad(run.points_px[-1])]), dtype=float)
    assert np.allclose(ends, want)


def test_closed_curve_falls_back_to_a_closed_spline(tmp_path):
    out, sidecar = _write(tmp_path, [_full_circle_run()], [], [])
    doc, _ = readfile(out)
    (spline,) = doc.modelspace()

    assert spline.dxftype() == "SPLINE"
    assert spline.closed
    assert len(spline.fit_points) == len(sidecar["entities"][0]["points"]) - 1


def test_label_rotation_reads_left_to_right_whichever_way_the_run_traced(tmp_path):
    points = np.array([[100.0, 100.0], [200.0, 240.0]])  # 140 px down over 100 across: -54.5 degrees in CAD
    rotations = []
    for index, run in enumerate([Run(points_px=points), Run(points_px=points[::-1])]):
        out, _ = _write(tmp_path / str(index), [run], [_bound(run, "N 1' E 1.00'", (0, 0, 10, 10))], [])
        doc, _ = readfile(out)
        mtext = next(entity for entity in doc.modelspace() if entity.dxftype() == "MTEXT")
        rotations.append(mtext.dxf.rotation)
        assert -90.0 < mtext.dxf.rotation <= 90.0
        assert mtext.text == "N 1' E 1.00'"
        assert mtext.dxf.char_height == pytest.approx(CHAR_HEIGHT)
        assert tuple(mtext.dxf.insert)[:2] == pytest.approx(_cad((5.0, 5.0)))
    assert rotations[0] == pytest.approx(-54.46, abs=0.01)
    assert rotations[1] == pytest.approx(rotations[0])


@pytest.mark.parametrize(("units", "insunits"), [("us-survey-foot", 21), ("international-foot", 2), ("metre", 6)])
def test_units_select_insunits(tmp_path, units, insunits):
    out, sidecar = _write(tmp_path, [_line_run()], [], [], units=units)

    doc, _ = readfile(out)
    assert doc.header["$INSUNITS"] == insunits == doc.units
    assert sidecar["units"] == units


def test_rejects_units_outside_the_contract(tmp_path):
    with pytest.raises(ValueError, match="units must be one of"):
        _write(tmp_path, [], [], [], units="furlong")
    assert list(tmp_path.iterdir()) == []


def test_two_runs_are_byte_identical(tmp_path):
    """A8: the same primitive set twice, in one process, byte for byte."""
    runs, bound, unbound = _planted()
    first = write_outputs(tmp_path / "one.dxf", runs, bound, unbound, scale=SCALE, img_height=IMG_HEIGHT)
    second = write_outputs(tmp_path / "two.dxf", runs, bound, unbound, scale=SCALE, img_height=IMG_HEIGHT)

    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_fixed_metadata_global_does_not_leak(tmp_path):
    runs, bound, unbound = _planted()
    write_outputs(tmp_path / "off.dxf", runs, bound, unbound, scale=SCALE, img_height=IMG_HEIGHT)
    assert ezdxf.options.write_fixed_meta_data_for_testing is False

    ezdxf.options.write_fixed_meta_data_for_testing = True
    try:
        write_outputs(tmp_path / "on.dxf", runs, bound, unbound, scale=SCALE, img_height=IMG_HEIGHT)
        assert ezdxf.options.write_fixed_meta_data_for_testing is True
    finally:
        ezdxf.options.write_fixed_meta_data_for_testing = False


def test_empty_sheet_writes_both_files(tmp_path):
    out, sidecar = _write(tmp_path, [], [], [])

    doc, auditor = readfile(out)
    assert auditor.errors == []
    assert list(doc.modelspace()) == []
    assert sidecar["entities"] == []
    assert sidecar["unbound_text"] == []


def test_scale_method_survives_into_the_sidecar(tmp_path):
    _, sidecar = _write(tmp_path, [_line_run()], [], [], scale=Scale(value=FPP, method="ransac"))
    assert sidecar["scale"] == {"value": FPP, "method": "ransac"}

"""Write the calibrated primitive set as a DXF drawing and its JSON sidecar.

Every traced run becomes exactly one boundary entity -- a ``LWPOLYLINE`` on
layer ``BOUNDARY_LINE``, a true ``ARC`` on ``BOUNDARY_CURVE`` where the circle
fit holds, or the ``SPLINE`` fallback where it does not -- and every bound
call becomes one ``MTEXT`` on layer ``LABEL``. The sidecar lists the same
entities in the same order, in CAD units, so the two files map 1:1.

No provenance data goes into the DXF: no XDATA, no APPID reference, no
metadata layer. The sidecar is the whole record of what was read and where it
came from.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import ezdxf

from vectorjuju.calibrate import Scale
from vectorjuju.convert import px_to_cad
from vectorjuju.curves import CircleFit, classify
from vectorjuju.text import BoundCall, ParsedCall, TextItem
from vectorjuju.tracing import Run

Units = Literal["us-survey-foot", "international-foot", "metre"]

LAYERS = ("BOUNDARY_LINE", "BOUNDARY_CURVE", "LABEL")
INSUNITS: dict[str, int] = {"us-survey-foot": 21, "international-foot": 2, "metre": 6}

# Call labels are 7.5 pt on the page -- the synthetic sheet's own FONT_SIZE and
# the size text.py's crop geometry assumes. MTEXT with no explicit height would
# take the DXF's unitless default, wrong at survey scale, so the height is that
# many page points at the working dpi, in drawing units.
_FONT_SIZE_PT = 7.5


def write_outputs(
    out: str | Path,
    runs: Sequence[Run],
    bound: Sequence[BoundCall],
    unbound: Sequence[TextItem],
    *,
    scale: Scale,
    img_height: int,
    units: Units = "us-survey-foot",
    dpi: int = 200,
) -> Path:
    """Write ``out`` (DXF) and ``<out stem>.json`` (sidecar); return the DXF path.

    ``scale`` is the calibrated units-per-pixel and how it was obtained, and
    ``img_height`` the ingested raster's height, the pixel-to-CAD flip's
    origin. ``bound`` are the calls bound to runs (one label each) and
    ``unbound`` everything that never bound, which the sidecar keeps as
    ``unbound_text``.

    Both files are byte-identical across runs over the same input: ezdxf
    otherwise stamps a version+timestamp marker at document creation, a
    written-by marker, save timestamps, and a random ``$VERSIONGUID`` at save
    time, and this global is its own documented switch for freezing all of
    them. It is restored afterwards; see the module test for the leak check.
    """
    out = Path(out)
    insunits = _insunits(units)
    labels = {call.run: call for call in bound}

    def to_cad(point) -> tuple[float, float]:
        return px_to_cad(tuple(point), fpp=scale.value, img_height=img_height)

    previous = ezdxf.options.write_fixed_meta_data_for_testing
    ezdxf.options.write_fixed_meta_data_for_testing = True
    try:
        doc = ezdxf.new("R2000")  # $ACADVER = AC1015
        doc.header["$INSUNITS"] = insunits
        for layer in LAYERS:
            doc.layers.add(layer)
        msp = doc.modelspace()
        char_height = _FONT_SIZE_PT / 72.0 * dpi * scale.value

        entities: list[dict] = []
        held: list[tuple[ParsedCall, TextItem, list[tuple[float, float]]]] = []
        for run in runs:
            cad = [to_cad(point) for point in run.points_px]
            closed = len(cad) > 2 and cad[0] == cad[-1]
            kind, fit = classify(run.points_px, dpi)
            _draw_boundary(msp, cad, closed, kind, fit, to_cad, scale.value)
            call = labels.get(run)
            entities.append(
                {
                    "type": kind,
                    "points": [[x, y] for x, y in cad],
                    "label": _sidecar_label(call, fit, scale.value),
                }
            )
            if call is not None:
                held.append((call.call, call.item, cad))

        # Labels last, so the boundary entities above and the sidecar's
        # entities[] line up index for index.
        for call, item, cad in held:
            _draw_label(msp, item, cad, char_height, to_cad)
        unbound_text = [
            {"raw_text": item.text, "insertion": list(to_cad(_box_center(item.box_px)))} for item in unbound
        ]
        sidecar = {
            "units": units,
            "scale": {"value": scale.value, "method": scale.method},
            "dpi": dpi,
            "entities": entities,
            "unbound_text": unbound_text,
        }

        doc.saveas(out)
        out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    finally:
        ezdxf.options.write_fixed_meta_data_for_testing = previous
    return out


def _insunits(units: str) -> int:
    try:
        return INSUNITS[units]
    except KeyError:
        raise ValueError(f"units must be one of {sorted(INSUNITS)}, got {units!r}") from None


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _draw_boundary(
    msp, cad: list[tuple[float, float]], closed: bool, kind: str, fit: CircleFit | None, to_cad, fpp: float
) -> None:
    points = cad[:-1] if closed else cad
    if kind == "line":
        msp.add_lwpolyline(points, close=closed, dxfattribs={"layer": "BOUNDARY_LINE"})
    elif fit is not None and not closed:
        cx, cy = to_cad(fit.center_px)
        first, last = cad[0], cad[-1]
        start = math.degrees(math.atan2(first[1] - cy, first[0] - cx))
        end = math.degrees(math.atan2(last[1] - cy, last[0] - cx))
        if fit.clockwise:  # ezdxf ARC always sweeps counter-clockwise
            start, end = end, start
        msp.add_arc(
            center=(cx, cy),
            radius=fit.radius_px * fpp,
            start_angle=start,
            end_angle=end,
            dxfattribs={"layer": "BOUNDARY_CURVE"},
        )
    else:
        spline = msp.add_spline(points, dxfattribs={"layer": "BOUNDARY_CURVE"})
        if closed:
            spline.closed = True


def _draw_label(msp, item: TextItem, cad: list[tuple[float, float]], char_height: float, to_cad) -> None:
    """One MTEXT at the label's own OCR position, middle-centered so the
    insertion point is the text box's centre however the text is rotated."""
    mtext = msp.add_mtext(
        item.text,
        dxfattribs={
            "layer": "LABEL",
            "rotation": _run_rotation(cad),
            "char_height": char_height,
            "attachment_point": 5,
        },
    )
    mtext.set_location(to_cad(_box_center(item.box_px)))


def _run_rotation(cad: list[tuple[float, float]]) -> float:
    """The run's own direction, folded to (-90, 90] like the fixture's labels:
    a run traced either way round gets the same rotation."""
    first, last = cad[0], cad[-1]
    angle = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def _sidecar_label(call: BoundCall | None, fit: CircleFit | None, fpp: float) -> dict | None:
    """The label's schema entry: the parsed call's own numbers, plus the fitted
    arc radius in CAD units -- radius never comes from a call's OCR text."""
    if call is None:
        return None
    parsed = call.call
    return {
        "raw_text": parsed.raw_text,
        "bearing": parsed.bearing_deg,
        "distance": parsed.distance_ft,
        "radius": fit.radius_px * fpp if fit is not None else None,
    }

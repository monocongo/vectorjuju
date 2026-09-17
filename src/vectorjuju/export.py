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

import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import ezdxf

from vectorjuju.calibrate import Scale
from vectorjuju.curves import CircleFit, classify
from vectorjuju.pipeline import Units, _require_fpp, _require_img_height, px_to_cad
from vectorjuju.text import BoundCall, ParsedCall, TextItem
from vectorjuju.tracing import Run

LAYERS = ("BOUNDARY_LINE", "BOUNDARY_CURVE", "LABEL")
INSUNITS: dict[str, int] = {"us-survey-foot": 21, "international-foot": 2, "metre": 6}

# Call labels are 7.5 pt on the page -- the synthetic sheet's own FONT_SIZE and
# the size text.py's crop geometry assumes. MTEXT with no explicit height would
# take the DXF's unitless default, wrong at survey scale, so the height is that
# many page points at the working dpi, in drawing units.
_FONT_SIZE_PT = 7.5

# ezdxf's fixed-metadata switch is process-global, so overlapping writes must
# not share the interval; whoever holds this holds the switch.
_META_LOCK = threading.Lock()


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
    origin. Both are enforced here, whatever the input: an empty sheet must
    not publish metadata that no conversion ever checked. ``out`` names the
    DXF and a ``.json`` path is rejected -- that is the sidecar's own.
    ``bound`` are the calls bound to runs (one label each) and ``unbound``
    everything that never bound, which the sidecar keeps as ``unbound_text``.

    Both artifacts are staged whole and then published sidecar first under an
    exclusive cross-process lock, so a failure writing either one leaves the
    previous pair in place and two conversions to one output cannot publish
    one run's sidecar beside the other's DXF. A failed DXF rename restores
    the previous sidecar, so the two files never disagree about their run.
    POSIX has no two-file transaction above that: a reader between the two
    renames can still see one run's sidecar beside the previous DXF.

    Both files are byte-identical across runs over the same input: ezdxf
    otherwise stamps a version+timestamp marker at document creation, a
    written-by marker, save timestamps, and a random ``$VERSIONGUID`` at save
    time, and this global is its own documented switch for freezing all of
    them. It is restored afterwards; see the module test for the leak check.
    """
    out = Path(out)
    sidecar_path = out.with_suffix(".json")
    if out.suffix.lower() == ".json":
        raise ValueError(f"out must name the DXF, got the sidecar's own path {out}")
    insunits = _insunits(units)
    fpp = _require_fpp(scale.value, "write_outputs")
    height = _require_img_height(img_height, "write_outputs")
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"write_outputs requires a finite dpi > 0, got {dpi!r}")
    labels = {call.run: call for call in bound}
    # Unique staged names: two conversions targeting the same output never
    # overwrite or delete each other's staging files.
    token = f"{os.getpid()}.{uuid4().hex}"
    staged_dxf = out.with_name(f"{out.name}.{token}.tmp")
    staged_sidecar = sidecar_path.with_name(f"{sidecar_path.name}.{token}.tmp")

    def to_cad(point) -> tuple[float, float]:
        return px_to_cad(tuple(point), fpp=fpp, img_height=height)

    _META_LOCK.acquire()
    previous = ezdxf.options.write_fixed_meta_data_for_testing
    ezdxf.options.write_fixed_meta_data_for_testing = True
    try:
        doc = ezdxf.new("R2000")  # $ACADVER = AC1015
        doc.header["$INSUNITS"] = insunits
        for layer in LAYERS:
            doc.layers.add(layer)
        msp = doc.modelspace()
        char_height = _FONT_SIZE_PT / 72.0 * dpi * fpp

        # One CAD list per run, shared by the entity and the sidecar entry: the
        # geometry is unbounded, so nothing keeps a second copy of it alive.
        entities: list[dict] = []
        held: list[tuple[ParsedCall, TextItem, list[list[float]]]] = []
        for run in runs:
            cad = [list(to_cad(point)) for point in run.points_px]
            closed = len(cad) > 2 and cad[0] == cad[-1]
            kind, fit = classify(run.points_px, dpi)
            _draw_boundary(msp, cad, closed, kind, fit, to_cad, fpp)
            call = labels.get(run)
            entities.append({"type": kind, "points": cad, "label": _sidecar_label(call, fit, fpp)})
            if call is not None:
                held.append((call.call, call.item, cad))

        # Labels last, so the boundary entities above and the sidecar's
        # entities[] line up index for index.
        for parsed, item, cad in held:
            _draw_label(msp, parsed.raw_text, item, cad, char_height, to_cad)
        unbound_text = [
            {"raw_text": item.text, "insertion": list(to_cad(_box_center(item.box_px)))} for item in unbound
        ]
        sidecar = {
            "units": units,
            "scale": {"value": fpp, "method": scale.method},
            "dpi": dpi,
            "entities": entities,
            "unbound_text": unbound_text,
        }

        doc.saveas(staged_dxf)
        with staged_sidecar.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2)
            handle.write("\n")
        # Both renames under one cross-process lock: without it two
        # conversions to the same output can interleave their two renames and
        # publish a sidecar from one beside a DXF from the other. The previous
        # sidecar is held only across the renames, so a DXF rename that fails
        # puts it back instead of leaving new metadata beside the old DXF.
        with _publish_lock_path(out).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous_sidecar = sidecar_path.read_bytes() if sidecar_path.is_file() else None
            try:
                staged_sidecar.replace(sidecar_path)
                try:
                    staged_dxf.replace(out)
                except OSError:
                    if previous_sidecar is None:
                        sidecar_path.unlink(missing_ok=True)
                    else:
                        staged_sidecar.write_bytes(previous_sidecar)
                        staged_sidecar.replace(sidecar_path)
                    raise
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        staged_dxf.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)
        ezdxf.options.write_fixed_meta_data_for_testing = previous
        _META_LOCK.release()
    return out


def _publish_lock_path(out: Path) -> Path:
    """The cross-process publish lock for ``out``, keyed by its resolved path.

    In the system temp directory, not beside the outputs: no lock file is left
    in the caller's output directory, and every process converting to the same
    output resolves to the same lock.
    """
    digest = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"vectorjuju-{digest}.lock"


def _insunits(units: str) -> int:
    try:
        return INSUNITS[units]
    except KeyError:
        raise ValueError(f"units must be one of {sorted(INSUNITS)}, got {units!r}") from None


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _draw_boundary(
    msp, cad: list[list[float]], closed: bool, kind: str, fit: CircleFit | None, to_cad, fpp: float
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


def _draw_label(msp, text: str, item: TextItem, cad: list[list[float]], char_height: float, to_cad) -> None:
    """One MTEXT at the label's own OCR position, middle-centered so the
    insertion point is the text box's centre however the text is rotated.

    ``text`` is the parsed call's transcription (``ParsedCall.raw_text``),
    which is also what the sidecar records: the drawing must not show a unit
    mark the parser already corrected.
    """
    mtext = msp.add_mtext(
        text,
        dxfattribs={
            "layer": "LABEL",
            "rotation": _run_rotation(cad),
            "char_height": char_height,
            "attachment_point": 5,
        },
    )
    mtext.set_location(to_cad(_box_center(item.box_px)))


def _run_rotation(cad: list[list[float]]) -> float:
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
    """The label's schema entry: the parsed call's own numbers, the untouched
    read and any unit-mark corrections the parser made, plus the fitted arc
    radius in CAD units -- radius never comes from a call's OCR text."""
    if call is None:
        return None
    parsed = call.call
    return {
        "raw_text": parsed.raw_text,
        "source_text": call.item.text,
        "suspect_tokens": list(parsed.suspect_tokens),
        "bearing": parsed.bearing_deg,
        "distance": parsed.distance_ft,
        "radius": fit.radius_px * fpp if fit is not None else None,
    }

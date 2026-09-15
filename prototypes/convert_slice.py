"""THROWAWAY spike for the map's build route -- the smallest convert() slice.

Real: raster ingest with EXIF normalization, one bearing/distance call parsed
from text, feet-per-pixel calibration from that call, px->CAD transform, DXF
(AC1015, ``$INSUNITS=21``) with one LWPOLYLINE + one MTEXT, JSON sidecar per
the issue-5 contract, and the raster/DXF/overlay comparison PNG.

Stand-ins, because tracing and docling are later tickets:
- the "traced run" is the ground-truth segment's raster endpoints
- the call text comes from ``ground_truth.json``, not OCR

Run::

    uv run --with ezdxf python prototypes/convert_slice.py \
        /tmp/vectorjuju-plat/sheet.tif /tmp/vectorjuju-plat/ground_truth.json \
        --out-dir /tmp/vectorjuju-slice
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import ezdxf
from PIL import Image, ImageDraw, ImageOps

CALL_RE = re.compile(r"^([NS])\s+(\d+)[\u00b0]\s*(\d+)'\s*(\d+(?:\.\d+)?)\"?\s+([EW])\s+([\d.]+)'$")
LAYERS = ("BOUNDARY_LINE", "BOUNDARY_CURVE", "LABEL")
FONT_SIZE_PT = 7.5
INSUNITS = {"us-survey-foot": 21, "international-foot": 2, "metre": 6}


def parse_call(text: str) -> tuple[float, float]:
    """One bearing/distance call -> (azimuth degrees 0-360, distance feet)."""
    m = CALL_RE.match(text.strip())
    if not m:
        raise ValueError(f"not a straight call: {text!r}")
    ns, d, mi, s, ew, dist = m.groups()
    ang = int(d) + int(mi) / 60 + float(s) / 3600
    if ns == "N":
        az = ang if ew == "E" else 360.0 - ang
    else:
        az = 180.0 - ang if ew == "E" else 180.0 + ang
    return az % 360.0, float(dist)


def pt_to_px(pt, dpi: float, img_h: int) -> tuple[float, float]:
    """Page points (bottom-left origin) -> raster px (top-left origin)."""
    return pt[0] * dpi / 72.0, img_h - pt[1] * dpi / 72.0


def write_dxf(
    path: Path, points_cad, label_cad, rotation_deg: float, text: str, units: str, text_height_ft: float
) -> None:
    # Byte-identical output across runs (acceptance gate A8): fixed GUIDs and
    # timestamps. Global ezdxf flag, named "for testing" upstream.
    ezdxf.options.write_fixed_meta_data_for_testing = True
    doc = ezdxf.new("R2000")  # $ACADVER = AC1015
    doc.header["$INSUNITS"] = INSUNITS[units]
    for layer in LAYERS:
        doc.layers.add(layer)
    msp = doc.modelspace()
    msp.add_lwpolyline(points_cad, dxfattribs={"layer": "BOUNDARY_LINE"})
    # char_height is drawing units: nominal 7.5 pt at the run DPI is not yet
    # in CAD units here, so the caller passes ft directly.
    mtext = msp.add_mtext(text, dxfattribs={"layer": "LABEL", "rotation": rotation_deg, "char_height": text_height_ft})
    mtext.set_location(label_cad)
    doc.saveas(path)


def read_dxf_points(path: Path, fpp: float, img_h: int):
    """Read the written DXF back the way a CAD consumer would, px for overlay."""
    doc = ezdxf.readfile(path)
    return [
        [(x / fpp, img_h - y / fpp) for x, y, *_ in e.get_points("xy")]
        for e in doc.modelspace()
        if e.dxftype() == "LWPOLYLINE"
    ]


def overlay_png(raster: Image.Image, traced_px, dxf_px, out: Path) -> None:
    w, h = raster.size
    geo, ov = Image.new("RGB", (w, h), "white"), raster.copy()
    dg, do = ImageDraw.Draw(geo), ImageDraw.Draw(ov)
    for pts in traced_px:
        dg.line(pts, fill=(0, 120, 0), width=2)
        do.line(pts, fill=(0, 120, 0), width=2)
    for pts in dxf_px:
        dg.line(pts, fill=(0, 0, 220), width=2)
        do.line(pts, fill=(0, 0, 220), width=2)

    def panel(im, label):
        p = Image.new("RGB", (850, 1130), "white")
        p.paste(im.resize((850, 1100)), (0, 30))
        ImageDraw.Draw(p).text((8, 8), label, fill="black")
        return p

    canvas = Image.new("RGB", (2550, 1130), "white")
    for i, (im, lab) in enumerate(
        [
            (raster, f"RASTER: {out.stem} input"),
            (geo, "DXF render (read back with ezdxf)"),
            (ov, "OVERLAY: raster + traced (green) + DXF (blue)"),
        ]
    ):
        canvas.paste(panel(im, lab), (850 * i, 0))
    canvas.save(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("ground_truth", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dpi", type=float, default=200.0)
    p.add_argument("--segment", default="S0")
    p.add_argument("--units", default="us-survey-foot", choices=sorted(INSUNITS))
    args = p.parse_args()

    # Ingest: EXIF-normalized raster, top-left px space.
    img = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    _, h = img.size
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))

    seg = next(s for s in gt["segments"] if s["id"] == args.segment)
    label = next(la for la in gt["labels"] if la["segment_id"] == args.segment)
    if seg["kind"] != "straight":
        raise SystemExit(f"{args.segment} is not a straight call")

    # Stand-in for tracing: planted endpoints in px.
    traced_px = [pt_to_px(seg["start_pt"], args.dpi, h), pt_to_px(seg["end_pt"], args.dpi, h)]
    length_px = math.dist(*traced_px)

    # Real: parse the call, calibrate feet-per-pixel from it.
    azimuth, distance_ft = parse_call(label["text"])
    fpp = distance_ft / length_px

    points_cad = [(x * fpp, (h - y) * fpp) for x, y in traced_px]
    for x, y in points_cad:
        assert 0 <= x and 0 <= y  # sane CAD quadrant

    label_px = pt_to_px(label["anchor_pt"], args.dpi, h)
    label_cad = (label_px[0] * fpp, (h - label_px[1]) * fpp)
    text_height_ft = FONT_SIZE_PT * args.dpi / 72.0 * fpp

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = out_dir / (args.input.stem + ".dxf")
    json_path = dxf_path.with_suffix(".json")
    write_dxf(dxf_path, points_cad, label_cad, label["rotation_deg"], label["text"], args.units, text_height_ft)
    sidecar = {
        "units": args.units,
        "scale": {"value": fpp, "method": "ransac"},
        "dpi": args.dpi,
        "entities": [
            {
                "type": "line",
                "points": [[round(x, 6), round(y, 6)] for x, y in points_cad],
                "label": {"raw_text": label["text"], "bearing": azimuth, "distance": distance_ft, "radius": None},
            }
        ],
        "unbound_text": [],
    }
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    dxf_px = read_dxf_points(dxf_path, fpp, h)
    overlay_png(img, [traced_px], dxf_px, out_dir / "compare.png")

    # Self-check: contract fields + geometry round trip.
    doc = ezdxf.readfile(dxf_path)
    planted_fpp = 72.0 / (gt["scale_pt_per_ft"] * args.dpi)
    errors = []
    if doc.header["$ACADVER"] != "AC1015":
        errors.append("ACADVER")
    if doc.header["$INSUNITS"] != INSUNITS[args.units]:
        errors.append("INSUNITS")
    used = {e.dxf.layer for e in doc.modelspace()}
    if not used <= set(LAYERS):
        errors.append(f"layers {used}")
    if abs(fpp - planted_fpp) / planted_fpp > 0.01:
        errors.append(f"scale {fpp:.6f} vs planted {planted_fpp:.6f}")
    back = dxf_px[0]
    if max(math.dist(a, b) for a, b in zip(back, traced_px, strict=True)) > 1e-6:
        errors.append("round trip")
    print(f"segment={args.segment} call={label['text']!r}")
    print(f"pixel length={length_px:.3f}  fpp={fpp:.6f}  planted={planted_fpp:.6f}")
    print(f"CAD extents={[(round(x, 2), round(y, 2)) for x, y in points_cad]}")
    print(f"wrote {dxf_path} and {json_path} and {out_dir / 'compare.png'}")
    print("SELF-CHECK:", "FAIL " + "; ".join(errors) if errors else "PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

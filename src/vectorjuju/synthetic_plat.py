"""Synthetic plat sheet generator for fixtures and ground truth.

Ported from docproc's ``prototypes/plat_extraction.py`` (``cmd_sheets`` and its
geometry/formatting helpers). Produces project-authored, non-real plat sheets
so tests and the docling prototype have known-good input with recoverable
ground truth -- no accuracy claim about real-world plats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PAGE_W, PAGE_H = 612.0, 792.0
SCALE_PT_PER_FT = 0.85
FONT, FONT_SIZE = "Helvetica", 7.5
RENDER_DPI = 200

# Parcel corners in feet, local survey coordinates. Chosen for a spread of label
# rotation angles -- flat/shallow/steep/near-vertical must all be represented.
PARCEL_FT = [
    (0.0, 0.0),
    (200.0, 8.0),
    (262.0, 96.0),
    (250.0, 215.0),
    (120.0, 252.0),
    (10.0, 170.0),
]
# Edge index -> radius in feet. These edges are drawn as circular arcs and get a
# curve-table row instead of a bearing/distance label.
CURVE_EDGES = {2: 190.0, 4: 240.0}
# Edge index -> offset in points for a parallel non-boundary line (right-of-way
# or adjacent tie), so a nearest-line rule has a real distractor to reject.
OFFSET_LINES = {0: 10.0, 5: 10.0}


def dms(angle_deg: float) -> str:
    d = int(angle_deg)
    m_full = (angle_deg - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60)
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return f"{d}°{m:02d}'{s:02d}\""


def bearing_distance(p0, p1) -> tuple[str, float]:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    az = math.degrees(math.atan2(dx, dy)) % 360.0
    if az <= 90:
        ns, ew, ang = "N", "E", az
    elif az <= 180:
        ns, ew, ang = "S", "E", 180 - az
    elif az <= 270:
        ns, ew, ang = "S", "W", az - 180
    else:
        ns, ew, ang = "N", "W", 360 - az
    return f"{ns} {dms(ang)} {ew}", math.hypot(dx, dy)


def label_rotation(p0, p1) -> float:
    """Rotation that keeps text reading left-to-right, in (-90, 90]."""
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    return ang


def angle_bucket(rot: float) -> str:
    a = abs(rot)
    if a < 15:
        return "flat"
    if a < 45:
        return "shallow"
    if a < 75:
        return "steep"
    return "near-vertical"


def arc_points(p0, p1, radius_ft, n=48):
    """Circular arc from p0 to p1 bulging away from the parcel centre."""
    cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    radius = max(radius_ft, chord / 2 + 1e-6)
    h = math.sqrt(radius**2 - (chord / 2) ** 2)
    ux, uy = (p1[0] - p0[0]) / chord, (p1[1] - p0[1]) / chord
    centroid = (
        sum(p[0] for p in PARCEL_FT) / len(PARCEL_FT),
        sum(p[1] for p in PARCEL_FT) / len(PARCEL_FT),
    )
    # Two candidate centres; pick the one that bulges away from the centroid.
    for nx, ny in ((-uy, ux), (uy, -ux)):
        ox, oy = cx + nx * h, cy + ny * h
        mid = (
            ox + (cx - ox) / math.hypot(cx - ox, cy - oy) * radius,
            oy + (cy - oy) / math.hypot(cx - ox, cy - oy) * radius,
        )
        if math.dist(mid, centroid) > math.dist((cx, cy), centroid):
            break
    a0 = math.atan2(p0[1] - oy, p0[0] - ox)
    a1 = math.atan2(p1[1] - oy, p1[0] - ox)
    while a1 - a0 > math.pi:
        a1 -= 2 * math.pi
    while a0 - a1 > math.pi:
        a1 += 2 * math.pi
    pts = [
        (ox + radius * math.cos(a0 + (a1 - a0) * i / n), oy + radius * math.sin(a0 + (a1 - a0) * i / n))
        for i in range(n + 1)
    ]
    delta = abs(a1 - a0)
    return pts, radius, delta, radius * delta, chord


def ft_to_pt(p):
    return (60.0 + p[0] * SCALE_PT_PER_FT, 300.0 + p[1] * SCALE_PT_PER_FT)


def generate_sheet(out_dir: Path) -> dict:
    """Generate a synthetic plat sheet (PDF, TIFF, EXIF-rotated JPG) plus its
    ground truth, in ``out_dir``. Returns the ground truth dict (also written
    to ``out_dir / "ground_truth.json"``).
    """
    from reportlab.lib.colors import black
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "sheet.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(PAGE_W, PAGE_H))
    c.setStrokeColor(black)
    c.setFillColor(black)

    labels, segments, curves, offset_lines = [], [], [], []
    n = len(PARCEL_FT)
    for i in range(n):
        p0_ft, p1_ft = PARCEL_FT[i], PARCEL_FT[(i + 1) % n]
        p0, p1 = ft_to_pt(p0_ft), ft_to_pt(p1_ft)
        bearing, dist_ft = bearing_distance(p0_ft, p1_ft)
        rot = label_rotation(p0, p1)
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)

        if i in CURVE_EDGES:
            pts_ft, radius, delta, arc_len, chord = arc_points(p0_ft, p1_ft, CURVE_EDGES[i])
            pts = [ft_to_pt(p) for p in pts_ft]
            c.setLineWidth(1.4)
            path = c.beginPath()
            path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)
            c.drawPath(path)
            cid = f"C{len(curves) + 1}"
            curves.append(
                {
                    "id": cid,
                    "radius_ft": radius,
                    "length_ft": arc_len,
                    "delta": dms(math.degrees(delta)),
                    "chord_bearing": bearing,
                    "chord_ft": chord,
                }
            )
            text = cid
            # An arc's own pixels bulge away from the chord; label at the bulge.
            mid = ft_to_pt(pts_ft[len(pts_ft) // 2])
            segments.append({"id": f"S{i}", "kind": "curve", "curve_id": cid, "start_pt": p0, "end_pt": p1})
        else:
            c.setLineWidth(1.4)
            c.line(p0[0], p0[1], p1[0], p1[1])
            text = f"{bearing}  {dist_ft:.2f}'"
            segments.append({"id": f"S{i}", "kind": "straight", "start_pt": p0, "end_pt": p1})

        offset = 5.0
        if i in OFFSET_LINES:
            rr_ = math.radians(rot)
            d = OFFSET_LINES[i]
            perp = (-math.sin(rr_) * d, math.cos(rr_) * d)
            q0 = (p0[0] + perp[0], p0[1] + perp[1])
            q1 = (p1[0] + perp[0], p1[1] + perp[1])
            c.setLineWidth(0.6)
            c.line(q0[0], q0[1], q1[0], q1[1])
            offset_lines.append({"id": f"X{i}", "near_segment_id": f"S{i}", "start_pt": q0, "end_pt": q1})

        c.saveState()
        c.translate(*mid)
        c.rotate(rot)
        c.setFont(FONT, FONT_SIZE)
        c.drawCentredString(0.0, offset, text)
        c.restoreState()

        w = pdfmetrics.stringWidth(text, FONT, FONT_SIZE)
        corners = [
            (-w / 2, offset - FONT_SIZE * 0.22),
            (w / 2, offset - FONT_SIZE * 0.22),
            (w / 2, offset + FONT_SIZE * 0.76),
            (-w / 2, offset + FONT_SIZE * 0.76),
        ]
        rr = math.radians(rot)
        quad = [
            (mid[0] + x * math.cos(rr) - y * math.sin(rr), mid[1] + x * math.sin(rr) + y * math.cos(rr))
            for x, y in corners
        ]
        labels.append(
            {
                "id": f"L{i}",
                "kind": "curve_ref" if i in CURVE_EDGES else "straight_call",
                "text": text,
                "anchor_pt": list(mid),
                "rotation_deg": rot,
                "bucket": angle_bucket(rot),
                "segment_id": f"S{i}",
                "quad_pt": [list(p) for p in quad],
            }
        )

    # Monuments at each corner -- realistic distractors that must not be read as calls.
    c.setFont(FONT, 6)
    for i, p_ft in enumerate(PARCEL_FT):
        p = ft_to_pt(p_ft)
        c.circle(p[0], p[1], 2.2, stroke=1, fill=0)
        c.drawString(p[0] + 4, p[1] + 4, "IPF" if i % 2 == 0 else "IPS")

    # Curve table.
    header = ["CURVE", "RADIUS", "LENGTH", "DELTA", "CH. BEARING", "CHORD"]
    rows = [header] + [
        [
            cu["id"],
            f"{cu['radius_ft']:.2f}'",
            f"{cu['length_ft']:.2f}'",
            cu["delta"],
            cu["chord_bearing"],
            f"{cu['chord_ft']:.2f}'",
        ]
        for cu in curves
    ]
    col_w = [38, 52, 52, 62, 92, 52]
    tx, ty = 60.0, 250.0
    row_h = 14.0
    c.setLineWidth(0.7)
    for r, row in enumerate(rows):
        y = ty - r * row_h
        x = tx
        c.setFont(FONT, 7 if r else 7.5)
        for w_, cell in zip(col_w, row, strict=True):
            c.rect(x, y - row_h, w_, row_h, stroke=1, fill=0)
            c.drawString(x + 3, y - row_h + 4, cell)
            x += w_
    table_bbox = [tx, ty - len(rows) * row_h, tx + sum(col_w), ty]
    c.setFont(FONT, 8)
    c.drawString(tx, ty + 6, "CURVE TABLE")

    # North arrow, scale bar, title block -- more distractor text.
    c.setLineWidth(1.2)
    c.line(520, 690, 520, 730)
    c.line(520, 730, 515, 720)
    c.line(520, 730, 525, 720)
    c.setFont(FONT, 9)
    c.drawCentredString(520, 675, "N")
    c.setLineWidth(1.0)
    c.line(60, 180, 60 + 100 * SCALE_PT_PER_FT, 180)
    for k in range(3):
        x = 60 + k * 50 * SCALE_PT_PER_FT
        c.line(x, 180, x, 175)
    c.setFont(FONT, 7)
    c.drawString(60, 166, "0        50       100 FEET")
    c.rect(360, 60, 192, 90, stroke=1, fill=0)
    c.setFont(FONT, 8.5)
    c.drawString(368, 134, "SYNTHETIC PLAT OF SURVEY")
    c.setFont(FONT, 7)
    for k, line in enumerate(
        [
            "PROJECT-AUTHORED TEST SHEET",
            "NOT A REAL PLAT / NOT A SURVEY",
            "vectorjuju synthetic fixture",
            "SCALE: 1\" = 100'",
        ]
    ):
        c.drawString(368, 118 - k * 11, line)
    c.showPage()
    c.save()

    # Raster variants from the same source, so all three media share one truth.
    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(str(pdf_path))[0]
    upright = page.render(scale=RENDER_DPI / 72.0).to_pil().convert("RGB")
    upright.save(out_dir / "sheet.tif", format="TIFF")
    # Store the JPEG physically rotated and tag Orientation=6, so a reader that
    # ignores EXIF lands in a different coordinate system than one that honours it.
    rotated = upright.rotate(90, expand=True)
    exif = rotated.getexif()
    exif[274] = 6
    rotated.save(out_dir / "sheet.jpg", format="JPEG", quality=88, exif=exif.tobytes())

    ground_truth = {
        "note": "Project-authored synthetic ground truth. No real plat content.",
        "page_size_pt": [PAGE_W, PAGE_H],
        "scale_pt_per_ft": SCALE_PT_PER_FT,
        "labels": labels,
        "segments": segments,
        "offset_lines": offset_lines,
        "curves": curves,
        "curve_table_cells": rows,
        "curve_table_bbox_pt": table_bbox,
        "distractor_text": [
            "IPF",
            "IPS",
            "N",
            "0        50       100 FEET",
            "CURVE TABLE",
            "SYNTHETIC PLAT OF SURVEY",
            "PROJECT-AUTHORED TEST SHEET",
            "NOT A REAL PLAT / NOT A SURVEY",
            "vectorjuju synthetic fixture",
            "SCALE: 1\" = 100'",
        ],
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    return ground_truth

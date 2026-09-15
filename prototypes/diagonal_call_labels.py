"""THROWAWAY prototype for issue #8 -- do not import, extend, or ship.

How does vectorjuju read bearing and distance labels drawn along diagonal
lines? Issue #4 found every docling OCR engine returns axis-aligned boxes with
no per-label angle and rotates only in quarter turns; issue #7 confirmed label
*content* survives rotation but the axis-aligned boxes over-cover. The
mitigation proposed there: OpenCV deskews a crop per label before OCR.

This prototype measures that mitigation on the project-authored synthetic
sheet, scored against its own ``ground_truth.json``:

* ``page-*`` -- prior-art style full-page OCR, boxes associated to labels.
  This is the baseline comparable to issue #7's docling numbers.
* ``crop-*`` -- OpenCV traces the line under each label, takes the deskew
  direction from that traced segment, cuts an upright band around the label,
  and OCRs the band alone.

Two deliberate methodological choices, both carried from issue #7:

* In the synthetic run a label's *position* comes from planted truth (issue
  #7's ``match_label`` did the same); only the deskew *direction* comes from
  OpenCV. Scoring the crop path this way isolates reading quality from
  localization quality. The private run locates labels from full-page OCR
  instead, since a real sheet has no truth.
* Docling is not run full-page here -- issue #7 measured exactly that. Only
  the new crop path uses it.

A verdict isolates digits/letters from unit marks, because losing the marks is
the systemic failure docproc and issue #7 both hit: ``text_only`` means every
digit and letter is right and only punctuation (foot/inch/degree marks,
spacing) differs.

Run on the synthetic sheet (macOS, docling models already cached):

    uv run --with 'docling[ocrmac]' --with opencv-python --with ocrmac \\
        python prototypes/diagonal_call_labels.py --out /private/tmp/vj8

Private plat (aggregate counts only -- no text, crop, or coordinate is printed
or written):

    uv run --with 'docling[ocrmac]' --with opencv-python --with ocrmac \\
        python prototypes/diagonal_call_labels.py \\
        --private /path/to/private_plat.png

Self-check without OCR or network:

    uv run --with opencv-python python prototypes/diagonal_call_labels.py --self-check
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from PIL import Image

MARKS = "°'\""
PAGE_PSM, CROP_PSM = 11, 7  # tesseract: sparse page, then single text line
TRACE_MAX_DIM = 2400  # Hough runs on a downscaled copy; crops come from the full raster
UNASSOCIATED_PX = 90.0  # label anchor farther than this from any traced segment: report
BEARING_RE = re.compile(r"[NS]\s*\d+.{0,3}\d+.{0,3}\d+.{0,3}[EW]")

_PUNCT = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "′": "'",
    "″": '"',
    "º": "°",
    "–": "-",
    "—": "-",
}


def norm_text(s: str) -> str:
    # _PUNCT first: NFKC folds the ordinal indicator to a letter "o" and splits
    # the double prime into two primes, so nothing running after it can undo either.
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().upper()


def alnum_text(s: str) -> str:
    """Digits, letters, and the decimal point -- the 'are the marks the only thing
    wrong?' view. Losing the period is a 100x distance error, not a mark loss."""
    return re.sub(r"[^0-9A-Z.]", "", norm_text(s))


def mark_counts(s: str) -> dict[str, int]:
    return {m: s.count(m) for m in MARKS}


def classify(recovered: str, truth: str) -> dict:
    """Verdict for one recovered label string against its planted truth."""
    if not recovered.strip():
        verdict = "missed"
    elif recovered == truth:
        verdict = "exact"
    elif norm_text(recovered) == norm_text(truth):
        verdict = "normalized"
    elif alnum_text(recovered) == alnum_text(truth):
        verdict = "text_only"
    else:
        ratio = difflib.SequenceMatcher(None, norm_text(recovered), norm_text(truth)).ratio()
        verdict = "garbled" if ratio >= 0.55 else "missed"
    # Marks are counted on normalized text for the same reason the verdict is:
    # a prime is an apostrophe, so raw counts report a loss the verdict calls fine.
    got, want = mark_counts(norm_text(recovered)), mark_counts(norm_text(truth))
    return {
        "verdict": verdict,
        "recovered": recovered,
        "lost_marks": [m for m in MARKS if got[m] < want[m]],
    }


def bounds(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def point_seg_dist(p, s0, s1) -> float:
    vx, vy = s1[0] - s0[0], s1[1] - s0[1]
    l2 = vx * vx + vy * vy
    if l2 == 0:
        return math.dist(p, s0)
    t = max(0.0, min(1.0, ((p[0] - s0[0]) * vx + (p[1] - s0[1]) * vy) / l2))
    return math.dist(p, (s0[0] + t * vx, s0[1] + t * vy))


def fold180(a: float) -> float:
    """Direction modulo 180 degrees -- a line has no arrowhead."""
    return (a + 90.0) % 180.0 - 90.0


def quad_to_px(quad_pt, gt: dict, px_size) -> list[tuple[float, float]]:
    """Planted quads are PDF points, bottom-left origin; rasters are top-left."""
    sx = px_size[0] / gt["page_size_pt"][0]
    ph = gt["page_size_pt"][1]
    return [(x * sx, (ph - y) * sx) for x, y in quad_pt]


def load_raster(path: Path) -> tuple[Image.Image, dict]:
    """Raster only, EXIF-transposed first -- docling does not apply EXIF itself."""
    from PIL import ImageOps

    raw = path.read_bytes()
    stored = Image.open(path)
    orientation = stored.getexif().get(274)
    img = ImageOps.exif_transpose(stored).convert("RGB")
    meta = {
        "basename": path.name,
        "sha256_12": hashlib.sha256(raw).hexdigest()[:12],
        "byte_size": len(raw),
        "exif_orientation": orientation,
        "exif_transpose_applied": bool(orientation and orientation != 1),
        "pixel_size": list(img.size),
    }
    return img, meta


def trace_segments(img: Image.Image) -> list[tuple[float, float, float, float]]:
    """OpenCV Hough segments, coordinates in the image's own pixel frame.

    Parameters and the downscale-for-speed split carry over from the docproc
    prior art (``threshold=110``, ``minLineLength=55``, ``maxLineGap=6`` at
    200 DPI). Coordinates are scaled back so crop bands can be cut from the
    full-resolution raster.
    """
    import cv2
    import numpy as np

    w, h = img.size
    scale = min(1.0, TRACE_MAX_DIM / max(w, h))
    small = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR) if scale < 1 else img
    gray = np.array(small.convert("L"))
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(bw, 1, math.pi / 360, threshold=110, minLineLength=55, maxLineGap=6)
    arr = np.asarray(lines).reshape(-1, 4) if lines is not None else np.empty((0, 4))
    return [(float(x0) / scale, float(y0) / scale, float(x1) / scale, float(y1) / scale) for x0, y0, x1, y1 in arr]


def nearest_segment(anchor, segments) -> tuple[tuple | None, float]:
    best, best_d = None, math.inf
    for s in segments:
        d = point_seg_dist(anchor, s[:2], s[2:])
        if d < best_d:
            best, best_d = s, d
    return best, best_d


def warp_band(img: Image.Image, seg, center, half_len: float, half_h: float) -> Image.Image:
    """Deskew an upright band centred on ``center`` with the source x axis along
    the traced segment. No mirroring: the linear part is a rotation."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = seg
    seg_len = math.hypot(x1 - x0, y1 - y0)
    u = ((x1 - x0) / seg_len, (y1 - y0) / seg_len) if seg_len else (1.0, 0.0)
    n = (-u[1], u[0])
    w, h = round(2 * half_len), round(2 * half_h)
    src = [
        (center[0] - half_len * u[0] - half_h * n[0], center[1] - half_len * u[1] - half_h * n[1]),
        (center[0] + half_len * u[0] - half_h * n[0], center[1] + half_len * u[1] - half_h * n[1]),
        (center[0] - half_len * u[0] + half_h * n[0], center[1] - half_len * u[1] + half_h * n[1]),
    ]
    dst = [(0.0, 0.0), (float(w), 0.0), (0.0, float(h))]
    m = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    warped = cv2.warpAffine(
        np.asarray(img.convert("RGB")), m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    return Image.fromarray(warped)


def crop_geometry(quad_px, seg) -> tuple[float, float]:
    """Band size from the label's own extent projected onto the traced direction."""
    x0, y0, x1, y1 = seg
    seg_len = math.hypot(x1 - x0, y1 - y0)
    u = ((x1 - x0) / seg_len, (y1 - y0) / seg_len) if seg_len else (1.0, 0.0)
    n = (-u[1], u[0])
    ts = [p[0] * u[0] + p[1] * u[1] for p in quad_px]
    ns = [p[0] * n[0] + p[1] * n[1] for p in quad_px]
    along, across = max(ts) - min(ts), max(ns) - min(ns)
    return along / 2 + max(8.0, 0.35 * across), across / 2 + max(12.0, 0.8 * across)


def save_png(img: Image.Image, path: Path) -> Path:
    path = path.resolve()
    img.save(path, format="PNG")
    return path


def tesseract_bin() -> str | None:
    return shutil.which("tesseract")


def ocr_tesseract(png_path: Path, psm: int) -> dict:
    exe = tesseract_bin()
    if not exe:
        return {"available": False, "reason": "tesseract binary not on PATH"}
    proc = subprocess.run([exe, str(png_path), "stdout", "--psm", str(psm), "tsv"], capture_output=True, check=False)
    if proc.returncode != 0:
        return {"available": False, "reason": proc.stderr.decode("utf-8", "replace")[:300]}
    rows = [line.split("\t") for line in proc.stdout.decode("utf-8", "replace").splitlines()]
    header, data = rows[0], rows[1:]
    keys = ("left", "top", "width", "height", "conf", "text", "block_num", "par_num", "line_num")
    idx = {k: header.index(k) for k in keys}
    items = []
    for r in data:
        if len(r) <= idx["text"]:
            continue
        text, conf = r[idx["text"]].strip(), float(r[idx["conf"]])
        if not text or conf < 0:
            continue
        x, y = float(r[idx["left"]]), float(r[idx["top"]])
        items.append(
            {
                "text": text,
                "conf": conf / 100.0,
                "box_px": [x, y, x + float(r[idx["width"]]), y + float(r[idx["height"]])],
                "line_key": (int(r[idx["block_num"]]), int(r[idx["par_num"]]), int(r[idx["line_num"]])),
            }
        )
    version = (
        subprocess.run([exe, "--version"], capture_output=True, check=False)
        .stdout.decode("utf-8", "replace")
        .splitlines()
    )
    return {
        "available": True,
        "engine": version[0] if version else "tesseract",
        "config": f"--psm {psm} tsv",
        "text": " ".join(it["text"] for it in items),
        "items": items,
    }


def group_page_lines(items) -> list[dict]:
    """Tesseract gives word boxes whose block/paragraph/line ids group them into
    text lines. Vision returns already-merged per-region strings with no such id,
    so each is its own line -- grouping those by a missing id would collapse the
    whole sheet into one candidate the size of the page."""
    groups: dict[tuple, list] = {}
    lines = []
    for it in items:
        if "line_key" not in it:
            lines.append({"text": it["text"], "box_px": list(it["box_px"]), "line_key": None})
            continue
        groups.setdefault(it["line_key"], []).append(it)
    for key, words in groups.items():
        words.sort(key=lambda it: it["box_px"][0])
        lines.append(
            {
                "text": " ".join(w["text"] for w in words),
                "box_px": bounds(
                    [(w["box_px"][0], w["box_px"][1]) for w in words]
                    + [(w["box_px"][2], w["box_px"][3]) for w in words]
                ),
                "line_key": key,
            }
        )
    return lines


def ocr_ocrmac(img: Image.Image) -> dict:
    try:
        from ocrmac import ocrmac
    except Exception as exc:  # noqa: BLE001 -- prototype: any import failure means unavailable
        return {"available": False, "reason": repr(exc)}
    w, h = img.size
    raw = ocrmac.OCR(img, recognition_level="accurate").recognize()
    items = []
    for text, conf, bbox in raw:
        # Vision returns normalized (x, y, w, h) with a bottom-left origin.
        x, y, bw, bh = bbox
        items.append(
            {
                "text": text,
                "conf": float(conf),
                "box_px": [x * w, (1.0 - y - bh) * h, (x + bw) * w, (1.0 - y) * h],
            }
        )
    import importlib.metadata

    try:
        version = importlib.metadata.version("ocrmac")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "available": True,
        "engine": f"ocrmac {version} / Apple Vision",
        "text": " ".join(it["text"] for it in items),
        "items": items,
    }


_DOCLING = {}


def ocr_docling(png_path: Path) -> dict:
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # noqa: BLE001 -- prototype: any import failure means unavailable
        return {"available": False, "reason": repr(exc)}
    import importlib.metadata

    if "conv" not in _DOCLING:
        _DOCLING["conv"] = DocumentConverter()
        _DOCLING["version"] = importlib.metadata.version("docling")
    try:
        doc = _DOCLING["conv"].convert(str(png_path)).document
    except Exception as exc:  # noqa: BLE001 -- one unreadable crop must not kill the run
        return {"available": True, "engine": f"docling {_DOCLING['version']}", "text": "", "error": repr(exc)}
    return {
        "available": True,
        "engine": f"docling {_DOCLING['version']}",
        "text": " ".join(t.text for t in doc.texts),
    }


def ocr_image(eng: str, pil: Image.Image, path: Path) -> dict:
    if eng == "docling":
        return ocr_docling(path)
    if eng == "tesseract":
        return ocr_tesseract(path, CROP_PSM)
    return ocr_ocrmac(pil)


def ocr_crop(eng: str, crop: Image.Image, workdir: Path, tag: str) -> dict:
    """A traced segment has no sign, so read the upright band and its 180-degree
    rotation and keep the read with more digits/letters. Prototype crutch: a
    production reader would take the label's own text direction from a direction
    classifier instead of OCRing every crop twice."""
    best = None
    for suffix, image in (("", crop), ("-rot180", crop.rotate(180))):
        res = ocr_image(eng, image, save_png(image, workdir / f"{tag}{suffix}.png"))
        if not res.get("available"):
            return res
        if best is None or len(alnum_text(res["text"])) > len(alnum_text(best["text"])):
            best = res
    return best


def axis_seg(center, half_len):
    """Horizontal stand-in for a traced segment: the un-deskewed control band.
    Same centre, same size, same crop; only the rotation differs."""
    return (center[0] - half_len, center[1], center[0] + half_len, center[1])


def label_center_px(label, gt, px_size):
    quad = quad_to_px(label["quad_pt"], gt, px_size)
    return (sum(p[0] for p in quad) / 4, sum(p[1] for p in quad) / 4)


def match_page_label(label, items, px_size, gt) -> str:
    """Prior-art association: item centres inside the planted (padded) quad,
    joined in reading order along the label's own direction."""
    quad = quad_to_px(label["quad_pt"], gt, px_size)
    bb = bounds(quad)
    pad = 0.004 * px_size[0]
    grown = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]
    ang = math.radians(-label["rotation_deg"])  # planted rotation is PDF-frame; raster y flips
    ux, uy = math.cos(ang), math.sin(ang)
    cands = [
        it
        for it in items
        if grown[0] <= (it["box_px"][0] + it["box_px"][2]) / 2 <= grown[2]
        and grown[1] <= (it["box_px"][1] + it["box_px"][3]) / 2 <= grown[3]
    ]
    cands.sort(
        key=lambda it: (it["box_px"][0] + it["box_px"][2]) / 2 * ux + (it["box_px"][1] + it["box_px"][3]) / 2 * uy
    )
    return " ".join(it["text"] for it in cands)


def run_synthetic(out_dir: Path, engine_names: list[str]) -> dict:
    from vectorjuju.synthetic_plat import generate_sheet

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "ground_truth.json").exists():
        generate_sheet(out_dir)
    gt = json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))
    labels = gt["labels"]
    img, meta = load_raster(out_dir / "sheet.tif")
    px_size = img.size
    print(f"sheet   : {meta['basename']} sha256={meta['sha256_12']} pixels={px_size} exif={meta['exif_orientation']}")

    segs = trace_segments(img)
    print(f"traced  : {len(segs)} Hough segments (opencv)")

    assoc, unassociated = {}, []
    for lab in labels:
        center = label_center_px(lab, gt, px_size)
        seg, dist = nearest_segment(center, segs)
        want_ang = -lab["rotation_deg"]  # PDF -> raster frame
        got_ang = math.degrees(math.atan2(seg[3] - seg[1], seg[2] - seg[0])) if seg else 0.0
        assoc[lab["id"]] = {"seg": seg, "dist": dist, "angle_err": fold180(got_ang - want_ang)}
        if dist > UNASSOCIATED_PX:
            unassociated.append(lab["id"])
    dists = [a["dist"] for a in assoc.values()]
    errs = [abs(a["angle_err"]) for a in assoc.values()]
    print(
        f"assoc   : anchor->segment mean {sum(dists) / len(dists):.1f}px max {max(dists):.1f}px | "
        f"|angle error| mean {sum(errs) / len(errs):.1f} deg max {max(errs):.1f} deg"
        + (f" | >{UNASSOCIATED_PX:.0f}px: {', '.join(unassociated)}" if unassociated else "")
    )

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="vj8-crops-") as tmp:
        workdir = Path(tmp)
        page_img = save_png(img, workdir / "page.png")

        page_engines = {}
        if "tesseract" in engine_names:
            page_engines["page-tesseract"] = ocr_tesseract(page_img, PAGE_PSM)
        if "ocrmac" in engine_names:
            page_engines["page-ocrmac"] = ocr_ocrmac(img)
        for name, res in page_engines.items():
            if not res.get("available"):
                print(f"{name}: UNAVAILABLE -- {res['reason']}")
                continue
            results[name] = {
                lab["id"]: classify(match_page_label(lab, res["items"], px_size, gt), lab["text"]) for lab in labels
            }
            print(f"{name}: {res['engine']} ({res.get('config', 'native')}) -- {len(res['items'])} items")

        for eng in engine_names:
            if eng not in ("docling", "tesseract", "ocrmac"):
                continue
            for variant in ("crop", "cropaxis"):
                name = f"{variant}-{eng}"
                per_label, note = {}, ""
                for lab in labels:
                    seg = assoc[lab["id"]]["seg"]
                    if seg is None:
                        per_label[lab["id"]] = classify("", lab["text"])
                        continue
                    quad = quad_to_px(lab["quad_pt"], gt, px_size)
                    center = (sum(p[0] for p in quad) / 4, sum(p[1] for p in quad) / 4)
                    half_len, half_h = crop_geometry(quad, seg)
                    band_seg = seg if variant == "crop" else axis_seg(center, half_len)
                    crop = warp_band(img, band_seg, center, half_len, half_h)
                    res = ocr_crop(eng, crop, workdir, f"{name}-{lab['id']}")
                    if not res.get("available"):
                        note = f"UNAVAILABLE -- {res['reason']}"
                        break
                    note = res["engine"]
                    per_label[lab["id"]] = classify(res["text"], lab["text"])
                if note.startswith("UNAVAILABLE"):
                    print(f"{name}: {note}")
                    continue
                results[name] = per_label
                print(f"{name}: {note}")

        print_report(labels, results)
        scores = {
            "note": "Throwaway issue-#8 prototype output. Synthetic sheet only.",
            "sheet": meta,
            "traced_segments": len(segs),
            "association": {k: {"dist_px": v["dist"], "angle_err_deg": v["angle_err"]} for k, v in assoc.items()},
            "results": results,
        }
        (out_dir / "diagonal_call_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"\nwrote {out_dir / 'diagonal_call_scores.json'}")
    return results


def print_report(labels, results: dict) -> None:
    buckets = ["flat", "shallow", "steep", "near-vertical", "ALL"]
    for path, per in results.items():
        print(f"\n{path}")
        print(
            f"  {'bucket':<15}{'n':>3}{'exact':>7}{'norm':>6}{'text-only':>11}{'garbled':>9}{'missed':>7}{'lost marks':>12}"
        )
        for b in buckets:
            labs = labels if b == "ALL" else [lab for lab in labels if lab["bucket"] == b]
            if not labs:
                continue
            verdicts = [per[lab["id"]]["verdict"] for lab in labs]
            lost = sum(len(per[lab["id"]]["lost_marks"]) for lab in labs)
            print(
                f"  {b:<15}{len(labs):>3}{verdicts.count('exact'):>7}{verdicts.count('normalized'):>6}"
                f"{verdicts.count('text_only'):>11}{verdicts.count('garbled'):>9}{verdicts.count('missed'):>7}{lost:>12}"
            )
        lost_by_mark = {m: sum(1 for lab in labels if m in per[lab["id"]]["lost_marks"]) for m in MARKS}
        print("  lost-mark breakdown: " + ", ".join(f"{m!r} x{n}" for m, n in lost_by_mark.items()))
        for lab in labels:
            r = per[lab["id"]]
            print(f"    {lab['id']:<4}{lab['bucket']:<15}{r['verdict']:<10}{lab['text']!r} -> {r['recovered']!r}")


def run_private(path: Path, engine_names: list[str], max_crops: int) -> None:
    """Aggregate counts only. No recognized text, crop image, or coordinate is
    printed or written: a real sheet is private, and there is no ground truth
    locally to score against."""
    img, meta = load_raster(path)
    px_size = img.size
    segs = trace_segments(img)
    print("PRIVATE PLAT -- aggregate counts only (no text, crop, or coordinate recorded)")
    print(f"raster  : sha256={meta['sha256_12']} pixels={px_size} exif={meta['exif_orientation']}")
    print(f"traced  : {len(segs)} Hough segments (opencv, downscaled to <= {TRACE_MAX_DIM}px for tracing)")

    with tempfile.TemporaryDirectory(prefix="vj8-priv-") as tmp:
        workdir = Path(tmp)
        page_img = save_png(img, workdir / "page.png")
        candidates: list[dict] = []
        for eng in ("ocrmac", "tesseract"):  # page OCR only *locates* candidates; docling is a crop engine here
            if eng not in engine_names:
                continue
            res = ocr_ocrmac(img) if eng == "ocrmac" else ocr_tesseract(page_img, PAGE_PSM)
            if not res.get("available"):
                print(f"page-{eng}: UNAVAILABLE -- {res['reason']}")
                continue
            lines = (
                group_page_lines(res["items"])
                if res.get("items")
                else [{"text": res["text"], "box_px": [0, 0, *px_size]}]
            )
            hits = [ln for ln in lines if BEARING_RE.search(norm_text(ln["text"]))]
            # Normalize before counting, as the crop path does below: a prime or an
            # ordinal indicator is the same mark as ' or °.
            counts = {m: sum(1 for h in hits if m in norm_text(h["text"])) for m in MARKS}
            mark_note = ", ".join(
                f"{counts[m]} with a {name}"
                for m, name in zip(MARKS, ("degree mark", "foot/minute mark", "seconds/inch mark"), strict=True)
            )
            print(f"page-{eng}: {res['engine']} -- {len(lines)} text lines, {len(hits)} bearing-shaped, {mark_note}")
            if not candidates:
                candidates = hits

        if not candidates:
            print("no bearing-shaped candidates from page OCR -- nothing to deskew")
            return
        # A label crop is small relative to the sheet; anything page-sized is a
        # locator failure (a merged region), not a label.
        page_w, page_h = px_size
        plausible = [c for c in candidates if c["box_px"][2] - c["box_px"][0] <= 0.3 * page_w]
        plausible = [c for c in plausible if c["box_px"][3] - c["box_px"][1] <= 0.06 * page_h]
        if len(plausible) != len(candidates):
            print(f"dropped {len(candidates) - len(plausible)} implausible (page-sized) candidates")
        candidates = plausible[:max_crops]
        print(f"deskewed crops: {len(candidates)} bearing-shaped page candidates, angle from nearest traced segment")
        print(
            "full-page reads of those candidates: "
            f"{sum(1 for c in candidates if BEARING_RE.search(norm_text(c['text'])))}/{len(candidates)} bearing-shaped, "
            f"{sum(1 for c in candidates if all(m in norm_text(c['text']) for m in MARKS))}/"
            f"{len(candidates)} with all three marks"
        )

        for eng in engine_names:
            if eng not in ("docling", "tesseract", "ocrmac"):
                continue
            read_back, all_marks, distinct = 0, 0, set()
            for i, cand in enumerate(candidates):
                center = ((cand["box_px"][0] + cand["box_px"][2]) / 2, (cand["box_px"][1] + cand["box_px"][3]) / 2)
                seg, _ = nearest_segment(center, segs)
                if seg is None:
                    continue
                # Inflate along the line more than across it: a call may reach the
                # locator as two regions (bearing, then distance), and the deskewed
                # band is cheap to extend in the direction that matters.
                w = cand["box_px"][2] - cand["box_px"][0]
                h = cand["box_px"][3] - cand["box_px"][1]
                crop = warp_band(img, seg, center, w / 2 + 1.5 * h, h / 2 + 0.75 * h)
                res = ocr_crop(eng, crop, workdir, f"priv-{eng}-{i}")
                if not res.get("available"):
                    print(f"crop-{eng}: UNAVAILABLE -- {res['reason']}")
                    break
                text = norm_text(res["text"])
                if BEARING_RE.search(text):
                    read_back += 1
                    distinct.add(re.sub(r"[^0-9A-Z°'\"]", "", text))
                    if all(m in text for m in MARKS):
                        all_marks += 1
            else:
                print(
                    f"crop-{eng}: bearing-shaped again {read_back}/{len(candidates)}, "
                    f"all three marks present {all_marks}/{len(candidates)}, "
                    f"distinct full-mark strings {len({s for s in distinct if all(m in s for m in MARKS)})} "
                    f"(not deduplicated; no local ground truth)"
                )


def ink_angle(gray: Image.Image) -> float:
    """Direction of the dominant ink extent, modulo 180 degrees.

    Principal axis, not ``minAreaRect``: OpenCV's rectangle angle reports either
    of its two perpendicular edges depending on which it calls the width, which
    is the 90-degree ambiguity a line direction must not have.
    """
    import numpy as np

    ys, xs = np.nonzero(np.array(gray.convert("L")) < 128)
    if len(xs) < 10:
        return 0.0
    x, y = xs - xs.mean(), ys - ys.mean()
    cov = np.array([[x @ x, x @ y], [x @ y, y @ y]]) / len(xs)
    values, vectors = np.linalg.eigh(cov)
    vx, vy = vectors[:, int(np.argmax(values))]
    return fold180(math.degrees(math.atan2(vy, vx)))


def self_check() -> int:
    """One runnable check of the two pieces that can silently break: the verdict
    tiers and the deskew transform."""
    call = "N 35°09'59\" E  200.16'"
    assert classify(call, call)["verdict"] == "exact"
    assert classify("N 35°09'59\"  E 200.16'", call)["verdict"] == "normalized"
    assert classify("N 35°0959\" E  200.16'", call)["verdict"] == "text_only", (
        "dropped minute mark must stay a mark-only loss"
    )
    assert classify('N 35°09\'59" E  200.16"', call)["verdict"] == "text_only", (
        "foot read as inch must stay a mark-only loss"
    )
    assert classify("N 350959 E 200.16", call)["verdict"] == "text_only", (
        "marks replaced by spaces are still mark-only losses"
    )
    assert classify("S 3°21'59\" W  170.29*", call)["verdict"] in ("garbled", "missed")
    # That string scores under the 0.55 ratio cut, i.e. "missed", so the garbled
    # tier needs a case of its own -- deleting the tier would otherwise go unnoticed.
    assert classify("N 35°09'59\" E 200.16'XXX", call)["verdict"] == "garbled"
    assert classify("", call)["verdict"] == "missed"
    assert classify("ZEBRA", call)["verdict"] == "missed"

    import numpy as np
    from PIL import ImageDraw

    angle = 54.8
    flat = Image.new("L", (900, 900), 255)
    draw = ImageDraw.Draw(flat)
    draw.line((200, 450, 700, 450), fill=0, width=3)
    draw.text((330, 430), call, fill=0)
    rotated = flat.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    before = ink_angle(rotated)
    expected = -angle  # PIL rotates counterclockwise; raster y increases downward
    assert abs(fold180(before - expected)) < 3.0, f"fixture is not at {angle} deg: measured {before}"

    seg = (
        450.0 - 250.0 * math.cos(math.radians(before)),
        450.0 - 250.0 * math.sin(math.radians(before)),
        450.0 + 250.0 * math.cos(math.radians(before)),
        450.0 + 250.0 * math.sin(math.radians(before)),
    )
    crop = warp_band(rotated.convert("RGB"), seg, (450.0, 450.0), 250.0, 60.0)
    assert crop.size == (500, 120), crop.size
    after = ink_angle(crop)
    assert abs(fold180(after)) < 3.0, f"deskewed band is still at {after} deg"
    _, xs = np.nonzero(np.array(crop.convert("L")) < 128)
    assert len(xs) > 0.002 * 500 * 120, "deskewed band lost the label ink"
    assert xs.max() - xs.min() > 0.5 * crop.size[0], "band deskewed across the line, not along it"
    print("self-check ok: verdict tiers + deskew transform")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output dir for the synthetic run (default: a temp dir)")
    p.add_argument("--private", default=None, help="raster of a private plat; aggregate counts only")
    p.add_argument("--engines", default="docling,tesseract", help="comma list of docling,tesseract,ocrmac")
    p.add_argument("--max-private-crops", type=int, default=30)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        return self_check()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.private:
        run_private(Path(args.private), engines, args.max_private_crops)
        return 0
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vj8-"))
    run_synthetic(out_dir, engines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

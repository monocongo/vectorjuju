"""THROWAWAY prototype for issue #7 -- do not import, extend, or ship.

Go/no-go: does docling recover boundary-call text and curve-table cells well
enough to be vectorjuju's text layer? Runs docling on the project-authored
synthetic sheet (PDF, TIFF, EXIF-rotated JPEG) and scores it against the
sheet's own ground_truth.json on the axes from issue #7: verbatim call
recovery, curve-table cell recovery, rotated-label handling, pixel-space
mapping.

The PDF medium has a REAL embedded text layer (reportlab draws vector text) --
docling reads that natively, no OCR involved. That is not representative of a
scanned/raster plat. It is reported as a ceiling, not a measurement of
docling's OCR fitness. TIFF and JPEG force the OCR path (ocrmac on macOS,
since docling does not apply EXIF orientation itself -- this script transposes
JPEGs before handing them to docling) and are the real test.

Run:

    uv run --with 'docling[ocrmac]' python prototypes/docling_extraction.py \
        --out /private/tmp/docling-plat

`--with ocrmac` is macOS-only; omit it elsewhere (docling falls back to
another OCR engine per docs/research/docling-plat-capabilities.md).

For the private plat (never committed, aggregate-only):

    uv run --with 'docling[ocrmac]' python prototypes/docling_extraction.py \
        --private /path/to/private_plat.tif
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import unicodedata
from pathlib import Path

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
    s = unicodedata.normalize("NFKC", s)
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip().upper()


def bounds(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_quad_to_page_frame(quad_pt, gt_page_pt, page_size):
    """Ground-truth quads are PDF points, bottom-left origin, 612x792. Docling
    reports bbox in the same bottom-left convention but in its own notion of
    page size (native points for a PDF, ~pixel size for a raster with no DPI
    tag). Scale, no flip -- both are bottom-left.
    """
    sx = page_size[0] / gt_page_pt[0]
    sy = page_size[1] / gt_page_pt[1]
    return [(x * sx, y * sy) for x, y in quad_pt]


def docling_text_items(doc):
    items = []
    for t in doc.texts:
        if not t.prov:
            continue
        b = t.prov[0].bbox
        items.append({"text": t.text, "box": [b.l, min(b.t, b.b), b.r, max(b.t, b.b)]})
    return items


def match_label(label, items, page_size, gt_page_pt):
    quad = gt_quad_to_page_frame(label["quad_pt"], gt_page_pt, page_size)
    gt_box = bounds(quad)
    # Docling (unlike raw word-level OCR) already returns whole merged text
    # lines, so a candidate must actually OVERLAP the label's own box -- a
    # nearby distractor (a monument tag beside the label) must not be pulled
    # in just for sitting within some padded radius of it.
    pad = 0.003 * page_size[0]
    grown = [gt_box[0] - pad, gt_box[1] - pad, gt_box[2] + pad, gt_box[3] + pad]
    rr = math.radians(label["rotation_deg"])
    rr_x, rr_y = math.cos(rr), math.sin(rr)
    cands = [it for it in items if iou(it["box"], grown) > 0.0]
    cands.sort(key=lambda it: (it["box"][0] + it["box"][2]) / 2 * rr_x + (it["box"][1] + it["box"][3]) / 2 * rr_y)
    joined = " ".join(c["text"] for c in cands)
    truth = label["text"]
    if not cands:
        verdict = "missed"
    elif joined == truth:
        verdict = "exact"
    elif norm_text(joined) == norm_text(truth):
        verdict = "normalized"
    else:
        ratio = difflib.SequenceMatcher(None, norm_text(joined), norm_text(truth)).ratio()
        verdict = "garbled" if ratio >= 0.55 else "missed"
    union = (
        bounds([(c["box"][0], c["box"][1]) for c in cands] + [(c["box"][2], c["box"][3]) for c in cands])
        if cands
        else None
    )
    return {
        "verdict": verdict,
        "recovered": joined,
        "fragments": len(cands),
        "iou": iou(union, gt_box) if union else 0.0,
    }


def score_table(gt_cells, doc):
    if not doc.tables:
        return None
    tb = doc.tables[0].data
    grid = {(c.start_row_offset_idx, c.start_col_offset_idx): c.text for c in tb.table_cells}
    n_rows, n_cols = len(gt_cells), len(gt_cells[0])
    shape_match = tb.num_rows == n_rows and tb.num_cols == n_cols
    hits = 0
    total = n_rows * n_cols
    for r, row in enumerate(gt_cells):
        for c, cell in enumerate(row):
            got = grid.get((r, c), "")
            if norm_text(got) == norm_text(cell):
                hits += 1
    return {
        "shape_match": shape_match,
        "got_shape": (tb.num_rows, tb.num_cols),
        "want_shape": (n_rows, n_cols),
        "cell_hits": hits,
        "cell_total": total,
    }


def run_medium(name: str, path: Path, gt: dict) -> None:
    from docling.document_converter import DocumentConverter

    print("=" * 78)
    print(f"MEDIUM: {name} ({path.name})")
    print("=" * 78)
    conv = DocumentConverter()
    result = conv.convert(str(path))
    doc = result.document
    page = doc.pages[1]
    page_size = (page.size.width, page.size.height)
    items = docling_text_items(doc)
    print(f"page size (docling frame): {page_size} | text items: {len(items)} | tables: {len(doc.tables)}")

    results = {lab["id"]: match_label(lab, items, page_size, gt["page_size_pt"]) for lab in gt["labels"]}
    buckets = sorted({lab["bucket"] for lab in gt["labels"]})
    print(f"\n{'bucket':<15}{'n':>3}{'exact':>7}{'norm':>6}{'garbled':>9}{'missed':>8}{'mean_iou':>10}")
    for bucket in buckets:
        labs = [lab for lab in gt["labels"] if lab["bucket"] == bucket]
        v = [results[lab["id"]]["verdict"] for lab in labs]
        ious = [results[lab["id"]]["iou"] for lab in labs if results[lab["id"]]["verdict"] != "missed"]
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        print(
            f"{bucket:<15}{len(labs):>3}{v.count('exact'):>7}{v.count('normalized'):>6}"
            f"{v.count('garbled'):>9}{v.count('missed'):>8}{mean_iou:>10.2f}"
        )
    print("\nper-label (truth -> recovered):")
    for lab in gt["labels"]:
        r = results[lab["id"]]
        print(f"  {lab['id']} {lab['bucket']:<14} {r['verdict']:<10} {lab['text']!r} -> {r['recovered']!r}")

    tbl = score_table(gt["curve_table_cells"], doc)
    print("\nCURVE TABLE:")
    if tbl is None:
        print("  no table detected")
    else:
        print(f"  shape recovered: {tbl['shape_match']} (got {tbl['got_shape']}, want {tbl['want_shape']})")
        print(f"  cell text exact/normalized matches: {tbl['cell_hits']}/{tbl['cell_total']}")
    print()


def run_private(path: Path) -> None:
    from docling.document_converter import DocumentConverter

    print("=" * 78)
    print(f"PRIVATE PLAT (aggregate only, path not recorded): {path.suffix}")
    print("=" * 78)
    conv = DocumentConverter()
    result = conv.convert(str(path))
    doc = result.document
    n_bearing_like = sum(1 for t in doc.texts if re.search(r"[NSns]\s*\d+.{0,3}\d+.{0,3}\d+.{0,3}[EWew]", t.text))
    print(f"text items: {len(doc.texts)} | bearing-like: {n_bearing_like} | tables: {len(doc.tables)}")
    for tb in doc.tables:
        print(f"  table shape: {tb.data.num_rows}x{tb.data.num_cols}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="dir with sheet.pdf/tif/jpg + ground_truth.json (generated if absent)")
    p.add_argument("--private", default=None, help="local path to a private plat; aggregate-only report")
    args = p.parse_args()

    if args.private:
        run_private(Path(args.private))
        return 0

    import json

    out = Path(args.out) if args.out else Path("/private/tmp/docling-plat")
    if not (out / "ground_truth.json").exists():
        from vectorjuju.synthetic_plat import generate_sheet

        generate_sheet(out)
    gt = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    run_medium("PDF (native text layer -- NOT OCR, ceiling only)", out / "sheet.pdf", gt)
    run_medium("TIFF (OCR path)", out / "sheet.tif", gt)

    from PIL import Image, ImageOps

    jpg_norm = out / "sheet_jpg_normalized.png"
    ImageOps.exif_transpose(Image.open(out / "sheet.jpg")).convert("RGB").save(jpg_norm)
    run_medium("JPEG (OCR path, EXIF-transposed by us first)", jpg_norm, gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

> Carried from docproc [`docs/plats/00-recon.md`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/plats/00-recon.md) at `3342063b5f001038020cf0ae6b14ac0c8de0c966`.

# Plat pipeline — Phase 0 recon

Read-only survey of what `docproc` already has, what the plat pipeline can reuse,
and what it must add. Every row below comes from a file read or a command run in
this session; nothing is carried over from the task brief unverified.

Scope decisions taken with the user before this document was written:

- `reference/legacy_binder.py` does not exist in this repo (no `reference/`
  directory at all), so the Phase 1 architecture review of it is dropped. Its
  defect checklist is carried forward as design requirements instead, and each
  claim is confirmed or refuted by experiment below.
- New code lives in `prototypes/`, extending the issue #108 work, not
  in `src/docproc/`.
- Dependencies go in a new `[project.optional-dependencies].plats` extra.
- The text source is the #108 prototype's own OCR path, not Docling. See
  "Docling reuse" below for why.

## Capability table

| Capability | Where it already lives | Version | Reuse or add |
|---|---|---|---|
| PDF → raster at declared DPI | `prototypes/plat_extraction.py` (`_load_normalized`, pypdfium2) | pypdfium2 5.13.0 in lock, **not installed** | **Reuse.** Add pypdfium2 to the `plats` extra |
| JPG/TIFF ingest with EXIF orientation | `prototypes/plat_extraction.py` (`PIL.ImageOps.exif_transpose`) | pillow 12.3.0, base dep | **Reuse as-is** |
| OCR with pixel bounding boxes | `prototypes/plat_extraction.py` (`_ocr_ocrmac`, Apple Vision; `_ocr_tesseract`) | ocrmac via `--with`; tesseract via `PATH` | **Reuse as-is** |
| Synthetic plat + ground truth | `prototypes/plat_extraction.py` (`cmd_sheets`) | reportlab 4.2 (dev group) | **Reuse.** Already plants curves, right-of-way distractors, a curve table and rotated labels |
| Label rotation normalized to (-90, 90] | `prototypes/plat_extraction.py` (`label_rotation`) | — | **Reuse.** This is exactly the P4 MTEXT rotation rule |
| Point-to-segment distance | `prototypes/plat_extraction.py` (`point_seg_dist`) | — | **Reuse** |
| Unicode/punctuation text normalization | `prototypes/plat_extraction.py` (`norm_text`) | — | **Reuse and extend** with OCR-confusion mapping |
| Line/contour detection | `prototypes/plat_extraction.py` (`_geometry`, HoughLinesP + findContours) | opencv-python 5.0.0.93 | **Replace.** Hough gives segments, not traced centerlines; see Experiment 4 |
| PDF layout with measured bboxes | `src/docproc/adapters/docling_parser.py` | docling 2.124.0, extra, **not installed** | **Do not use.** See below |
| DXF writing | — | — | **Add** `ezdxf` |
| Skeletonization, circle fit, RANSAC | — | — | **Add** `scikit-image` |
| k-d tree, assignment problem | — | scipy 1.18.1 in lock (transitive only) | **Add** explicit `scipy` pin |

## Dependency reality

`numpy`, `opencv-python`, `scipy` and `shapely` appear in `uv.lock`, which reads
as "already available" but is not. Parsing the lock shows they are present
**only** as transitive dependencies of the `docling` extra:

```
numpy         <- accelerate, docling-ibm-models, opencv-python, pandas,
                 rapidocr, scipy, shapely, torchvision, transformers
opencv-python <- rapidocr
shapely       <- rapidocr
docproc deps  : boto3, duckdb, openai, pillow, pydantic, pypdf, sqlalchemy, typer
docproc extras: {'docling': ['docling'], 'gemini': ['google-genai']}
```

None of them is installed in `.venv` (base `uv sync` pulls no extras), and
relying on another extra's transitive tree would couple the plat pipeline to
rapidocr's pin. They are pinned explicitly in the new `plats` extra.

Absent from the lock entirely: **`ezdxf`**, **`scikit-image`**,
**`opencv-contrib-python`**.

## The four questions

### 1. What coordinate space do the text bboxes arrive in?

Two different answers, which is why the question matters.

**The repo's Docling path emits bottom-left PDF points.**
`src/docproc/adapters/docling_parser.py:277` converts every box explicitly:

```python
def _bbox_to_pdf_points(bbox: Any, page_height: float) -> tuple[float, float, float, float]:
    bottom_left = bbox.to_bottom_left_origin(page_height)
```

So `coord_origin` is normalized to **BOTTOMLEFT**, in **PDF points**, not
pixels of a rasterized page. Docling itself may emit either origin; the adapter
absorbs that at its boundary.

**The path this pipeline actually uses emits top-left pixels.** ocrmac boxes are
converted from Apple Vision's normalized bottom-left space to top-left pixels in
`plat_extraction._ocr_ocrmac`:

```python
# Vision returns normalized (x, y, w, h) with a bottom-left origin.
x, y, bw, bh = bbox
boxes.append({"box_px": [x * w, (1.0 - y - bh) * h, (x + bw) * w, (1.0 - y) * h], ...})
```

and tesseract's are natively top-left pixels (`:447`). Both are in the **upright
raster's** pixel space, the same space OpenCV contours come out of — which is
the whole reason to use them rather than Docling.

**Consequence for P1.** The pipeline standardizes on
`raster px, top-left origin` and asserts it at ingestion. The brief's
`img_height - y` flip is correct *only* for that space; it would silently
corrupt Docling's PDF-point boxes, which are already bottom-left and are not in
pixels at all. A future Docling-backed text source owes a
`PDF points → page pixels at DPI` transform on top of the origin handling, and
must not reuse the pixel path's flip.

### 2. Is `opencv-contrib-python` installed?

No. Absent from `uv.lock`, and confirmed at runtime under `opencv-python 5.0.0`:

```
opencv       5.0.0
cv2.ximgproc present? False
```

`cv2.ximgproc.thinning` is therefore unavailable. **`skimage.morphology.skeletonize`
is the centerline path**, which is also why `scikit-image` earns its place —
it supplies the circle fit and RANSAC too.

Installing `opencv-contrib-python` instead was rejected: it conflicts with the
`opencv-python` that the `docling` extra already pulls (both provide `cv2`), and
it would still not give us the circle fit.

### 3. Which NumPy major version?

**2.5.3** (lock says 2.5.2; the resolver picked 2.5.3 for the experiment
environment — either way, 2.x).

`np.cross` on 2-D vectors does not warn under 2.x. It **raises**:

```
--- EXPERIMENT 1: np.cross on 2-D vectors ---
  RESULT: ValueError: Both input arrays must be (arrays of) 3-dimensional vectors,
          but they are 2 and 2 dimensional instead.
  scalar replacement a[0]*b[1]-a[1]*b[0] = 1.0
```

This is a hard failure, not a deprecation, so any ported code using it is dead
on arrival. Every 2-D cross product in the pipeline uses the explicit scalar
`ax*by - ay*bx`.

### 4. Which `ezdxf` major version?

None — `ezdxf` is absent from the lock, so the pin is ours to choose. Pinning
`>=1.3,<2` puts us past 1.0, which means **`set_placement()`**, not the
superseded `Text.set_pos()`. Recorded here because the version is a decision,
not an observation.

## Experiments

Run with `uv run --no-project --with ...`; `pyproject.toml` is unchanged at this
checkpoint. Scripts are in the session scratchpad and reproduced in `HANDOFF.md`.

### Experiment 2 — `cKDTree.query` sentinels (scipy 1.18.1)

```
  query=(1.0, 0.0) ub=5.0:     dist=1.0     idx=0  idx==n(3)? False  isinf=False
  query=(500.0, 500.0) ub=5.0: dist=inf     idx=3  idx==n(3)? True   isinf=True
  query=(500.0, 500.0) ub=inf: dist=693.108 idx=2  idx==n(3)? False  isinf=False
  k=2 far query:               dist=array([inf, inf]) idx=array([3, 3])
```

Confirmed: a miss under `distance_upper_bound` returns `dist=inf` **and**
`idx == n`, one past the end. Indexing the point array with it raises
`IndexError`, so the guard is required — and `idx == n` is the check to write,
since it holds for both the scalar and the `k>1` array form.

### Experiment 4 — double-edge tracing

A single 3px-wide horizontal stroke drawn at y=100:

```
  RETR_LIST      n_contours=1 pts=6 distinct_y_in_contour=[98, 100, 102]
  RETR_EXTERNAL  n_contours=1 pts=6 distinct_y_in_contour=[98, 100, 102]
  skeletonize    on-pixels=319 distinct_y=[100]  -> single 1px centerline
```

**Confirmed.** `findContours` returns the stroke *outline*, spanning y 98..102 —
both edges of one line — and `RETR_EXTERNAL` does not help, because the problem
is not nesting. `skeletonize` collapses it to a single 1px centerline at exactly
y=100.

Verdict on the brief's alternative: `RETR_EXTERNAL` + stroke-width compensation
is rejected. It cannot recover the centerline of a stroke whose width varies
(dashed linetypes, anti-aliased raster edges, arcs drawn as polylines) without
re-deriving the medial axis, which is what skeletonize already does correctly.

### Experiment 5 — `approxPolyDP(closed=False)` on a closed contour

```
  closed=True  -> 2 vertices: [(38, 100), (362, 100)]
  closed=False -> 4 vertices: [(40, 98), (40, 102), (360, 102), (360, 98)]
```

**Confirmed, and worse than the brief states.** Neither setting is usable on a
stroke outline. `closed=True` collapses the outline rectangle to a degenerate
2-point line; `closed=False` traces all four corners of the outline, and its
first and last vertices are an artifact of wherever `findContours` happened to
start the trace — not a real endpoint.

The finding is that `closed` is the wrong knob. Simplification must run on
skeleton traces, which are genuinely open paths, and then `closed=False` is
correct because the topology actually is open.

### Experiment 6 — line-vs-curve classification

An L-corner of two perfectly straight segments versus a true 90° arc of similar
extent:

```
  L-corner (2 straight segs)   max_chord_dev=  70.71  circle_max_resid=  13.224  fit_radius= 59.08  turn_consistent=True
  true 90deg arc               max_chord_dev=  29.29  circle_max_resid=   0.000  fit_radius=100.00  turn_consistent=True
```

**Confirmed, decisively.** The L-corner's chord deviation (70.71) is *larger*
than the real arc's (29.29), so no threshold on deviation alone can separate
them — any cutoff that admits the arc admits the corner. The circle residual
separates them cleanly: 13.224 versus 0.000.

**One part of the brief's proposed fix is refuted.** The consistent-turn-direction
check does **not** discriminate here: both shapes report `turn_consistent=True`,
because an L-corner turns exactly once, and one turn is trivially consistent.
The check still earns its place against S-curves and noisy traces that reverse
curvature, but it must not be relied on for the corner case. **The normalized
circle-fit residual is the discriminator**, with the turn check as a secondary
guard.

### Experiments 1, 3, 4–7 (regex layer)

`reference/legacy_binder.py` does not exist, so the patterns were reconstructed
verbatim from the brief's own descriptions and tested directly.

| Claim | Verdict | Evidence |
|---|---|---|
| Quadrant `([E W])` holds a literal space | **Confirmed** (with a correction) | see below |
| Minutes mandatory → `N 45° E` fails | **Confirmed** | `'N 45° E'` → NO MATCH |
| `(\d{1,3})` accepts out-of-range degrees | **Confirmed** | `"N 450°30' E"` → groups `('N','450','30','E')`; `"N 999°99' E"` also matches |
| `\b` after `'` | **Confirmed** | `"100.00'E"` → matches; `"100.00' "` and `"100.00'"` → NO MATCH |
| `l` alternative conflates arc length | **Confirmed** | `"L=150.00'"` → `150.00` as a *distance*; `'24l'` → matches |
| `RADIUS_REGEX` mandates `=` | **Confirmed** | `'R 150.00'`, `'RAD. 150'`, `"RADIUS = 150.00'"` → all NO MATCH |
| No OCR-confusion normalization | **Confirmed** | `"N 450 30' E"`, `"5 45°30' E"`, `'R = 150.00"'` → all NO MATCH |

**Correction on the quadrant claim.** The obvious test cases all *look* fine:

```
  "N 45°30' E"  -> MATCH quadrant='E'
  "N 45°30' W"  -> MATCH quadrant='W'
```

because `\s*` is greedy — with a real quadrant letter present it consumes the
space and `[E W]` lands on the letter. The class only reveals itself when no
quadrant letter follows, at which point `\s*` backtracks to zero width and
`[E W]` consumes the space instead:

```
  "N 45°30' 100.00'"       -> MATCH quadrant=' '   <-- captured a SPACE
  "S 12°05' X"             -> MATCH quadrant=' '   <-- captured a SPACE
  "N 45°30'  "             -> MATCH quadrant=' '   <-- captured a SPACE
```

The first of those is an ordinary bearing-and-distance call. So the defect is
real and its failure mode is worse than "does not match": it produces a bearing
with a blank quadrant, which is a silent wrong answer where a no-match would
have been an honest one.

## Docling reuse

The brief instructed reusing Docling for OCR/layout. Three verified facts make
that the wrong call for this pipeline, and the user confirmed the divergence:

1. **PDF bytes only.** `DoclingParser.parse(self, pdf_bytes: bytes)`
   (`docling_parser.py:88`). There is no raster entry point, and plats arrive as
   flattened JPGs as often as PDFs. A JPG would have to be wrapped into a PDF
   first, purely to satisfy the signature.
2. **Wrong coordinate space.** Bottom-left PDF points, not raster pixels
   (`docling_parser.py:277`). Every box would need a points→pixels transform at
   the declared DPI before it could mask contours drawn in pixel space.
3. **Cost.** Docling is an optional extra precisely because it pulls a
   torch-backed stack and downloads model weights on first use; the adapter's own
   docstring records runs taking "seconds to tens of seconds". It is not
   installed.

The #108 prototype already chose ocrmac and tesseract for these reasons and
**measured** them. That measurement is the reason to trust the path.

This is recorded rather than discarded: a future Docling-backed text source is a
legitimate option, and it owes exactly one thing — a
`PDF points (bottom-left) → raster px (top-left) at DPI` transform at its
boundary, mirroring what `_bbox_to_pdf_points` does in the other direction.

## Proposed `plats` extra

Not yet applied to `pyproject.toml` — that is CHECKPOINT 2.

```toml
plats = [
    "ezdxf>=1.3,<2",
    "opencv-python>=5.0,<6",
    "numpy>=2.5,<3",
    "scipy>=1.18,<2",
    "scikit-image>=0.25,<1",
    "pypdfium2>=5.13,<6",
]
```

`ocrmac` stays a `--with` flag: it is macOS-only and cannot sit in a portable
extra. `tesseract` is the cross-platform fallback the prototype already handles.

## API notes for implementation

- **scikit-image 0.26 deprecations.** `CircleModel()` + `.estimate()` + `.params`
  all emit `FutureWarning` and are removed in 2.2. Use
  `CircleModel.from_estimate(...)` and the `.center` / `.radius` attributes.
- **OpenCV 5 return shapes.** `HoughLinesP` returns `Nx4` where cv4 returned
  `Nx1x4`; `plat_extraction._geometry` already handles both. Any new cv call needs
  the same check rather than an assumed shape.

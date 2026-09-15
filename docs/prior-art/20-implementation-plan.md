> Carried from docproc [`docs/plats/20-implementation-plan.md`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/plats/20-implementation-plan.md) at `3342063b5f001038020cf0ae6b14ac0c8de0c966`.

# Plat pipeline — implementation plan

Four stages of one data flow, in one module. Written against the verified facts
in [`00-recon.md`](00-recon.md); every requirement below traces to an experiment
there or to a line in the existing prototype.

## Layout

| Path | Status | Responsibility |
|---|---|---|
| `prototypes/plat_vectorize.py` | new | P1–P4 plus CLI |
| `prototypes/tests/test_plat_vectorize.py` | new | Gate tests and the regex harness |
| `prototypes/plat_extraction.py` | reused (lint/format fixes; render-DPI parameter) | Ingest, OCR, fixture generation, shared helpers |
| `pyproject.toml` | modified | `[project.optional-dependencies].plats` |

One module rather than a package: the four phases share one context object and
one data flow, and a prototype gains nothing from four files and an `__init__`.

Tests live in `prototypes/tests/`, outside `pyproject.toml`'s
`testpaths = ["tests"]`, so `uv run pytest -q` and CI stay exactly as they are
today. They are run explicitly.

## Shared types

```python
@dataclass(frozen=True)
class PageContext:
    """Everything spatial derives from this. No magic pixel constants anywhere else.

    Coordinate space: `raster px`, top-left origin, y increasing downward --
    the space both OCR engines and OpenCV contours report in. Asserted at
    ingestion, never re-inferred.
    """
    dpi: float
    img_width: int
    img_height: int
    feet_per_pixel: float | None      # None until calibrated
    source_sha256: str
    source_media_type: str

    @property
    def stroke_px(self) -> float:       # nominal stroke width at this DPI
    @property
    def simplify_tol_px(self) -> float: # RDP tolerance, DPI-derived
    @property
    def bind_radius_px(self) -> float:  # k-d tree search radius, DPI-derived
    def with_scale(self, feet_per_pixel: float) -> "PageContext": ...


@dataclass(frozen=True)
class ParsedCall:
    raw_text: str
    bearing: float | None      # decimal degrees, azimuth 0-360, None if absent
    distance: float | None     # feet
    radius: float | None       # feet
    arc_length: float | None   # feet -- never merged into `distance`
    type: Literal["line_dimension", "curve_dimension", "unknown"]


@dataclass(frozen=True)
class CircleFit:
    center_px: tuple[float, float]
    radius_px: float
    max_residual_px: float
    turn_consistent: bool


@dataclass
class Primitive:
    kind: Literal["line", "curve"]
    points_px: np.ndarray          # (N, 2) float, top-left px
    fit: CircleFit | None
    calls: list[tuple[ParsedCall, tuple[float, float]]]  # (call, insertion px)
```

`arc_length` is a separate field precisely because the brief's `l` alternative
collapsed it into `distance` (Experiment 5 in the recon doc: `"L=150.00'"` read
as a 150-foot chord).

---

## P1 — Ingest and scale standardization

### Signatures

```python
def load_page(path: Path, dpi: float) -> tuple[Image.Image, PageContext]: ...
def deskew(img: Image.Image, ctx: PageContext) -> tuple[Image.Image, float]: ...
def px_to_cad(pt: tuple[float, float], ctx: PageContext) -> tuple[float, float]: ...
def cad_to_px(pt: tuple[float, float], ctx: PageContext) -> tuple[float, float]: ...
def calibrate_feet_per_pixel(
    primitives: Sequence[Primitive],
    ctx: PageContext,
    override: float | None = None,
) -> tuple[float, dict]: ...  # (feet_per_pixel, diagnostics)
```

### Behaviour

`load_page` wraps `plat_extraction._load_normalized`, which already handles PDF
(pypdfium2 at a declared DPI) and JPG/TIFF (PIL EXIF transpose), and already
returns the sha256 and media type that `PageContext` carries. Nothing is
reimplemented.

**The one transform chain, stated once:**

```
raster px (top-left, y down)  ->  CAD units (bottom-left, y up, feet)

    cad_x = px_x * feet_per_pixel
    cad_y = (img_height - px_y) * feet_per_pixel
```

`px_to_cad` raises if `ctx.feet_per_pixel is None`. Geometry is extracted and
bound entirely in pixels; conversion happens once, at emission, after
calibration. That ordering is what keeps `feet_per_pixel` derivable *from* the
bound geometry.

**Calibration order**, first that succeeds:

1. Explicit `--scale` override.
2. RANSAC regression (`skimage.measure.ransac`, `LineModelND` through the
   origin) of parsed `distance` values against measured pixel lengths of the
   segments they bound. Needs >= 3 bound distance calls.
3. Parsed scale bar, if a `N FEET` label binds to a horizontal tick run.
4. **Fail loudly** with exit code 3.

There is no pixel fallback. Emitting pixel-unit DXF with annotations reading
`150.00'` is the silent-wrong-answer failure this pipeline exists to avoid.

### Tests

- `test_transform_round_trip` — `cad_to_px(px_to_cad(p))` within 1e-6 across a
  sweep of corners and interior points, at 2 DPIs x 2 `feet_per_pixel` values.
- `test_px_to_cad_requires_calibration` — raises when `feet_per_pixel is None`.
- `test_origin_flip` — a point at the top of the raster maps to a *large* CAD y,
  and one at the bottom maps to ~0. Guards against the flip being dropped.
- `test_jpg_and_pdf_agree` — the same sheet as PDF and as EXIF-rotated JPG
  produces the same `img_height` and the same CAD position for a planted corner.
  This is the EXIF trap `plat_extraction.cmd_sheets` deliberately plants.

**Gate:** round-trip within 1e-6.

---

## P2 — Structural isolation and classification

### Signatures

```python
def build_text_mask(boxes: Sequence[dict], ctx: PageContext) -> np.ndarray: ...
def binarize(img: Image.Image, mask: np.ndarray) -> np.ndarray: ...
def trace_skeleton(bw: np.ndarray) -> list[np.ndarray]: ...
def split_at_corners(path: np.ndarray, ctx: PageContext) -> list[np.ndarray]: ...
def merge_dashes(runs: Sequence[np.ndarray], ctx: PageContext) -> list[np.ndarray]: ...
def classify(points: np.ndarray, ctx: PageContext) -> tuple[str, CircleFit | None]: ...
def extract_primitives(img: Image.Image, boxes: Sequence[dict], ctx: PageContext) -> list[Primitive]: ...
```

### Pipeline order

```
OCR boxes --> dilate by stroke_px --> exclusion mask
                                          |
gray --> Otsu(THRESH_BINARY_INV) --> apply mask --> skeletonize
     --> trace_skeleton --> merge_dashes --> split_at_corners
     --> RDP simplify (simplify_tol_px) --> classify --> [Primitive]
```

### Requirements carried forward from the dropped Phase 1 review

Each cites the recon experiment that settled it.

- **Double-edge tracing** (Exp 4, confirmed). `findContours` returns the stroke
  *outline* — a 3px stroke at y=100 traces `y ∈ {98,100,102}` under both
  `RETR_LIST` and `RETR_EXTERNAL`. Fix: `skimage.morphology.skeletonize` to a
  1px centerline (`y ∈ {100}`), then walk it. `RETR_EXTERNAL` + stroke-width
  compensation is rejected — it cannot handle variable stroke width without
  re-deriving the medial axis.

- **`trace_skeleton` is a pixel-graph walk, not `findContours`.** Running
  `findContours` on a 1-px skeleton re-introduces the same doubling: the contour
  runs out along the path and back. Instead: 8-neighbour degree per on-pixel;
  endpoints are degree 1, junctions degree >= 3; walk degree-2 chains from every
  endpoint and junction; any remaining unvisited cycle (a closed parcel loop has
  no endpoints at all) is walked from an arbitrary seed until it returns.

- **Corner splitting is what makes the L-corner problem tractable** (Exp 6).
  A closed parcel traces as *one* path; the gate's five primitives only exist
  after splitting. Turning angle is accumulated over a DPI-scaled window; a
  corner is a local maximum above ~30°, where an arc distributes low turning
  evenly. Splitting at corners means the classifier never sees an L.

- **Classification is the circle residual, not chord deviation** (Exp 6,
  confirmed decisively). The L-corner's `max_chord_dev` is 70.71 against the real
  arc's 29.29 — *larger*, so no deviation threshold can separate them. Normalized
  circle residual gives 13.224 vs 0.000. Rule: `curve` iff
  `max_residual / radius` is below threshold **and** the fit radius is sane.

- **The consistent-turn check is a secondary guard only** (Exp 6, refuted as a
  primary discriminator). Both the L-corner and the true arc report
  `turn_consistent=True`, because one turn is trivially consistent. It is kept to
  reject S-curves and noisy traces that reverse curvature, and it is recorded in
  `CircleFit`, but it never decides the corner case.

- **Text pixels never become geometry.** OCR boxes, dilated by one `stroke_px`,
  are zeroed before skeletonization, so glyph strokes cannot enter the trace.

- **`approxPolyDP` `closed` matches real topology** (Exp 5, confirmed worse than
  stated). On a stroke outline neither setting is usable — `closed=True`
  collapsed the rectangle to 2 degenerate vertices, `closed=False` traced all 4
  outline corners with start/end an artifact of the trace origin. Simplification
  runs on skeleton traces, which are genuinely open, so `closed=False` is then
  correct for what it actually is.

- **No `np.cross` on 2-D** (Exp 1: it *raises* under NumPy 2.5.3, it does not
  warn). Every 2-D cross product is the explicit scalar `ax*by - ay*bx`.

- **No magic pixel constants.** `simplify_tol_px` and the size filter derive from
  `ctx.dpi`. The brief's `tolerance=2.0` is DPI-blind.

- **Size filter on length, not area.** `contourArea` is meaningless for an open
  polyline. Filter on arc length and bbox diagonal, both DPI-scaled, combined
  with `or` where the brief used `and`.

- **Dashed linetypes merge before classification.** Easement and setback dashes
  fragment into many short collinear runs; `merge_dashes` joins runs that are
  collinear within tolerance and whose endpoints are within a DPI-scaled gap.

- **scikit-image 0.26 API.** `CircleModel.from_estimate(...)` with `.center` /
  `.radius`; the bare constructor, `.estimate()` and `.params` are deprecated and
  removed in 2.2.

### Tests

- `test_gate_five_primitives` — the gate, below.
- `test_l_corner_not_a_curve` — the Exp 6 L-corner classifies `line` after
  splitting, and its chord deviation is asserted to exceed the real arc's, so
  the test fails if anyone reintroduces a deviation-only rule.
- `test_skeleton_is_single_width` — a 3px stroke traces one centerline, not two
  edges. This is the double-edge regression.
- `test_text_excluded` — a raster of glyphs and no lines yields zero primitives.
- `test_dashes_merge` — a dashed line becomes one primitive, not eleven.
- `test_tolerance_scales_with_dpi` — the same figure at 2 DPIs gives the same
  primitive count.

**Gate:** a minimal fixture drawn directly with PIL — 4 straight edges and 1 arc
closing the figure, on white, nothing else — yields exactly 5 primitives,
exactly 1 classified `curve`, and no duplicate parallel edge within one stroke
width. Drawn in the test rather than via `cmd_sheets`, whose sheet also carries
6 edges, 2 arcs, offset lines, monuments, a curve table, a north arrow, a scale
bar and a title block.

---

## P3 — Spatial graph

### Signatures

```python
def normalize_ocr(raw: str) -> str: ...
def parse_call(raw: str) -> ParsedCall: ...
def densify(primitives: Sequence[Primitive], ctx: PageContext) -> tuple[np.ndarray, np.ndarray]: ...
def bind(
    primitives: Sequence[Primitive], boxes: Sequence[dict], ctx: PageContext
) -> tuple[list[Primitive], list[dict], dict]: ...  # (bound, unbound_text, diagnostics)
```

### Parsing requirements

All seven confirmed in the recon doc; each gets a named regression case.

| Requirement | Failure it prevents |
|---|---|
| Quadrant is `([EW])`, never `([E W])` | `"N 45°30' 100.00'"` matched with `quadrant=' '` — a silent blank quadrant on an ordinary call |
| Minutes and seconds optional | `'N 45° E'` was dropped entirely |
| Degrees 0–90, minutes/seconds 0–59, validated after match | `"N 999°99' E"` was accepted |
| Distance terminated by lookahead, not `\b` after `'` | `"100.00' "` and `"100.00'"` failed while `"100.00'E"` matched — exactly backwards |
| `L=` captured to `arc_length`, never `distance`; no bare `l` alternative | `"L=150.00'"` read as a 150-ft chord; `'24l'` matched |
| Radius accepts `R=150.00`, `R 150.00`, `RAD. 150`, `RADIUS = 150.00'` | three of four forms were dropped |
| `normalize_ocr` runs first: `0`↔`°`, `5`↔`S`, `1`↔`I`, `"`↔`''` | `"N 450 30' E"`, `"5 45°30' E"`, `'R = 150.00"'` all dropped |

`normalize_ocr` extends `plat_extraction.norm_text` (which already does NFKC and
smart-punctuation folding) rather than replacing it. Confusion substitutions are
positional, not global — a `0` is only read as `°` where a degree sign is
syntactically expected, so `100.00` is not corrupted.

### Binding requirements

- **Densified samples, not centroids.** The brief's centroid k-d tree binds a
  label near the end of a long boundary to a wrong short segment. `densify`
  samples every polyline at a DPI-derived spacing and returns
  `(points, sample_index -> primitive_index)`, so a `cKDTree` query approximates
  true point-to-segment distance.
- **Search radius is `ctx.bind_radius_px`**, derived from DPI. The brief's
  `distance_upper_bound=75.0` is meaningless across DPIs.
- **Sentinel guard** (Exp 2, confirmed on scipy 1.18.1): a miss returns
  `dist=inf` **and** `idx == n`, one past the end, for both scalar and `k>1`
  forms. The guard checks `idx == n`, which holds in both.
- **Angular gate.** OCR box orientation versus segment bearing must agree within
  a tolerance, compared modulo 180° since text reads either way along a line.
- **Semantic gate.** A `radius` or `arc_length` call may not bind to a primitive
  classified `line`; a `curve_dimension` may not bind to a `line`.
- **Assignment, not greedy.** Greedy nearest-neighbour is one-directional and
  lets one label steal a segment another fits better.
  `scipy.optimize.linear_sum_assignment` over a cost matrix of gated distances.
  The greedy result is computed alongside and both are reported under
  `diagnostics`, so the difference is measured rather than asserted.
- **Unbound text is bucketed, never force-bound.** Anything failing a gate or
  exceeding the radius goes to `unbound_text`.

### Tests

- `test_regex_harness` — a table of every case in the recon doc, asserting the
  fixed patterns get them right. Doubles as the runbook's regex harness.
- `test_no_bare_l_alternative` — `'PARCEL 5'`, `'LOT 12'`, `'24l'` parse to no
  distance.
- `test_centroid_vs_sample_binding` — a long boundary and a short segment with a
  label near the long one's end; the centroid rule binds wrong, the sample rule
  binds right. Both are computed so the test states the improvement.
- `test_kdtree_sentinel_guard` — a query with no neighbour in range returns
  unbound rather than raising `IndexError`.
- `test_assignment_beats_greedy` — the deliberately ambiguous label pair.
- `test_semantic_gate` — a `Radius = 150.00'` call does not bind to a `line`.

**Gate:** on the `cmd_sheets` fixture, every label binds to its planted
`segment_id`; the two `OFFSET_LINES` right-of-way distractors — planted at
`plat_extraction.OFFSET_LINES` precisely to steal calls — carry **zero** boundary calls;
and the distractor text (`IPF`, `IPS`, `N`, `CURVE TABLE`, the scale bar) lands
in `unbound_text`.

That zero is the honest measure of this work. Issue #108's Axis 4 showed a naive
nearest-line rule attaches calls to right-of-way lines silently, and its
"planted geometry" scores were an upper bound no production system has.

---

## P4 — DXF and JSON emission

### Signatures

```python
def write_dxf(primitives: Sequence[Primitive], ctx: PageContext, path: Path) -> dict: ...
def write_json(primitives: Sequence[Primitive], unbound: Sequence[dict], ctx: PageContext, path: Path) -> None: ...
def sort_key(p: Primitive) -> tuple: ...
```

### Layers

`PL_BOUNDARY_LINE`, `PL_BOUNDARY_CURVE`, `PL_DIM_TEXT`, `PL_UNBOUND_TEXT`,
`PL_DEBUG`.

### Requirements

- Lines → `LWPOLYLINE`. Curves → `ARC` with the real center and radius from the
  P2 `CircleFit` where the fit holds, `SPLINE` otherwise. Not dense polylines.
- Dimensions → `MTEXT`, rotation normalized to the half-open interval
  `(-90, 90]`. `plat_extraction.label_rotation` (`:85`) already implements
  exactly this normalization — reused, not rewritten.
- `doc.units = ezdxf.units.FT`, so `$INSUNITS` is not left unitless. ezdxf is
  pinned `>=1.3`, so text placement uses `set_placement()`, not the 1.0-superseded
  `set_pos()`.
- Text height is a function of `ctx.dpi` and `feet_per_pixel`, not a literal.
- **Deterministic ordering.** Primitives sorted by a stable geometric key
  (rounded min-x, min-y, kind) before emission. `unbound_text` is a separate
  top-level key, never appended to the primitive list — the brief's
  `spatial_graph.extend(geo_metadata)` put unbound text first and made ordering
  unstable.

### JSON contract

Fixed by the brief; existing keys keep their names and types. Extra keys only
under `diagnostics`.

```json
{
  "spatial_graph": [
    {
      "type": "curve",
      "points": [[105.2, 400.1], [120.4, 430.7]],
      "inferred_angle": 41.5,
      "associated_elements": [{
        "spatial_insertion": [130.0, 440.0],
        "parsed_data": {
          "raw_text": "Radius = 150.00'",
          "bearing": null, "distance": null, "radius": 150.0,
          "type": "curve_dimension"
        }
      }]
    }
  ],
  "unbound_text": [],
  "diagnostics": {}
}
```

### Tests

- `test_dxf_audit_clean` — `ezdxf.recover.readfile` reopens with zero audit
  errors.
- `test_layer_counts_match_json` — per-layer entity counts equal the JSON.
- `test_rotation_interval` — every MTEXT rotation in `(-90, 90]`.
- `test_curves_are_arcs` — a fitted curve emits `ARC`, not a 48-vertex polyline.
- `test_units_set` — `doc.units` is `FT` and `$INSUNITS` is non-zero.
- `test_deterministic_bytes` — two runs on one input produce identical JSON
  bytes.

**Gate:** clean audit, matching counts, every rotation in interval, identical
bytes across runs.

---

## CLI

```
python prototypes/plat_vectorize.py INPUT --out DIR [--dpi 200]
                                    [--scale FEET_PER_PIXEL] [--ocr auto|ocrmac|tesseract]
```

Artifacts: `DIR/plat.dxf`, `DIR/plat.json`, `DIR/overlay.png` (debug).

| Exit | Meaning |
|---|---|
| 0 | Success |
| 2 | Input unreadable or unsupported media type |
| 3 | Scale calibration failed and no `--scale` given |
| 4 | No geometry survived extraction |

Phase-boundary counts print to stderr — primitives in/out, bind rate, unbound
count — matching how `plat_extraction.py` already reports. No `structlog`: it is
not a repo dependency, and a prototype's progress lines do not justify a
lockfile entry.

## Out of scope

No `src/docproc/` changes, no adapter registration, no `ProcessingDefinition`
wiring — the brief did not specify how a vectorizer participates in digest
identity. No Docling code path; the transform it would owe is recorded in
`00-recon.md`. No georeferencing, no multi-page handling, no real-plat corpus.
All synthetic, all local.

> Carried from docproc [`docs/plats/HANDOFF.md`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/plats/HANDOFF.md) at `3342063b5f001038020cf0ae6b14ac0c8de0c966`. Its embedded prototype source has been replaced below with links to the pinned files — see "3. Consolidated code".
>
> Written for docproc's persisted, review-required application. vectorjuju's `convert()` is a stateless function with no Registry, persistence, review workflow, or UI (see the map's Out of scope) — every "Survey Review Revision" or "reviewer-built Course" mention below is docproc-only context, not applicable here. The extraction pipeline design (P1-P4 stages), coordinate-space contract, failure modes, and test approach still apply.

# Plat pipeline — handoff

Self-contained pickup point for a fresh session with no prior context on
this work. Read this file, `00-recon.md`, and `20-implementation-plan.md` in
that order if more depth is needed than this document gives.

## Relationship to accepted decisions

This pipeline is a prototype. Three of its mechanisms are not authorized for
production by accepted decisions (read those before promoting any of it into
`src/docproc/`):

- **Fitted scale.** `calibrate_scale` fits `feet_per_pixel` by RANSAC.
  [Survey geometry and provenance model](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/decisions/survey-geometry-provenance-model.md)
  explicitly does not authorize "similarity or affine fitting, fitted scale,
  force-closure, or snapping".
- **Automatic call-to-line association.** `bind` assigns calls to primitives
  itself. The same decision makes association "a reviewer-built Course, not an
  extractor output", and this repository's own #108 findings
  ([local-plat-extraction.md](local-plat-extraction-prototype.md)) say it
  "must remain a reviewable candidate, never an automatic determination".
- **DXF straight from extraction.** `write_dxf` emits machine output. In the
  accepted model, DXF is derived geometry: a deterministic function of one
  Survey Review Revision. The entity and layer mapping belongs to the open
  [DXF export contract](https://github.com/monocongo/docproc/issues/112), so the
  `PL_*` layer names here are provisional.

The `plats` optional extra and the synthetic-only fixtures do follow
[Survey-plat Phase 2 scope](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/decisions/survey-plat-scope.md). Vocabulary for
production work (Boundary Call, Course, Parcel, Survey Review Revision) is in
the root `CONTEXT.md` and those decisions, not in this document.

## 1. Executive technical summary

**Data flow.** `prototypes/plat_vectorize.py` extends the issue
#108 prototype (`prototypes/plat_extraction.py`, ingest + OCR + synthetic
fixture generation, reused with lint/format fixes and a render-DPI parameter) with four stages:

```
load_page (P1)          plat_extraction._load_normalized wrapped in a
                         PageContext -- DPI, dims, uncalibrated scale.

extract_primitives (P2) text-box exclusion mask -> Otsu -> skeletonize ->
                         pixel-graph trace -> dash merge -> corner split ->
                         classify (circle-fit residual) -> simplify (lines
                         only; curves keep full resolution for classify).

bind (P3)                densified k-d tree samples -> radius + semantic
                         gates -> linear_sum_assignment (not greedy) ->
                         bound Primitives + unbound_text.

calibrate_scale (P1)     RANSAC (skimage, through-origin ratio model) of
                         parsed distance calls vs. bound-segment pixel
                         length, or --scale override. No pixel fallback.

write_dxf / write_json  (P4) layered DXF (LWPOLYLINE/ARC/MTEXT) + the JSON
                         contract, both from the same calibrated primitives.
```

**Coordinate-space contract.** Exactly two spaces, one conversion point:

- **raster px** — top-left origin, y down. What OCR boxes
  (`plat_extraction._ocr_ocrmac`/`_ocr_tesseract`) and OpenCV/skimage
  geometry are already in. All of P2 and P3 work here.
- **CAD units** — bottom-left origin, y up, feet. What DXF and the JSON
  `points` fields are in.
- `PageContext.px_to_cad`/`cad_to_px` are the *only* conversion point, and
  only run at P4 emission, after calibration. `PageContext.feet_per_pixel`
  is `None` until then; P1–P3 never assume a value for it, and P4 raises
  `ValueError` if it's still `None`.
- Docling's own bbox space (`src/docproc/adapters/docling_parser.py`) is
  bottom-left **PDF points**, not raster pixels — a third space this module
  never touches. See `00-recon.md`, "Docling reuse."

**Scale-calibration strategy.** `calibrate_scale`: explicit `--scale`
override wins outright; otherwise RANSAC regression of every bound
`line_dimension` call's `distance_ft` against the measured pixel length of
the line primitive it's bound to, using a custom `_OriginRatioModel` (feet =
k·pixels, no intercept — the physical relationship passes through the
origin, which `skimage`'s general 2-parameter line model doesn't enforce).
Fewer than 3 usable pairs raises `CalibrationError` (CLI exit 3). No pixel
fallback exists anywhere — emitting pixel-unit DXF whose own text reads
`150.00'` is the silent-wrong-answer failure this whole exercise exists to
avoid.

## 2. Module map

| Path | Status | Responsibility |
|---|---|---|
| `prototypes/plat_vectorize.py` | new | `PageContext` + transforms (P1), structural isolation + classification (P2), call parsing + spatial binding (P3), scale calibration, DXF/JSON emission (P4), CLI |
| `prototypes/tests/test_plat_vectorize.py` | new | Every gate test, the regex harness, calibration tests — run explicitly, outside `pyproject.toml`'s `testpaths` |
| `prototypes/plat_extraction.py` | reused (lint/format fixes; `_load_normalized` takes the render DPI) | `_load_normalized` (PDF/JPG/TIFF ingest), `_ocr_ocrmac`/`_ocr_tesseract`, `cmd_sheets` (synthetic fixture + ground truth), `label_rotation`, `norm_text`, `point_seg_dist` |
| `pyproject.toml` | modified | New `[project.optional-dependencies].plats` extra: `ezdxf`, `opencv-python`, `numpy`, `scipy`, `scikit-image`, `pypdfium2` |
| `docs/plats/00-recon.md` | new | Phase 0 recon table + the 4 brief questions + regex/geometry experiments |
| `docs/plats/20-implementation-plan.md` | new | Per-phase signatures, tests, gates (CHECKPOINT 2 plan) |
| `docs/plats/HANDOFF.md` | new, this file | Pickup point |
| `docs/plats/CLAUDE.md` | new | Conventions that must survive context compaction |

## 3. Consolidated code

Full source lives in docproc, not copied here:

- [`prototypes/plat_vectorize.py`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/prototypes/plat_vectorize.py) — `PageContext` + transforms (P1), structural isolation + classification (P2), call parsing + spatial binding (P3), scale calibration, DXF/JSON emission (P4), CLI.
- [`prototypes/tests/test_plat_vectorize.py`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/prototypes/tests/test_plat_vectorize.py) — every gate test, the regex harness, calibration tests.
- [`prototypes/plat_extraction.py`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/prototypes/plat_extraction.py) — the reused ingest/OCR/synthetic-fixture module `plat_vectorize.py` extends.

### Coordinate space contract

From `plat_vectorize.py`'s module docstring. Every function taking or returning
a plain `(x, y)` tuple documents which of these two spaces it is in; a
`PageContext` is the only thing allowed to convert between them:

- **raster px** — top-left origin, y increasing downward. The space OCR boxes
  (`plat_extraction._ocr_ocrmac` / `_ocr_tesseract`) and OpenCV contours are
  already in. All P2/P3 geometry work happens here.
- **CAD units** — bottom-left origin, y increasing upward, in feet. What DXF
  and the output JSON's `points` fields are in. Conversion happens once, at P4
  emission, via `PageContext.px_to_cad`.

`PageContext.feet_per_pixel` is `None` until calibrated (see `calibrate_scale`
below). Nothing in P1-P3 may assume a value for it; P4 refuses to emit
without one.

Docling's own bbox convention is bottom-left *PDF points*, not raster pixels —
a third space this module never touches. See `00-recon.md`, "Docling reuse".

## 4. Runbook

Every command below was actually executed against this branch; output is
real, not illustrative.

### Install

```console
$ uv sync --extra plats
Resolved 160 packages in 3.12s
 ~ docproc==0.1.0 (from file:///.../prototype+local-plat-extraction)
 + ezdxf==1.4.4
 + numpy==2.5.2
 + opencv-python==5.0.0.93
 + scikit-image==0.26.0
 + scipy==1.18.1
 + pypdfium2==5.13.0
 ... (7 more transitive)
```

### Gate tests (outside `testpaths`, run explicitly)

```console
$ uv run --extra plats pytest prototypes/tests -q
70 passed, 18 warnings in 0.65s
```

The 18 warnings are all `skimage.measure.fit`'s own internal
`scipy.spatial.minkowski_distance` deprecation notice (scikit-image 0.26
calling an already-deprecated SciPy 1.18 function inside `CircleModel`) —
nothing in this module's own code triggers them.

### Normal suite and lint, unaffected by the new extra

```console
$ uv run pytest -q
1173 passed, 16 skipped in 4.70s

$ uv run ruff check .
[]
```

### Generate a synthetic fixture + ground truth

```console
$ uv run --extra plats --with reportlab --with ocrmac \
    python prototypes/plat_extraction.py sheets --out /tmp/plat_run
wrote /tmp/plat_run/sheet.pdf, sheet.tif, sheet.jpg, ground_truth.json in /tmp/plat_run
  L0  flat               2.3  N 87°42'34" E  200.16'
  L1  steep             54.8  N 35°09'59" E  107.65'
  L2  near-vertical    -84.2  C1
  L3  shallow          -15.9  N 74°06'46" W  135.16'
  L4  shallow           36.7  C2
  L5  near-vertical     86.6  S 3°21'59" W  170.29'
bucket spread: {'flat': 1, 'steep': 1, 'near-vertical': 2, 'shallow': 2}
image size px: (1700, 2200) | jpeg stored rotated with EXIF orientation 6
```

### Single-file run — PDF

```console
$ uv run --extra plats --with ocrmac \
    python prototypes/plat_vectorize.py /tmp/plat_run/sheet.pdf --out /tmp/plat_run/dxf --dpi 200
[P1] loaded sheet.pdf: 1700x2200px @ 200.0 dpi, sha256=e99a4afaa399
[P1] OCR (ocrmac / Apple Vision): 34 text regions
[P2] 33 primitives: 30 line, 3 curve
[P3] bind_rate=0.18 n_bound=6 n_unbound=28
[P1] calibrated feet_per_pixel=0.688812 (ransac)
[P4] wrote /tmp/plat_run/dxf/plat.dxf (39 entities) and /tmp/plat_run/dxf/plat.json
```

### Single-file run — JPG (EXIF-rotated)

```console
$ uv run --extra plats --with ocrmac \
    python prototypes/plat_vectorize.py /tmp/plat_run/sheet.jpg --out /tmp/plat_run/dxf-jpg --dpi 200
[P1] loaded sheet.jpg: 1700x2200px @ 200.0 dpi, sha256=292e86f33f8c
[P1] OCR (ocrmac / Apple Vision): 34 text regions
[P2] 34 primitives: 31 line, 3 curve
[P3] bind_rate=0.18 n_bound=6 n_unbound=28
[P1] calibrated feet_per_pixel=0.511940 (ransac)
[P4] wrote /tmp/plat_run/dxf-jpg/plat.dxf (40 entities) and /tmp/plat_run/dxf-jpg/plat.json
```

Both media types run end-to-end and audit clean (see "DXF audit" below). The
primitive-count and calibration-value differences between PDF and JPG are
real and expected: JPEG is lossy (re-encoded at quality 88 in
`cmd_sheets`), so its edges anti-alias slightly differently than the PDF's
direct pypdfium2 rasterization, changing a few marginal classify/split
decisions. Neither is "wrong" — see the curve-table contamination limitation in section 6 for why the
*calibrated value itself* is noisy on this specific noisy real-OCR run.

### DXF audit + determinism check

```console
$ uv run --extra plats python -c "
import ezdxf.recover
doc, aud = ezdxf.recover.readfile('/tmp/plat_run/dxf/plat.dxf')
print('audit errors:', len(aud.errors))"
audit errors: 0

$ uv run --extra plats --with ocrmac python prototypes/plat_vectorize.py /tmp/plat_run/sheet.pdf --out /tmp/plat_run/a --dpi 200
$ uv run --extra plats --with ocrmac python prototypes/plat_vectorize.py /tmp/plat_run/sheet.pdf --out /tmp/plat_run/b --dpi 200
$ diff /tmp/plat_run/a/plat.json /tmp/plat_run/b/plat.json && echo "DETERMINISTIC: byte-identical"
DETERMINISTIC: byte-identical
```

### Exit codes

```console
$ uv run --extra plats python prototypes/plat_vectorize.py /tmp/nonexistent.pdf --out /tmp/xx
error: unreadable or unsupported input: /tmp/nonexistent.pdf
$ echo $?
2

$ uv run --extra plats --with ocrmac python prototypes/plat_vectorize.py /tmp/blank.jpg --out /tmp/xx2
[P1] OCR (ocrmac / Apple Vision): 0 text regions
[P2] 0 primitives: 0 line, 0 curve
error: no geometry survived extraction
$ echo $?
4

$ uv run --extra plats --with ocrmac python prototypes/plat_vectorize.py /tmp/no_labels.jpg --out /tmp/xx3
error: scale calibration failed: only 0 bound distance call(s) with a measurable line primitive -- need >= 3 for RANSAC regression, and no --scale override was given
$ echo $?
3

$ uv run --extra plats --with ocrmac python prototypes/plat_vectorize.py /tmp/no_labels.jpg --out /tmp/xx4 --scale 0.5
[P1] calibrated feet_per_pixel=0.500000 (override)
[P4] wrote /tmp/xx4/plat.dxf (5 entities) and /tmp/xx4/plat.json
$ echo $?
0
```

### Regex test harness

Every claim in `00-recon.md`'s defect table has a named regression case;
run just those:

```console
$ uv run --extra plats pytest prototypes/tests -q \
    -k "test_quadrant_never or test_whole_degree or test_out_of_range or \
        test_distance_boundary or test_arc_length_not_conflated or \
        test_radius_forms or test_ocr_confusion or test_full_call_round_trip or \
        test_curve_dimension_type or test_curve_ref_label_binds_to_curve"
17 passed, 53 deselected in 0.49s
```

## 5. Integration notes

**CLI:**

```
python prototypes/plat_vectorize.py INPUT --out DIR [--dpi 200]
                                    [--scale FEET_PER_PIXEL] [--ocr auto|ocrmac|tesseract]
```

`INPUT` must be `.pdf`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`. `--ocr auto`
(default) tries ocrmac (macOS/Apple Vision) first, falling back to
tesseract if unavailable; `ocrmac` requires `--with ocrmac` since it is
macOS-only and cannot live in a portable `pyproject.toml` extra; tesseract
requires the `tesseract` binary on `PATH`.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Input missing or unsupported media type, or invalid arguments (e.g. `--dpi`/`--scale` not a finite number > 0) |
| 3 | Scale calibration failed and no `--scale` was given |
| 4 | No geometry survived extraction |
| 1 | Uncaught exception (a bug — should not happen) |

**Artifacts**, written to `--out DIR`:

- `plat.dxf` — layers `PL_BOUNDARY_LINE`, `PL_BOUNDARY_CURVE`,
  `PL_DIM_TEXT`, `PL_UNBOUND_TEXT` (declared, currently unpopulated — see
  L5), `PL_DEBUG` (declared, unused). `doc.units = ezdxf.units.FT`.
- `plat.json` — see the schema note below.
- (tesseract fallback only) `_tesseract_input.png`, an intermediate file
  `plat_extraction._ocr_tesseract` writes into `--out`.

**JSON schema — read this before consuming the file.** The task brief's
own example is a bare top-level JSON *array* of primitive objects. This
pipeline emits a top-level *object* instead:

```json
{
  "spatial_graph": [ /* the brief's array, one entry per primitive */ ],
  "unbound_text": [ /* boxes bind() could not attach anywhere */ ],
  "diagnostics": { /* pipeline counts, bind rate, calibration method */ }
}
```

This was the structure presented and approved at CHECKPOINT 2, not a
silent change made after the fact — a bare array has nowhere to carry
unbound text or diagnostics without inventing a non-primitive "type" value
to smuggle them into the same list. Within each `spatial_graph` entry, the
brief's `parsed_data` keys (`raw_text`, `bearing`, `distance`, `radius`,
`type`) are reproduced exactly, mapped from this module's `ParsedCall`
field names (`bearing_deg`→`bearing` etc.). `arc_length_ft` has no
counterpart in the brief's schema and is carried under a nested
`"diagnostics"` key on the call, never silently dropped. A curve's own
fitted center/radius (this pipeline's *measurement*, distinct from whatever
a radius label *states*) live under `"diagnostics"` on the primitive entry
for the same reason. **A consumer expecting the brief's bare array must
read `data["spatial_graph"]`, not the top-level value.**

**How an automated runner should consume the pair:** read `plat.json` for
anything semantic (which primitive has which calls, unbound text, bind
rate); open `plat.dxf` only for CAD rendering/audit. `layer_counts`
returned by `write_dxf` and the JSON's own `type` tally
(`diagnostics.n_lines`/`n_curves`) are cross-checkable, which is exactly
what `test_dxf_json_gate` asserts.

## 6. Known limitations and next steps

Carried forward from the Phase 1 defect checklist (dropped as a literal
`reference/legacy_binder.py` review — that file doesn't exist in this repo
— but tracked here as originally intended), plus everything found during
the build that recon did not predict.

### Fixed

| Item | Where |
|---|---|
| Double-edge tracing (`findContours` outline vs. centerline) | `trace_skeleton` uses `skimage.morphology.skeletonize`, confirmed by `test_skeleton_is_single_width` |
| Text pixels becoming fake geometry | `build_text_mask`, confirmed by `test_text_excluded` |
| Chord-deviation-only curve misclassification (the recon L-corner case) | `classify`'s two-stage design, confirmed by `test_l_corner_not_a_curve` |
| `tolerance=2.0` independent of DPI | Every spatial constant is a `PageContext` property |
| `np.cross` on 2-D (raises under NumPy 2.x, confirmed in recon) | Explicit scalar cross product everywhere, never `np.cross` |
| `approxPolyDP(..., closed=False)` on closed contours | Simplification runs on skeleton traces, which are genuinely open; `split_at_corners` handles the closed case explicitly |
| Size filter using `and` / `contourArea` on open polylines | Filters on arc length OR bbox span, both DPI-scaled |
| Dashed linetypes fragmenting | `merge_dashes`, confirmed by `test_dashes_merge` |
| Quadrant `[E W]` silently matching a bare space | `[EW]` character class, confirmed by `test_quadrant_never_captures_a_space` |
| Whole-degree bearings (`N 45° E`) dropped | Nested-optional bearing regex, confirmed by `test_whole_degree_bearing_matches` — **also fixed a worse bug recon didn't predict**: the first version of this regex couldn't parse `N 45°30' E` (minutes with a trailing tick and no seconds) at all, arguably the single most common real-world format |
| Out-of-range degrees/minutes/seconds accepted | Validated post-match, confirmed by `test_out_of_range_degrees_rejected` |
| Distance `\b` boundary backwards | Lookahead-based termination, confirmed by `test_distance_boundary` |
| Arc length conflated with chord distance | Separate `arc_length_ft` field + span-overlap exclusion, confirmed by `test_arc_length_not_conflated_with_distance` |
| Radius requiring `=` | Accepts `R=`, `R `, `RAD.`, `RADIUS =`, confirmed by `test_radius_forms` |
| No OCR-confusion normalization | `normalize_ocr` (context-anchored `0`→`°`, `''`→`"`), confirmed by `test_ocr_confusion_normalization` |
| Centroid-based binding | `densify` samples every primitive at a DPI-scaled spacing |
| Hardcoded `distance_upper_bound=75.0` | `PageContext.bind_radius_px`, DPI-derived |
| Greedy, one-directional binding | `linear_sum_assignment`, confirmed by `test_assignment_beats_greedy` |
| `IndexError`/sentinel risk on k-d tree miss | Replaced entirely — see Mitigated |
| Curves as dense polylines | `ARC` with a real fitted center/radius, confirmed by `test_curves_are_arcs` |
| `$INSUNITS` never set | `doc.units = ezdxf.units.FT`, confirmed by `test_units_set` |
| Unstable output ordering | Deterministic sort key + `sort_keys=True` JSON, confirmed by `test_dxf_json_gate`'s byte-identical check |
| Text rotation not normalized | Reuses `plat_extraction.label_rotation`'s existing `(-90, 90]` normalization |

### Mitigated (a real fix, with a residual caveat)

| Item | Fix | Caveat |
|---|---|---|
| `cKDTree.query`'s `idx == n` sentinel (recon Experiment 2 confirmed the behavior) | Replaced the whole approach: `query_ball_point` (uncapped, radius-only) has no sentinel to guard at all | The original fixed-`k` `query(..., distance_upper_bound=...)` design was abandoned mid-build for a *different*, more severe bug (below), which happened to also retire the sentinel question |
| L-corner vs. arc discrimination | Circle-fit residual, not chord deviation (recon Exp 6) | The recon-proposed secondary guard (turn-consistency) does **not** discriminate the L-corner case either — confirmed, both report a single trivially-consistent turn. It is kept only as an S-curve guard, and even then required redesigning from per-pixel stride comparison (swamped by 8-connected skeleton staircase noise at every stride tried, 1–8px) to coarse whole-trace chunk comparison before it worked at all |
| Angular gate (text orientation vs. segment bearing) | **Not implemented, by necessity, not oversight** | This module's OCR engines report only axis-aligned boxes (confirmed: `region_shape: "axis-aligned rect"` is the only shape either engine emits) — there is no orientation to compare. A box-aspect-ratio proxy was considered and rejected: a rotated label's axis-aligned box tends toward square regardless of true angle, so the proxy is least reliable exactly on the rotated labels this problem is hardest for. The fixture's specific ambiguous case (a right-of-way line positioned so its label is geometrically *equidistant* from it and the true boundary — confirmed by direct calculation) is resolved instead via `Primitive.is_boundary` (closed parcel loop vs. isolated run), a real structural signal this pipeline does have |

### Deferred, ranked by risk

**High risk — curve-table / legend content contaminates binding and
calibration.** Confirmed on the real `cmd_sheets` fixture (not the isolated
gate fixtures): the sheet's curve table is drawn as a grid of bordered
cells, and cell borders extract as ordinary closed-loop `line` primitives
(`is_boundary=True`, since a rectangle is a closed loop too — the same
signal that correctly discriminates the true boundary from an isolated
right-of-way line does *not* discriminate it from an unrelated closed
rectangle elsewhere on the sheet). Individual cell values (`"190.00'"`,
`"240.00'"`) parse as valid `line_dimension` calls and bind to nearby cell
borders. On the real run this pulled the RANSAC-calibrated
`feet_per_pixel` to 0.689 and 0.512 across two media of the *same* sheet,
against a true value of ≈0.424 — RANSAC's outlier rejection cannot save a
sample where the *majority* of usable (primitive, distance) pairs are this
kind of false-positive, not genuine outliers among genuine boundary calls.
Needs a distinct-region concept (e.g. exclude OCR boxes/geometry inside a
detected tabular/bordered layout region from boundary binding) that this
prototype's scope did not include. The P2/P3/P4 gate tests remain valid:
each isolates the *mechanism* under test (classification, tie-breaking,
assignment, emission) from this integration-level contamination by
construction, using hand-built minimal fixtures rather than the full noisy
sheet — but that means **the gates do not, and cannot, catch this
particular failure mode**; it was found only by running the full CLI
end-to-end.

**Medium risk — real OCR text quality on small/rotated labels.**
Confirmed on the same real run: ocrmac merges adjacent regions (a
monument's `"IPF"` label and a nearby boundary call arrived as one box),
drops characters (`"S 3°21'59\" W 170.29'"` came back as `"S 32159\"W
170.29*"`, dropping the degree sign and a digit), and loses ticks entirely
(`"N 35°09'59\" E 107.65'"` came back as `"\N 35°0959\" E 107.65"`). This
is not new — it is issue #108's own Axis 1 finding, which this handoff
re-confirms rather than discovers. `normalize_ocr`'s confusion mapping
handles the *specific*, unambiguous confusions the recon brief named
(`0`↔`°`, `''`↔`"`); it deliberately does not attempt `5`↔`S` or `1`↔`I`
beyond what's structurally unambiguous, since guessing wrong on a
direction/digit letter produces a silently wrong bearing rather than a
correctly-rejected non-match, which is a worse failure than the miss it
would "fix." No further action taken here; flagged so a future session
does not assume OCR text is clean input.

**Medium risk — `inferred_angle` can be 180° from the label's own stated
bearing.** Confirmed on the real run's JSON output. A primitive's chord
azimuth depends on which of its two endpoints `trace_skeleton`'s walk
happened to visit first, which has no enforced relationship to the
direction convention a bound label's bearing was written in (deed
traversal order, clockwise vs. counterclockwise). Reconciling trace
direction with a consistent parcel-traversal convention is a real,
scoped-out piece of work, not a bug in either value individually.

**Low risk — no real-plat corpus, no georeferencing, no multi-sheet
handling.** Stated in the original task scope and never in question; noted
here only so "known limitations" is complete. All fixtures are synthetic
and local, as `plat_extraction.py` already states about its own.

**Low risk — `PL_UNBOUND_TEXT` and `PL_DEBUG` layers are declared but
unpopulated.** `write_dxf` creates both layers (so `doc.layers` matches the
brief's named set) but only ever adds entities to `PL_BOUNDARY_LINE`,
`PL_BOUNDARY_CURVE`, and `PL_DIM_TEXT`. Unbound text is fully captured in
the JSON's `unbound_text` array; it was not additionally drawn into the DXF
as text entities, since the brief's acceptance gate checks layer counts
against the JSON, not that every declared layer is non-empty. A follow-up
session wanting unbound text visible *in the DXF itself* (for a human
reviewer working from the drawing rather than the JSON) can add that in a
few lines using the same `write_dxf` machinery.

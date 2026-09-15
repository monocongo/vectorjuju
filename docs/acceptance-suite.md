# Acceptance suite for `convert()`

What proves [`vectorjuju.convert()`](https://github.com/monocongo/vectorjuju/issues/5)
does what its contract promises, end to end. Settled by [Define the acceptance
test suite that proves convert() works as
advertised](https://github.com/monocongo/vectorjuju/issues/14); built alongside
`convert()`.

Two verdict classes:

- **Mechanical gates** (A1–A11, U1–U5 below) — pytest assertions over a
  generated synthetic sheet. A red gate means the MVP is not done.
- **Eyeballed evidence** — one `overlay.png` per end-to-end run plus a private
  plat checklist, reviewed by a human. Never a pass/fail, never committed.

## Coordinate spaces and fixture math

Ground truth is in **page points**; rasters are rendered at 200 dpi
(`synthetic_plat.RENDER_DPI`); CAD is in the run's `units`. All gate distances
are **raster pixels at the run DPI** unless noted, and gates run at the default
`dpi=200`: `convert()` rasterises the PDF there and the generator pre-renders
TIFF/JPG there, so the `px = pt * dpi / 72` math holds for all three media.

```
px = pt * dpi / 72
cad_x = px_x * fpp                       # what convert() promises
cad_y = (img_height - px_y) * fpp
planted fpp = 72 / (0.85 * dpi)          # 0.423529 ft/px at the default dpi=200
planted fpp_m = planted fpp * 1200 / 3937  # 0.129092 m/px, for a metre run
```

`0.85` is `synthetic_plat.SCALE_PT_PER_FT`; `fpp` is the sidecar's
`scale.value`, in the run's unit. US survey and international feet differ by
2 ppm — far below the 1 % scale budget — so both foot runs share the `fpp`
oracle and are told apart by `$INSUNITS` alone; a `metre` run uses `fpp_m` and
every CAD-space gate compares metres. The sheet, its three media, and
`ground_truth.json` come from `synthetic_plat.generate_sheet()` at test-session
scope — generated, never committed ([Port the synthetic plat sheet
generator](https://github.com/monocongo/vectorjuju/issues/6)).

Planted counts: 6 parcel edges (4 straight calls, 2 curve refs, no two
straight edges collinear), 2 parallel right-of-way offset lines, a 3-row curve
table (header plus `C1`/`C2`), monuments, north arrow, scale bar, title block.

## Gate tolerances

Thresholds are derived, not tuned:

- **3 px** — just under one stroke width at 200 dpi (1.4 pt = 3.9 px), so it
  separates the planted line from a distractor 10 pt (28 px) away while
  absorbing rasterisation.
- **1 %** on `fpp` covers ~0.2 % length quantisation and 2-decimal call
  rounding on a 110–200 ft edge.
- **8 px** end reach ≈ corner-splitting slop where a straight run meets an arc.
- **3 %** on arc radius, **4 px** on arc endpoints, is looser than a clean
  circle fit needs on purpose. On this fixture's clean 48-vertex arcs a SPLINE
  fallback is a failure (A4); the contract's SPLINE fallback is for real traces
  that don't fit a circle, and the fixture plants none.
- **5°** on label rotation: trace-direction jitter (~1° for 2 px over a 100 px
  run) has to pass, and the magnitude-only buckets cannot reject a mirrored
  label; 5° does, except on the flat ~2° edge, where the mirror is inside the
  jitter band and harmless anyway — text reads left-to-right at either sign.
- **20 px** label insertion is the distance from the emitted insertion point to
  the planted text box `quad_pt`, inside the box counting as zero — that
  absorbs OCR box drift and any corner-versus-centre convention, and the
  planted 5 pt draw offset is already inside the box. The contract says
  insertion is the OCR position, not a synthesized offset.

Label text fidelity reuses the classification from the throwaway
`prototypes/diagonal_call_labels.py` (`text_only` etc.) — criteria, not its
code: a read passes if its verdict is `exact` or `normalized`, or `text_only`
with no unit mark gained or swapped — for each mark, `count(recovered) <=
count(truth)`. Lost unit-mark punctuation
(`'` `"` `°`) is tolerated; a wrong digit, letter, dropped `.`, or a `200.16'`
read as `200.16"` — a 12× unit error the bare `text_only` verdict would wave
through — is not.

## Unit gates — no OCR, no full pipeline

`tests/test_convert_units.py`. Anything assertable against planted geometry or
a pure function lives here.

| # | Gate | Assertion | Tolerance |
|---|---|---|---|
| U1 | Transform | `px_to_cad(cad_to_px(p))` round trip; origin flip (top of raster → large CAD y) | ≤ 1e-6, 2 DPI × 2 fpp |
| U2 | Calibration | Planted call distances vs planted pixel lengths recover `fpp` | relative error ≤ 1 % |
| U3 | Classification | Planted 48-vertex arc → `curve` + circle fit; L-corner → `line`; 3 px stroke traces one centreline; glyph raster yields zero primitives; dashed run merges to one; same figure at 2 DPIs gives the same count | prior-art gates, carried |
| U4 | DXF contract | `$ACADVER` AC1015; `$INSUNITS` 21 / 2 / 6 per `units`; the three layers exist and every entity sits on one of them, with the writer's mandatory `0` and `Defpoints` layers empty; fitted curve emits `ARC`; MTEXT rotation ∈ (-90, 90]; no XDATA and no APPID referenced by any entity (the DXF-mandated `ACAD` APPID stays in the table); reopen audit error list empty; per-layer counts match sidecar; sidecar keys/types match the sidecar schema settled in [issue #5](https://github.com/monocongo/vectorjuju/issues/5) (`units`, `scale`, `dpi`, `entities[]`, `unbound_text`) — the prior-art `spatial_graph` shape is superseded; two writes byte-identical | exact |
| U5 | Parsing and binding | Parser table incl. unit-mark variants and non-calls (`PARCEL 5`, `LOT 12`); a planted call beside a 10 pt-parallel offset line binds to the planted line; off-gate text lands in `unbound_text`, never force-bound | exact |

## Acceptance gates — full `convert()` per media

`tests/test_convert_acceptance.py`, parametrised over `sheet.pdf`,
`sheet.tif`, and the EXIF-rotated `sheet.jpg` from one generated fixture. Each
medium converts into its own output directory: the contract's default `out` is
`<stem>.dxf`, all three media share the stem `sheet`, and one shared directory
would let a run overwrite the previous medium's output — turning A9 into a
comparison of a file with itself.

| # | Gate | Assertion | Tolerance |
|---|---|---|---|
| A1 | Scale | `sidecar.scale.value` vs planted `fpp`; `method == "ransac"`; with `scale=` override `method == "override"` | ≤ 1 % / ≤ 1e-9 |
| A2 | Calls bound | Each of the 4 planted straight calls appears exactly once as the `label.raw_text` of a `type: line` sidecar `entities[]` entry; its interior vertices lie within 3 px of the planted segment and its two extreme vertices reach within 8 px of that segment's endpoints — the corner-splitting slop A7 uses, which replaces the 3 px bound at the ends; text passes the fidelity rule above | as derived |
| A3 | Distractors | No label-carrying entity lies within 3 px of either planted offset line; no string from `ground_truth["distractor_text"]`, no title-block text (a class the fixture must add to `distractor_text`, including `SCALE: 1" = 100'`), and no `curve_table_cells` value appears in any `entities[].label.raw_text` except the `C1`/`C2` cells, which are textually identical to the planted on-curve curve labels — A4's insertion check is what proves the on-curve label was read rather than the table cell; `unbound_text` is fine | 3 px, exact |
| A4 | True arcs | Exactly 2 `ARC` on `BOUNDARY_CURVE`, 0 `SPLINE` (the fixture's arcs are clean circles), no dense polyline standing in; each planted curve ref `C1`/`C2` appears exactly once as the `label.raw_text` on its planted curve entity, with that label's insertion within 20 px of the curve-ref label's planted `quad_pt` (inside = 0) — the table cell carrying the same text sits far away, so this is what distinguishes an on-curve read from the table cell; radius vs planted `radius_ft`; endpoints vs planted chord endpoints; ARC midpoint vs the planted curve label anchor (proves bulge side) | 20 px / ≤ 3 % / ≤ 4 px / ≤ 8 px |
| A5 | Units | Default run: `$INSUNITS == 21` (`doc.units` reads that same header var back), `sidecar.units == "us-survey-foot"`; `international-foot` variant → 2; `metre` variant → 6, `sidecar.scale.value` vs `fpp_m` and every CAD coordinate in metres | exact / ≤ 1 % on the scale |
| A6 | Labels | Every rotation ∈ (-90, 90]; each straight call's rotation within 5° of its planted `rotation_deg` (already normalised, so it is the same whichever way the run was traced) — the bucket follows, and a mirror of a non-flat label fails where a magnitude-only bucket test would pass it; insertion within 20 px of the planted text box `quad_pt` (inside = 0) | exact / 5° / 20 px |
| A7 | Run continuity | Per planted straight edge, exactly one `LWPOLYLINE` runs within 3 px of it, its two extreme vertices reach within 8 px of the edge's endpoints, and no second parallel entity runs within 3 px (double-edge regression) | 3 px / 8 px |
| A8 | Determinism | Two runs on one input produce identical DXF bytes and identical JSON bytes | exact |
| A9 | Media agreement | A1–A8 pass for each medium; matching entities agree across media within 3 px; bound label text sets identical | 3 px |
| A10 | Failure paths | Multi-page PDF, multi-page TIFF, zero-page PDF, unsupported extension, corrupt file → `UnsupportedInputError`; text-only page (no distances) → `ScaleCalibrationError`; after either failure, on a fresh output path, neither `out` nor its `<stem>.json` sidecar exists | exact |
| A11 | Sidecar ↔ DXF | Every sidecar entity maps 1:1 onto a DXF entity of its layer; no entity on a layer other than the three, and no entity on the writer's mandatory `0`/`Defpoints` layers | exact |

DXF is read back through `ezdxf.recover.readfile` (which is also U4's audit
check), so gates read what a CAD consumer reads.

## Visual evidence

One artifact per end-to-end run: `overlay.png` — the input raster with the
DXF geometry on top (`BOUNDARY_LINE` blue, `BOUNDARY_CURVE` red, label
insertions as small crosses). `tests/acceptance_helpers.py::write_overlay(page,
dxf, sidecar, out)` inverts the sidecar transform with Pillow `ImageDraw`
only: no matplotlib, no ezdxf drawing add-on, no image-comparison dependency.
It is a test helper, not shipped code, so nothing in the package needs Pillow;
the private-plat run imports it from the repo checkout.

Where it lands: CI uploads `overlay-<media>.png` from the acceptance job
(`if: always()`), and the PR that claims the MVP links it for review. Nothing
binary is committed.

**No golden images.** A legitimate tracing or fit change moves pixels, so a
golden would gate on rendering and anti-aliasing rather than correctness, and
the mechanical gates already assert the properties a human checks here. Add
goldens with tolerance only if a regression class is shown to escape every
mechanical gate.

## Layering and files

```
tests/conftest.py                     session-scoped synthetic sheet; per-media converted fixtures, one output dir each
tests/test_convert_units.py           U1–U5, no OCR
tests/test_convert_acceptance.py      A1–A11, OCR, pytest.mark.acceptance
tests/acceptance_helpers.py           pt<->px<->cad, label classify, segment/arc comparison, write_overlay()
```

Split rule: assertable against planted geometry or a pure function → unit.
Only properties of the full input→DXF path are acceptance gates.

## CI and local split

CI, every PR:

- Unit gates in the existing job, which becomes `pytest -m "not acceptance"`
  (add the `acceptance` marker to `pyproject.toml`'s
  `[tool.pytest.ini_options]`) so the OCR path runs once, in the acceptance job.
- A dedicated `acceptance` job (ubuntu-latest) running `pytest -m acceptance`
  over all three media plus the failure paths. Sheet generated once per
  session (~2 s), one `convert()` per medium, docling model cache keyed to the
  pinned docling version. Target ≤ 5 min wall; if exceeded, split the job,
  never move a medium to local-only.
- `overlay-*.png` uploaded always.
- **A skip is not a pass.** If the OCR engine cannot initialise in CI, the
  acceptance job fails. Linux/CPU OCR parity is the map's open question — the
  suite is what surfaces it, so it must not be papered over by `skipif`.

Local only: the private plat checklist below, and running the acceptance file
against an alternate OCR engine. Neither runs in CI.

## Private plat hand-check

Local, manual, uncommitted, following the issue-13 prototype's private-run
convention: aggregate counts only, no recognized text, coordinates, crops, or
the file itself ever reported.

1. Work in a directory outside the repo that removes itself on success, error,
   and interrupt (`tempfile.TemporaryDirectory`, or an `EXIT`/`INT`/`TERM` trap
   around `mktemp -d`); all outputs (DXF, sidecar, overlay, any crop) exist
   only there.
2. Run `convert()` with the plat's real `units` and no `scale` override —
   calibration must earn its result.
3. Run twice; DXF and JSON must be byte-identical.
4. Generate `overlay.png`, open the DXF in a viewer, and walk: the three
   layers and their entity types, boundary lines tracking the raster, arcs
   that read as circular, labels legible and on the right line, monument and
   table text not bound as calls, sidecar in CAD units with
   `scale.method == "ransac"`.
5. Hand-verify 3–5 printed distances against measured DXF lengths (≤ 2 %) and
   the curve-table radii against fitted ARC radii (≤ 5 %); count labels bound
   vs unbound.
6. Report only aggregates: pixel size, layer/entity counts, bound/unbound
   label counts, failure categories. No content-derived identifier of the
   source — a hash prefix lets anyone holding a candidate copy confirm it was
   the file used — with the temp dir already gone before anything is reported.
7. Failures (uncalibratable, unreadable) are reported by category, not
   skipped silently.

Steps 4–5 are eyeballed by definition; the checklist makes the eyeball
repeatable rather than a vibe.

## Dependencies

`convert()` brings `ezdxf`, `numpy`, `docling`, OpenCV, and `scikit-image` as runtime
dependencies; the suite adds none beyond those — `ezdxf` is what the gates
reopen the DXF with, and `pillow`/`pypdfium2` are already dev dependencies, the
overlay helper included. No matplotlib, no golden-image tooling.

## Definition of done

A1–A11 green on PDF, TIFF, and EXIF-rotated JPG in CI, U1–U5 green, the
private plat checklist completed with its aggregates reported, and the
overlay reviewed on the PR that claims the MVP.

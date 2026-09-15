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

Ground truth is in **page points**; rasters are rendered at the run's `dpi`;
CAD is feet. All gate distances are **raster pixels at that DPI** unless noted.

```
px = pt * dpi / 72
cad_x = px_x * fpp                       # what convert() promises
cad_y = (img_height - px_y) * fpp
planted fpp = 72 / (0.85 * dpi)          # 0.423529 at the default dpi=200
```

`0.85` is `synthetic_plat.SCALE_PT_PER_FT`; `fpp` is the sidecar's
`scale.value`. The sheet, its three media, and `ground_truth.json` come from
`synthetic_plat.generate_sheet()` at test-session scope — generated, never
committed ([Port the synthetic plat sheet
generator](https://github.com/monocongo/vectorjuju/issues/6)).

Planted counts: 6 parcel edges (4 straight calls, 2 curve refs), 2 parallel
right-of-way offset lines, 6 curve-table rows including header, monuments,
north arrow, scale bar, title block.

## Gate tolerances

Thresholds are derived, not tuned:

- **3 px** — just under one stroke width at 200 dpi (1.4 pt = 3.9 px), so it
  separates the planted line from a distractor 10 pt (28 px) away while
  absorbing rasterisation.
- **1 %** on `fpp` covers ~0.2 % length quantisation and 2-decimal call
  rounding on a 110–200 ft edge.
- **8 px** end reach ≈ corner-splitting slop where a straight run meets an arc.
- **3 %** on arc radius, **4 px** on arc endpoints, is looser than a clean
  circle fit needs on purpose — a SPLINE fallback is a failure (A4), a
  slightly-off radius is not.
- **20 px** label insertion ≈ the planted anchor ± the 5 pt (14 px) text
  offset and OCR box drift. The contract says insertion is the OCR position,
  not a synthesized offset.

Label text fidelity reuses the classification from the throwaway
`prototypes/diagonal_call_labels.py` (`text_only` etc.) — criteria, not its
code: a read passes if its verdict is `exact`, `normalized`, or `text_only`.
Lost unit-mark punctuation (`'` `"` `°`) is tolerated; a wrong digit, letter,
or dropped `.` is not.

## Unit gates — no OCR, no full pipeline

`tests/test_convert_units.py`. Anything assertable against planted geometry or
a pure function lives here.

| # | Gate | Assertion | Tolerance |
|---|---|---|---|
| U1 | Transform | `px_to_cad(cad_to_px(p))` round trip; origin flip (top of raster → large CAD y) | ≤ 1e-6, 2 DPI × 2 fpp |
| U2 | Calibration | Planted call distances vs planted pixel lengths recover `fpp` | relative error ≤ 1 % |
| U3 | Classification | Planted 48-vertex arc → `curve` + circle fit; L-corner → `line`; 3 px stroke traces one centreline; glyph raster yields zero primitives; dashed run merges to one; same figure at 2 DPIs gives the same count | prior-art gates, carried |
| U4 | DXF contract | `$ACADVER` AC1015; `$INSUNITS` 21 / 2 / 6 per `units`; layers exactly `BOUNDARY_LINE`/`BOUNDARY_CURVE`/`LABEL`; fitted curve emits `ARC`; MTEXT rotation ∈ (-90, 90]; no XDATA/APPID; reopen audit error list empty; per-layer counts match sidecar; sidecar keys/types match the contract schema; two writes byte-identical | exact |
| U5 | Parsing and binding | Parser table incl. unit-mark variants and non-calls (`PARCEL 5`, `LOT 12`); a planted call beside a 10 pt-parallel offset line binds to the planted line; off-gate text lands in `unbound_text`, never force-bound | exact |

## Acceptance gates — full `convert()` per media

`tests/test_convert_acceptance.py`, parametrised over `sheet.pdf`,
`sheet.tif`, and the EXIF-rotated `sheet.jpg` from one generated fixture.

| # | Gate | Assertion | Tolerance |
|---|---|---|---|
| A1 | Scale | `sidecar.scale.value` vs planted `fpp`; `method == "ransac"`; with `scale=` override `method == "override"` | ≤ 1 % / ≤ 1e-9 |
| A2 | Calls bound | Each of the 4 planted straight calls appears exactly once in `entities[].label`, on a `line` entity whose vertices lie within 3 px of its planted segment and whose endpoints reach within 8 px of that segment's endpoints; text verdict ∈ {`exact`, `normalized`, `text_only`} | as derived |
| A3 | Distractors | No label-carrying entity lies within 3 px of either planted offset line; no planted distractor string (monuments, north arrow, scale bar, `CURVE TABLE`, table cells other than `C1`/`C2` on their own arcs) appears in any `entities[].label` — `unbound_text` is fine | 3 px, exact |
| A4 | True arcs | Exactly 2 `ARC` on `BOUNDARY_CURVE`, 0 `SPLINE`, no dense polyline standing in; radius vs planted `radius_ft`; endpoints vs planted chord endpoints; ARC midpoint vs the planted curve label anchor (proves bulge side) | ≤ 3 % / ≤ 4 px / ≤ 8 px |
| A5 | Units | Default run: `$INSUNITS == 21`, `doc.units == FT`, `sidecar.units == "us-survey-foot"`; metre variant → 6 | exact |
| A6 | Labels | Every rotation ∈ (-90, 90]; each straight call's rotation bucket equals its planted `bucket`; insertion within 20 px of the planted anchor | exact / 20 px |
| A7 | Run continuity | Per planted straight edge, exactly one `LWPOLYLINE` runs within 3 px of it, its two extreme vertices reach within 8 px of the edge's endpoints, and no second parallel entity runs within 3 px (double-edge regression) | 3 px / 8 px |
| A8 | Determinism | Two runs on one input produce identical DXF bytes and identical JSON bytes | exact |
| A9 | Media agreement | A1–A8 pass for each medium; matching entities agree across media within 3 px; bound label text sets identical | 3 px |
| A10 | Failure paths | Multi-page PDF, multi-page TIFF, unsupported extension, corrupt file → `UnsupportedInputError`; text-only page (no distances) → `ScaleCalibrationError`; no DXF at `out` after either | exact |
| A11 | Sidecar ↔ DXF | Every sidecar entity maps 1:1 onto a DXF entity of its layer; no unaccounted layer | exact |

DXF is read back through `ezdxf.recover.readfile` (which is also U4's audit
check), so gates read what a CAD consumer reads.

## Visual evidence

One artifact per end-to-end run: `overlay.png` — the input raster with the
DXF geometry on top (`BOUNDARY_LINE` blue, `BOUNDARY_CURVE` red, label
insertions as small crosses). `src/vectorjuju/overlay.py::write_overlay(page,
dxf, sidecar, out)` inverts the sidecar transform with Pillow `ImageDraw`
only: no matplotlib, no ezdxf drawing add-on, no image-comparison dependency.

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
src/vectorjuju/overlay.py             write_overlay(); also used by the local plat run
tests/conftest.py                     session-scoped synthetic sheet + per-media converted fixtures
tests/test_convert_units.py           U1–U5, no OCR
tests/test_convert_acceptance.py      A1–A11, OCR, pytest.mark.acceptance
tests/acceptance_helpers.py           pt<->px<->cad, label classify, segment/arc comparison
```

Split rule: assertable against planted geometry or a pure function → unit.
Only properties of the full input→DXF path are acceptance gates.

## CI and local split

CI, every PR:

- Unit gates in the existing job.
- A dedicated `acceptance` job (ubuntu-latest) running all three media plus
  the failure paths. Sheet generated once per session (~2 s), one `convert()`
  per medium, docling model cache keyed to the pinned docling version. Target
  ≤ 5 min wall; if exceeded, split the job, never move a medium to local-only.
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

1. Work in `mktemp -d` outside the repo; all outputs (DXF, sidecar, overlay,
   any crop) exist only there.
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
6. Report only aggregates: sha256 prefix, pixel size, layer/entity counts,
   bound/unbound label counts, failure categories. Delete the temp dir when
   done.
7. Failures (uncalibratable, unreadable) are reported by category, not
   skipped silently.

Steps 4–5 are eyeballed by definition; the checklist makes the eyeball
repeatable rather than a vibe.

## Dependencies

`convert()` brings `ezdxf`, `numpy`, `docling`, and OpenCV as runtime
dependencies; the suite adds none beyond those — `ezdxf` is what the gates
reopen the DXF with, and `pillow`/`pypdfium2` are already dev dependencies.
No matplotlib, no golden-image tooling.

## Definition of done

A1–A11 green on PDF, TIFF, and EXIF-rotated JPG in CI, U1–U5 green, the
private plat checklist completed with its aggregates reported, and the
overlay reviewed on the PR that claims the MVP.

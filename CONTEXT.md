# Context

Glossary of domain terms used in vectorjuju code, docs, and issues.

- **Plat Sheet** — one page or image of a survey plat.
- **Boundary Call** — a bearing and distance, or a circular-curve element, recorded for a boundary course.
- **Curve Table** — a table on the sheet listing curve elements keyed by labels such as `C1`.
- **Planted Ground Truth** — the synthetic sheet's `ground_truth.json`: its input geometry and text in page points. Distinct from the `convert()` JSON sidecar, which describes calibrated CAD-unit output.
- **Acceptance Gate** — a mechanical pass/fail assertion in [docs/acceptance-suite.md](docs/acceptance-suite.md): over a full `convert()` run on a synthetic sheet (A1–A11), or over its planted geometry and pure functions (U1–U5).
- **Overlay** — the human-review artifact: input raster with emitted DXF geometry drawn on top.

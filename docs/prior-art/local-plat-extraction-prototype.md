> Carried from docproc [`docs/prototypes/local-plat-extraction.md`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/docs/prototypes/local-plat-extraction.md) at `3342063b5f001038020cf0ae6b14ac0c8de0c966`.

# Local extraction of grounded boundary calls — prototype findings

## Question and boundary

Using only local tooling, the private exploratory plat rendered as raster, and
project-authored synthetic raster sheets: how much verbatim boundary-call text,
source geometry, curve-table structure, and call-to-line association can be
recovered from flattened content without unsupported inference, and what failure
modes and candidate interface must the production design accommodate across PDF,
TIFF, and JPEG?

This is a throwaway prototype, not production code. It acquired no external
corpus, downloaded no models, made no network call, and makes no accuracy claim
about real-world plats. The private exploratory sample was read locally only;
nothing derived from it — no verbatim text, crop, overlay, or coordinate — is
recorded here or committed. Numbers attributed to it below are aggregate counts.

Prototype: [`prototypes/plat_extraction.py`](https://github.com/monocongo/docproc/blob/3342063b5f001038020cf0ae6b14ac0c8de0c966/prototypes/plat_extraction.py).

## Conclusion

The stack recommended by
[Research local extraction options for scanned survey plats](https://github.com/monocongo/docproc/issues/106)
runs end to end on all three media and does produce OCR regions and separate
geometry candidates. It is nevertheless **insufficient as the sole extraction
path**:

1. **Not one planted boundary call was recovered verbatim** — on any medium, by
   either engine, at any rotation. Every recovered call differed from the
   planted string, and the differences are survey-critical punctuation rather
   than cosmetic noise.
2. **Neither engine supplies table row/column structure at all**, so the curve
   and line tables come back as loose text.
3. **Call-to-line association fails in the presence of ordinary parallel
   non-boundary lines** — and this failure is not a reading problem, so a
   stronger reader does not fix it.

Points 1 and 2 are recognition-quality limits that better tooling could plausibly
raise. Point 3 is structural: association must remain a reviewable candidate,
never an automatic determination, whatever the extractor.

## Method

`plat_extraction.py sheets` draws a synthetic plat with reportlab and emits
planted ground truth alongside three media from one source: the PDF, a TIFF
rasterization, and a JPEG stored physically rotated and tagged EXIF
Orientation 6. The parcel has six edges — four straight calls at deliberately
spread rotations (flat 2.3°, shallow −15.9°, steep 54.8°, near-vertical 86.6°)
and two circular arcs carrying `C1`/`C2` references into a curve table. It also
carries monument tags, a north arrow, scale bar, title block, and — added after a
first pass scored a meaningless 100% on association — two **parallel
non-boundary lines** at 10pt offset, the ordinary right-of-way and setback
geometry every real plat contains.

`extract` normalizes (pypdfium2 render for PDF; Pillow `exif_transpose` for
raster, recording the applied orientation), runs both OCR engines, runs OpenCV
for a separate geometry-candidate channel, and writes one evidence JSON plus
overlays. `score` compares evidence to planted truth on four axes.

Engines: **ocrmac 1.0 / Apple Vision** and **Tesseract 5.5.3** at `--psm 11`.

## Axis 1 — verbatim boundary-call text

Recovery of the four planted straight calls plus two curve references, lossless
PDF path (TIFF is identical; JPEG is worse):

| engine | bucket | n | exact | normalized | garbled | missed | fragments/label |
|---|---|---:|---:|---:|---:|---:|---:|
| ocrmac | flat | 1 | 0 | 0 | 1 | 0 | 1.0 |
| ocrmac | shallow | 2 | 0 | 0 | 1 | 1 | 0.5 |
| ocrmac | steep | 1 | 0 | 0 | 1 | 0 | 1.0 |
| ocrmac | near-vertical | 2 | 0 | 0 | 1 | 1 | 0.5 |
| tesseract | flat | 1 | 0 | 0 | 0 | 1 | 1.0 |
| tesseract | shallow | 2 | 0 | 0 | 1 | 1 | 3.0 |
| tesseract | steep | 1 | 0 | 0 | 0 | 1 | 3.0 |
| tesseract | near-vertical | 2 | 0 | 0 | 0 | 2 | 0.0 |

**Zero exact matches in every cell of that table.** The error taxonomy, all
observed on the synthetic fixture:

| planted | recovered | consequence |
|---|---|---|
| `200.16'` | `200.16"` | foot mark read as inch mark — **silent unit corruption** |
| `N 35°09'59" E` | `N 35°0959" E` | minute and second marks dropped |
| `S 3°21'59" W  170.29'` | `S 32159"W 170.29*` (JPEG) | bearing no longer parseable at all |
| `N 74°06'46" W …` | `IPF N 74°06'46" W …` | adjacent monument tag absorbed into the label |
| `N 74°06'46" W` | `N74º0646" W` | `°` U+00B0 → `º` U+00BA masculine ordinal |
| `107.65'` | `107,65` | decimal point → comma |

Rotation is the dominant variable for Tesseract, which missed the near-vertical
call on **all three media** and shattered the flat call into three fragments.
Apple Vision degrades gracefully with rotation — it returned whole-label regions
at every angle — but drops standalone short rotated tokens: `C1` and `C2` were
missed by both engines on every medium, so a curve reference on the drawing
cannot currently be tied back to its table row.

Lossy JPEG measurably worsens recognition: the near-vertical call scored
`normalized` on the lossless path in an earlier run and `garbled` once
compression was applied.

## Axis 2 — source geometry carried with the text

| engine | labels with a region | mean IoU vs planted box | region shape |
|---|---|---:|---|
| ocrmac | 4/6 | 0.69 | axis-aligned rect |
| tesseract | 1/6 | 0.88 | axis-aligned rect |

Both engines return **axis-aligned rectangles only**. For rotated text this
over-covers by construction: the bounding rect of a 55°-rotated label is far
larger than the glyphs, which is precisely how the `IPF` monument tag ended up
inside a boundary call's region. Tesseract's higher IoU is not better
performance — it is the arithmetic of scoring one surviving near-flat label.

Representing a rotated label's true extent requires an **oriented quadrilateral**.
Neither engine exposes one through the interface used here, and the repository's
current `BBox` could not store it if they did.

## Axis 3 — curve and line table structure

| engine | planted cells recovered as text | row/column grouping |
|---|---:|---|
| ocrmac | 15/18 (14/18 on JPEG) | **none** |
| tesseract | 10/18 | **none** |

Cell *text* is largely recoverable; cell *structure* is not recovered at all.
Both engines return a flat list of regions with no notion of row, column, or
header. Reconstructing a curve table therefore means inferring the grid from
region positions — inference this prototype deliberately did not attempt.

The exploratory sample confirms a further constraint: its curve table uses a
different column set than the fixture's. Per the reviewer, **table schema must be
treated as per-plat**, so any fixed canonical column set is wrong by
construction; the schema has to be discovered per sheet and confirmed.

## Axis 4 — call-to-line association

One naive nearest-segment rule, anchored on the region the OCR engine actually
returned (not on the planted line, which no production system knows).

| condition | ocrmac | note |
|---|---|---|
| boundary lines only, planted geometry | **100%** | meaningless — the first fixture had no competing lines |
| including non-boundary lines, planted geometry | **50%** | 2 of 4 calls attached to a right-of-way/setback line |
| including non-boundary lines, detected geometry | **50%** | |
| ambiguity margin, nearest vs second | min 14px, median 79px | |

Adding two ordinary parallel lines halved the association rate. Every label that
had a parallel line beside it was mis-assigned. This is the finding that matters:
**a boundary call attached to a right-of-way line is not a near miss, it is a
wrong boundary, produced silently.**

The exploratory sample makes the scale of the problem concrete. On one real
sheet the geometry channel returned **1,402 Hough segments and 10,332 contours**
for a parcel whose boundary is roughly ten lines — a candidate set two orders of
magnitude larger than the answer. Nearest-segment selection across that set is
not a viable rule at any reading quality.

## Aggregate observations from the exploratory sample

No verbatim content, coordinate, or crop from the private sample appears here.

- Rendered at 4800×3590 px. Apple Vision returned **171 regions**, of which
  **14** matched a bearing-like pattern and only **6** contained a complete
  bearing-and-distance pair. Tesseract returned **1,101 regions**, of which
  **1** was bearing-like and **537** were three characters or fewer — scanner
  speckle, not text.
- The sheet is multi-purpose: certification and review-officer blocks, thirteen
  numbered general notes, a vicinity map, legend, line table, curve table,
  adjoiner blocks, setback lines, easements, right-of-way, and sanitary sewer
  linework. Boundary calls are a small minority of the text on the page.
- **More than one parcel appears on the sheet**, confirming that operator
  selection of subject parcels is a requirement rather than a convenience.
- The same physical line carries **paired grid and ground values**. Per the
  reviewer, a boundary has a single authoritative value — **ground governs; grid
  is presentational** — so both observations must be retained but grid must never
  be silently substituted for ground.
- Setback and right-of-way lines run parallel to boundaries **carrying their own
  dimension labels**, which is the axis-4 failure mode occurring naturally and
  more densely than the fixture models it.

## Reviewer-confirmed decisions from this session

1. **Grid versus ground.** A boundary has one authoritative value. Ground
   governs; grid values are presentational. Retain both as observations; never
   silently substitute.
2. **Punctuation loss is reviewer-correctable but must be surfaced.** A dropped
   foot mark is trivial for a survey-literate reviewer to fix — but the system
   must *highlight* suspect tokens rather than accept them silently. Silent
   acceptance of `200.16"` for `200.16'` is the failure to design against.
3. **Curve/line table schema is per-plat**, not canonical.

## Candidate interface the production design must accommodate

Grounded in what this run actually had to carry:

- Source digest, media type, and byte length; page or frame index and page count.
- Normalizer identity and version, render scale/DPI, and pixel dimensions.
- EXIF or page rotation as found, whether a transpose was applied, and the
  **inverse transform** — without it, coordinates from an EXIF-rotated JPEG land
  in a different frame than the one the reviewer sees.
- The page-to-pixel affine, retained rather than recomputed.
- OCR engine identity and version, and per-region: raw recognized text, score,
  and **the raw region as an oriented quadrilateral**, unreduced.
- Geometry candidates in a channel that is never promoted to an extracted
  boundary, with detector identity and parameters.
- A **suspect-token signal** per recovered value, so unit marks and DMS
  punctuation that failed to recognize are surfaced for review (decision 2).
- Per-call retention of **both** grid and ground observations with an explicit
  authority marker (decision 1).
- Table cells with a **per-plat, reviewer-confirmed** column mapping (decision 3).

### Concrete gaps in today's code

| gap | location | why it blocks |
|---|---|---|
| `Parser.parse()` takes PDF bytes only | `src/docproc/adapters/base.py:32` | TIFF and JPEG have no entry point at all |
| `BBox` is an axis-aligned 4-tuple | `src/docproc/models.py:57-66` | cannot store an oriented quadrilateral |
| `BBoxSource` is `estimated`\|`measured` | `src/docproc/models.py:68-78` | a detector-inferred region is neither |
| `VisualAnnotation` has no region channel | `src/docproc/models.py:292-300` | nowhere to hang a geometry candidate |

## Failure modes the design must accommodate

1. Silent unit corruption — foot mark read as inch mark.
2. DMS punctuation dropped, rendering a bearing unparseable.
3. Unicode confusables in the degree sign.
4. Neighbouring annotation absorbed into a call's region, a direct consequence
   of axis-aligned regions over rotated text.
5. Rotation-dependent recall collapse — engine-specific and severe.
6. Standalone short rotated tokens (curve references) missed entirely.
7. Lossy compression compounding all of the above.
8. Table structure absent; only loose cell text available.
9. Association capture by parallel non-boundary lines.
10. Geometry candidate sets orders of magnitude larger than the boundary.
11. Multiple parcels and dense non-boundary text on one sheet.
12. Paired grid/ground values for one physical line.

## Non-claims

No external corpus was acquired. No accuracy claim is made about real plats, and
the synthetic scores measure recovery of planted content only. No association
capability is asserted — the naive rule exists to quantify its failure, not to
propose it. Neither engine was tuned; a different Tesseract page-segmentation
mode or a preprocessing pass would change its numbers, and the comparison is not
offered as an engine benchmark.

## Reproducing

```bash
uv run --extra plats python prototypes/plat_extraction.py sheets --out /private/tmp/plat108
uv run --extra plats --with ocrmac python prototypes/plat_extraction.py extract /private/tmp/plat108/sheet.pdf --out /private/tmp/plat108/pdf
uv run --extra plats python prototypes/plat_extraction.py score /private/tmp/plat108/pdf/evidence.json /private/tmp/plat108/ground_truth.json
```

`--with ocrmac` (Apple Vision) is macOS-only; omit it elsewhere.

Tesseract must be on `PATH` (`brew install tesseract`); without it that engine
reports as unavailable and the run continues. Outputs are written outside the
repository by design.

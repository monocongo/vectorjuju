# Docling capabilities for survey plat sheets

Research for [Research docling capabilities for plat sheets](https://github.com/monocongo/vectorjuju/issues/4).

- **Scope:** what docling can do with a single-page survey plat supplied as a JPG, TIFF, or PDF.
- **Method:** read-only review of primary sources — docling docs and source, docling-core source, and Hugging Face model cards.
- **Not done:** docling was not installed or run. Nothing here is a measurement; that is the job of [Prototype docling extraction on synthetic and private plats](https://github.com/monocongo/vectorjuju/issues/7).

**Versions reviewed:**

- docling `v2.127.0`, released 2026-09-14, which is the latest release. Its PyPI distributions are `docling` and `docling-slim`.
- docling-core `v2.96.1`, the latest release.
- Hugging Face model cards at `main`.

**Starting lead:** docproc's paper review of docling `v2.124.0` in [`stronger-plat-extraction.md`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/docs/research/stronger-plat-extraction.md). Each of its claims below was re-checked against `v2.127.0` source.

Where a primary source is silent, this doc says **undocumented**. Statements marked **(inference)** are derived from the cited facts and are not stated by any source.

## Answers

### 1. Does docling accept JPG and TIFF directly?

**Yes, no PDF wrapping needed.**

- The supported input formats list includes "PNG, JPEG, TIFF, BMP, WEBP". [S1]
- `InputFormat.IMAGE` maps the extensions `jpg, jpeg, png, tif, tiff, bmp, webp` and the matching MIME types. [S2]

**Multi-page TIFF.** Each frame becomes its own page. The image backend "handles multi-page TIFF by extracting frames eagerly", iterating `n_frames`. [S3]

**Page size and DPI.**

- Page coordinates use 72 points per inch.
- For images, embedded DPI metadata sets the physical page size. Missing DPI, or DPI of `(1, 1)`, is treated as 72 DPI. [S4, S3]
- So a JPG with no DPI metadata has 1 page point = 1 source pixel.

**EXIF orientation: not handled.**

- The image backend converts each frame with `convert("RGB")` and has no EXIF or `transpose` handling. [S3]
- **(inference)** An EXIF-rotated JPG reaches OCR in its stored orientation. vectorjuju should apply the EXIF transpose itself before handing an image to docling.

### 2. Which OCR engines, how heavy, and how is rotated text handled?

**Engines.** The model catalog lists Auto (default), Tesseract (CLI or Python bindings), EasyOCR, RapidOCR (ONNX, OpenVINO, PaddlePaddle), macOS Vision, and SuryaOCR. [S5, S6]

Each engine is installed through its own extra: [S7, S8]

| Engine | Extra | Notes |
|---|---|---|
| ocrmac | `docling[ocrmac]` | darwin only; wraps the Apple Vision framework |
| RapidOCR + ONNX Runtime | `docling[rapidocr]` | — |
| EasyOCR | `docling[easyocr]` | — |
| tesserocr | `docling[tesserocr]` | needs a system Tesseract install |
| Nemotron | — | Linux x86_64 on Python 3.12 only |

**What `Auto` picks.** On macOS it tries ocrmac first. After that it tries Nemotron, then RapidOCR with onnxruntime, then EasyOCR, then RapidOCR with torch, taking the first engine that is installed and supports the requested language. [S9]

**What the plain install pulls.**

- `pip install docling` depends on `docling-slim[standard]`. [S8]
- `standard` includes `models-local` and `feat-ocr-rapidocr`. [S7]
- `models-local` brings `torch`, `torchvision`, `docling-ibm-models`, `accelerate`, and `huggingface_hub`. [S7]
- `feat-ocr-rapidocr` brings RapidOCR but not onnxruntime. [S7]

**Weight.** The only size figure in the sources is the comment on the minimal `docling-slim` base: "8 packages – ~50MB". [S7] Installed size per OCR engine, and for the full `docling` package, is **undocumented**.

**OCR resolution.** `OcrOptions.scale` defaults to `3.0`, which renders at 216 DPI. [S10]

**Rotated text: no per-label angle survives.**

- RapidOCR and EasyOCR return quadrilaterals. docling keeps only corners 0 and 2 and builds an axis-aligned box from them. [S11, S12]
- ocrmac builds an axis-aligned box from Vision's `x, y, w, h`. [S13]
- Only Tesseract runs page-level orientation detection (OSD), rotating the whole page before OCR. [S14] The shared rotation helper accepts only `0, 90, 180, 270` and raises on any other angle. [S15]
- RapidOCR exposes a `use_cls` text-direction classifier, described only as "Enable text direction classification stage". What it corrects upstream is **undocumented** in docling. [S16]
- docling-core's `BoundingRectangle` can hold an arbitrary rotated quad and exposes `angle`. [S17] But no OCR engine fills it with anything other than an axis-aligned box. [S11–S13]

**How docling handles text at arbitrary angles (e.g. a bearing label drawn along a 37° boundary) is undocumented.**

### 3. What table structure does it return, and does it handle rotated tables?

**Real cell structure.**

- TableFormer runs in `accurate` mode by default, with `fast` as the alternative. [S5, S18]
- Each `TableCell` carries `start_row_offset_idx`, `end_row_offset_idx`, `start_col_offset_idx`, `end_col_offset_idx`, `row_span`, `col_span`, `column_header`, `row_header`, `row_section`, and an optional `bbox`. [S19]
- `do_cell_matching` chooses whether cell text comes from page OCR cells (the default) or from TableFormer's own prediction. [S4, S18]

**Rotated tables: quarter turns only.**

- `TableData` has an `Orientation` of `ROT_0`, `ROT_90`, `ROT_180`, or `ROT_270`. [S19]
- The reading-order stage copies an element's `orientation` onto the table. [S20] Which stage sets a value other than `ROT_0` was not found in the source reviewed and is **undocumented**. docproc's `v2.124.0` review found it only ever defaulted.
- Tables at other angles are **undocumented**.
- There is no per-table confidence here. docproc's lead reported `table_score` as not implemented. This was not re-checked, because the question didn't ask about confidence.

### 4. What coordinate space do boxes use, and can they be mapped to page pixels?

**Units.**

- Page units are points at 72 per inch. For image input the page size comes from DPI. [S4, S3]
- `BoundingBox` has an explicit `coord_origin` (`TOPLEFT` or `BOTTOMLEFT`) and the converters `to_top_left_origin(page_height)` and `to_bottom_left_origin(page_height)`. [S21]

**Origins differ between stages.**

- OCR cells are built in `TOPLEFT` page coordinates. [S11–S13]
- The document output uses `BOTTOMLEFT`: the reading-order stage converts element and table-cell boxes with `to_bottom_left_origin(page_height)` before writing `ProvenanceItem(page_no, bbox, charspan)`. [S20, S22]

**Mapping back to pixels (inference).** It is exact arithmetic:

- `px_x = l × dpi_x / 72`
- `px_y = (page_height − t) × dpi_y / 72`, for a `BOTTOMLEFT` box.

**Precision for binding labels to traced lines (inference).**

- An axis-aligned box around a label drawn at angle θ grows with θ and overlaps neighbouring lines. Near 45° its centroid is still usable, but its extent is not.
- Binding by box centroid is plausible for upright or quarter-turned labels. For diagonal labels it needs measuring in the prototype.
- docling reports no per-label angle to help.

### 5. Install footprint on Apple Silicon, and model licenses?

**Device choice.**

- `AcceleratorOptions.device` accepts `auto`, `cpu`, `cuda`, `mps`, and `xpu`. `auto` picks MPS when torch reports it is built and available. [S23, S24]
- TableFormer overrides MPS back to CPU: "Disable MPS here, until we know why it makes things slower." [S25]
- Runtime on CPU vs. MPS for the other stages is **undocumented**.

**Offline use.** Models download on first use. `docling-tools models download` plus `artifacts_path` or `DOCLING_ARTIFACTS_PATH` prefetches them for offline use. [S4]

**Licenses.**

| Component | License |
|---|---|
| docling code | MIT [S26] |
| `docling-project/docling-models` card | `cdla-permissive-2.0` and `apache-2.0` [S27] |
| `docling-layout-heron` (the default layout model) | `apache-2.0` [S28, S5] |

- The `docling-models` card does not say which model carries which of its two licenses. [S27]
- The model cards were read at `main`, a moving revision. Pin a revision for reproducible results.
- Licenses of the OCR engines' own models (RapidOCR, EasyOCR, Apple Vision) are outside docling's docs and **undocumented** here.

## Implications for vectorjuju (inference)

- **Give docling images, not PDFs**, after vectorjuju itself applies the EXIF transpose and sets DPI.
- **Engine choice:** use `ocrmac` locally on macOS; use RapidOCR (ONNX) or tesserocr on Linux CI.
- **Curve tables** are usually upright, so TableFormer's cell structure is a plausible fit.
- **Bearing and distance labels drawn along diagonal boundaries** are the main risk: docling returns no angle and only axis-aligned boxes. One mitigation is opencv crops of each label, de-rotated before OCR. The prototype should measure both.
- **Pin versions:** docling, docling-core, and the Hugging Face model revisions.

## Sources

Links point at docling [`v2.127.0`](https://github.com/docling-project/docling/tree/v2.127.0) and docling-core [`v2.96.1`](https://github.com/docling-project/docling-core/tree/v2.96.1) unless noted.

- **[S1]** [docs/usage/supported_formats.md](https://github.com/docling-project/docling/blob/v2.127.0/docs/usage/supported_formats.md#L24)
- **[S2]** [docling/datamodel/base_models.py L150–219](https://github.com/docling-project/docling/blob/v2.127.0/docling/datamodel/base_models.py#L150-L219)
- **[S3]** [docling/backend/image_backend.py](https://github.com/docling-project/docling/blob/v2.127.0/docling/backend/image_backend.py): DPI L30–53; frames L179–226
- **[S4]** [docs/usage/advanced_options.md](https://github.com/docling-project/docling/blob/v2.127.0/docs/usage/advanced_options.md): offline models L1–67; image resolution L105–109; table options L111–140
- **[S5]** [docs/usage/model_catalog.md L14–85](https://github.com/docling-project/docling/blob/v2.127.0/docs/usage/model_catalog.md#L14-L85)
- **[S6]** [docs/concepts/OCR.md L1–65](https://github.com/docling-project/docling/blob/v2.127.0/docs/concepts/OCR.md#L1-L65)
- **[S7]** [pyproject.toml (`docling-slim`)](https://github.com/docling-project/docling/blob/v2.127.0/pyproject.toml): base deps L48–59; OCR extras; `models-local`; `standard`/`all` L306–317
- **[S8]** [packages/docling/pyproject.toml](https://github.com/docling-project/docling/blob/v2.127.0/packages/docling/pyproject.toml)
- **[S9]** [docling/models/stages/ocr/auto_ocr_model.py L67–178](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/ocr/auto_ocr_model.py#L67-L178)
- **[S10]** [docling/datamodel/pipeline_options.py `OcrOptions` L192–282](https://github.com/docling-project/docling/blob/v2.127.0/docling/datamodel/pipeline_options.py#L192-L282)
- **[S11]** [docling/models/stages/ocr/rapid_ocr_model.py L687–725](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/ocr/rapid_ocr_model.py#L687-L725)
- **[S12]** [docling/models/stages/ocr/easyocr_model.py L343–358](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/ocr/easyocr_model.py#L343-L358)
- **[S13]** [docling/models/stages/ocr/ocr_mac_model.py L195–218](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/ocr/ocr_mac_model.py#L195-L218)
- **[S14]** [docling/models/stages/ocr/tesseract_ocr_model.py L190–280](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/ocr/tesseract_ocr_model.py#L190-L280)
- **[S15]** [docling/utils/orientation.py](https://github.com/docling-project/docling/blob/v2.127.0/docling/utils/orientation.py)
- **[S16]** [docling/datamodel/pipeline_options.py `RapidOcrOptions.use_cls` L403–408](https://github.com/docling-project/docling/blob/v2.127.0/docling/datamodel/pipeline_options.py#L403-L408)
- **[S17]** [docling_core/types/doc/page.py `BoundingRectangle` L104–203, `TextCell` L281](https://github.com/docling-project/docling-core/blob/v2.96.1/docling_core/types/doc/page.py#L104-L203)
- **[S18]** [docling/datamodel/pipeline_options.py `TableFormerMode`/`TableStructureOptions` L121–171](https://github.com/docling-project/docling/blob/v2.127.0/docling/datamodel/pipeline_options.py#L121-L171)
- **[S19]** [docling_core/types/doc/items/table/table_data.py `TableCell` L20–33, `Orientation` L86–97](https://github.com/docling-project/docling-core/blob/v2.96.1/docling_core/types/doc/items/table/table_data.py#L20-L97)
- **[S20]** [docling/models/stages/reading_order/readingorder_model.py](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/reading_order/readingorder_model.py): origin conversion L69, L139–142, L394–397; orientation L262–269
- **[S21]** [docling_core/types/doc/base.py `CoordOrigin` L17–21, converters L207, L245](https://github.com/docling-project/docling-core/blob/v2.96.1/docling_core/types/doc/base.py#L17-L21)
- **[S22]** [docling_core/types/doc/common/reference.py `ProvenanceItem` L183–193](https://github.com/docling-project/docling-core/blob/v2.96.1/docling_core/types/doc/common/reference.py#L183-L193)
- **[S23]** [docling/datamodel/accelerator_options.py L19–58](https://github.com/docling-project/docling/blob/v2.127.0/docling/datamodel/accelerator_options.py#L19-L58)
- **[S24]** [docling/utils/accelerator_utils.py L13–54](https://github.com/docling-project/docling/blob/v2.127.0/docling/utils/accelerator_utils.py#L13-L54)
- **[S25]** [docling/models/stages/table_structure/table_structure_model.py L83–87](https://github.com/docling-project/docling/blob/v2.127.0/docling/models/stages/table_structure/table_structure_model.py#L83-L87)
- **[S26]** [LICENSE](https://github.com/docling-project/docling/blob/v2.127.0/LICENSE)
- **[S27]** [Hugging Face `docling-project/docling-models` model card](https://huggingface.co/docling-project/docling-models/blob/main/README.md)
- **[S28]** [Hugging Face `docling-project/docling-layout-heron` model card](https://huggingface.co/docling-project/docling-layout-heron/blob/main/README.md)

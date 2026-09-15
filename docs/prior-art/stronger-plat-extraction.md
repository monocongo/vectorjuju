> Carried from docproc [`docs/research/stronger-plat-extraction.md`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/docs/research/stronger-plat-extraction.md) at `5b53f9a125b9ee94517aea3179a82007b4a28aaf`.

# Stronger extraction options for scanned survey plats

**Scope.** This answers [#117](https://github.com/monocongo/docproc/issues/117):
what first-party materials establish about a local VLM (Qwen2.5-VL via Ollama),
Docling with TableFormer, and three hosted document-AI services (AWS Textract,
Google Document AI, Azure Document Intelligence), and what each implies for an
auditable adapter. It builds on
[#108](https://github.com/monocongo/docproc/issues/108), which found that the
local raster stack recovers no boundary call verbatim, supplies no table
structure, and loses call-to-line association to parallel lines.

This is a documentation-only assessment. It acquired no corpus, uploaded
nothing, installed no model, called no extraction service, and makes no accuracy
claim. Every source is an official vendor document, model card, license, API
reference, paper, or official repository at a pinned tag or commit. All sources
were retrieved 2026-09-10. "Not established" means the cited sources do not say,
not that the capability is absent.

## Answer

| | Rotated / near-vertical dense text, DMS and foot marks | Region shape | Table row/column/cell structure | Signals to highlight suspect values | Local requirements / hosted data terms |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-VL via Ollama** (local) | Card claims text, chart, and layout analysis and structured output for tables and forms. No rotated-text or symbol statement. [Q1, Q2] | Axis-aligned `x1 y1 x2 y2` in the resized input image's absolute pixels. Model-proposed, not detector-measured. [Q2 §2.2.1, Q1] | Generated structure (for example QwenVL HTML `<table>`). No cell-index contract. [Q2] | Per generated token `logprobs` / `top_logprobs` from Ollama. Not an OCR confidence. [O2] | 3.2–49 GB per tag; macOS 14+ on Apple M series. 3B is non-commercial; 7B/32B are Apache-2.0; 72B is the Qwen License. Ollama can offload to its cloud unless disabled. [O1, O3, O4, Q3] |
| **Docling 2.124.0 + TableFormer** (local, already pinned) | Only whole-page orientation: Tesseract OSD at 0/90/180/270. No per-label angle and no symbol statement. [D4] | Output provenance is an axis-aligned `BoundingBox`. OcrMac and RapidOCR cells are built from axis-aligned corners, and RapidOCR's quadrilateral is discarded. [D4, D6] | Real cells: row/column offsets, spans, `column_header`, `row_header`. Table `orientation` exists but is only ever defaulted to `ROT_0`. [D6, D1] | Page/document grades (`ocr_score`, `layout_score`); `table_score` not implemented. Per-cell OCR confidence is only in intermediate pages (`generate_parsed_pages`, default off). [D2, D4] | TableFormer on CPU/CUDA/XPU only (MPS disabled). Weights are downloaded on first use or prefetched for air-gapped use. MIT code; CDLA-Permissive-2.0/Apache-2.0 weights. The default layout weights track the mutable `main` revision. [D1, D3, D5, D7] |
| **AWS Textract** (hosted) | All in-plane *document* rotations are supported (for example 45°). `RotationAngle` per WORD is only 0/90/180/270. Its character list has `'` (U+0027), `"` (U+0022), and `°`, but no prime marks. [A1, A2] | `Polygon` ("fine-grained polygon") plus an axis-aligned `BoundingBox`. [A2, A3] | `CELL` with `RowIndex`/`ColumnIndex`; `MERGED_CELL` spans; `COLUMN_HEADER`, title, and footer entity types. Its FAQ says tables work best when text is upright. [A4, A6] | `Confidence` 0–100 per WORD, LINE, CELL, and TABLE. [A3, A4, A6] | Content may be used to improve AWS AI and **stored in another region** unless you opt out through AWS Organizations. Async results are kept 7 days. Base model version cannot be selected; it is returned as `AnalyzeDocumentModelVersion`. [A5, A6, A7, A8, A9] |
| **Google Document AI** (hosted) | "Rotation correction" preprocesses images. "Even for rotated texts" applies to *native PDF text*, not OCR. Orientation is one of four quadrants. No symbol statement. [G1, G2] | `boundingPoly` vertices, a generic polygon (shape not specified). [G2] | Form Parser rows have `rowSpan`/`colSpan` always 1 (conventional tables only). Layout Parser table blocks carry spans; v1.5+ is Gemini-based and release-candidate. [G3, G4, G2] | `confidence` [0,1] per layout element, including tokens; optional symbols; page-level image-quality scores. [G1, G2, G3] | Online requests are processed in memory and not persisted; batch has a TTL of up to one day; "never use customer data to improve our models". Versions are selectable but deprecate, and Google recommends not pinning. [G5, G6, G7] |
| **Azure Document Intelligence** (hosted, or container) | Page `angle` is a float in (−180, 180]. The Layout docs say "supports extracting tables that are rotated". No symbol statement. [M1, M3] | `polygon` vertices "clockwise from the left … relative to the element orientation" for words, lines, and cells. [M3] | `rowCount`/`columnCount`; cells with `rowIndex`, `columnIndex`, `rowSpan`, `columnSpan`, and `kind` (`columnHeader`, `rowHeader`, `stubHead`, …). [M1, M3] | Per-word `confidence` [0,1]. [M2, M3] | Input and results are stored 24 h in the resource's region, with a delete API. Results carry `apiVersion` and `modelId`. v4.0 Read/Layout **containers** keep content local but require continuous metering; fully disconnected use needs Microsoft approval. [M3, M4, M5, M6] |

**Bottom line.** Only the two local options satisfy "no source leaves the
deployment" by default, and among vendor models only an Azure container can run
in-deployment. The hosted services document the strongest geometry and table
contracts: non-axis-aligned polygons, explicit cell indices, and per-word
confidence. No candidate documents recovery of DMS punctuation or foot marks,
and none documents text-to-linework association. That leaves #108's association
finding untouched.

## Local VLM: Qwen2.5-VL via Ollama

**Established.**

- The model card claims the model is "highly capable of analyzing texts, charts,
  icons, graphics, and layouts within images", "can accurately localize objects
  in an image by generating bounding boxes or points", and "supports structured
  outputs" for "scans of invoices, forms, tables". [Q1]
- The technical report says grounding uses "absolute position coordinates" based
  on input image dimensions, and document parsing emits `data-bbox="x1 y1 x2 y2"`
  per element. [Q2 §2.2.1] Both are two-corner, axis-aligned boxes.
- Input images are resized to keep aspect ratio within `min_pixels`/`max_pixels`,
  with dimensions rounded to multiples of 28. [Q1] Returned coordinates therefore
  live in a resized frame.
- Licensing differs by size. 3B uses the Qwen Research License, "FOR
  NON-COMMERCIAL PURPOSES ONLY". 7B and 32B are `apache-2.0`. 72B uses the Qwen
  License, which requires a separate license above 100 million monthly active
  users. [Q3]
- Ollama publishes `qwen2.5vl` tags of 3.2 GB (3b), 6.0 GB (7b), 21 GB (32b),
  and 49 GB (72b), and states the model "requires Ollama 0.7.0". [O1] Ollama
  needs macOS 14+ with Apple M series (CPU and GPU) and warns that models "can
  be tens to hundreds of GB". Models are stored in `~/.ollama/models`,
  relocatable with `OLLAMA_MODELS`. [O3, O4] Ollama is MIT-licensed. [O7]
- The Ollama API accepts base64 images, enforces JSON schemas via `format`, and
  returns per-output-token `logprobs` with optional `top_logprobs`. [O2, O6]
  `/api/tags` reports a SHA256 `digest` of model contents and can list
  `remote_model`/`remote_host` entries. [O2]
- "Cloud models are automatically offloaded to Ollama's cloud service". Local-only
  mode is `OLLAMA_NO_CLOUD=1` or `disable_ollama_cloud`. [O4, O5] On macOS and
  Windows, Ollama "will automatically download updates". [O4]

**Not established.** Recovery of rotated or near-vertical text. Correct output of
`°`, `′`/`'`, or `″`/`"`. That generated text is a verbatim transcription of the
glyphs inside the proposed box. That token log-probabilities are calibrated
indicators of recognition error. Whether Ollama resizes images the way the
reference `transformers` processor does. The quantization of Ollama library tags.
Whether a tag name always resolves to the same digest.

## Docling 2.124.0 with TableFormer

The repository pins `docling==2.124.0` (lock: `docling-core` 2.92.0,
`docling-ibm-models` 4.0.0) as an optional extra.
[`pyproject.toml`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/pyproject.toml)

**Established.**

- Inputs include PDF and PNG, JPEG, TIFF, BMP, and WEBP. For image inputs,
  "missing DPI and `(1, 1)` DPI are treated as 72 DPI". [D3, D8]
- TableFormer "recognizes table structure (rows, columns, cells) and
  relationships". The paper frames it as generic image-based table-structure
  identification. [D1, D9] `TableCell` carries `start/end_row_offset_idx`,
  `start/end_col_offset_idx`, `row_span`, `col_span`, `column_header`, and
  `row_header`. [D6]
- `TableData.orientation` supports `ROT_0`/`90`/`180`/`270`. In Docling v2.124.0
  source, the only assignment found is the `ROT_0` default. [D6, D4]
- Geometry is axis-aligned end to end for plats:
  - `ProvenanceItem.bbox` is a `BoundingBox`. [D6]
  - docling-core's `TextCell.rect` is an "Oriented rectangle" with four corners,
    but the OcrMac adapter fills it from `(left, top, right, bottom)`. [D6, D4]
  - The RapidOCR adapter keeps only corners 0 and 2 of RapidOCR's quadrilateral.
    [D4]
  - The Tesseract adapter rotates cells only by the page-level OSD angle
    (0/90/180/270). [D4]
- Confidence: `ConversionResult.confidence` gives page- and document-level
  grades. Numeric scores "are for informational purposes only; their computation
  and weighting may change", and `table_score` is "not yet implemented". [D2]
  Per-cell OCR `confidence` lives on intermediate page cells. Those are retained
  only with `generate_parsed_pages=True` (default `False`). [D4]
- Hardware: TableFormer supports CPU, CUDA, and XPU; "MPS is currently disabled
  for TableFormer due to performance issues". [D1]
- Model storage: models "are downloaded automatically upon first usage", or
  prefetched with `docling-tools models download` into
  `$HOME/.cache/docling/models` and passed via `artifacts_path` for air-gapped
  use. [D3]
- Sending data to remote services requires `enable_remote_services=True`;
  otherwise `OperationNotAllowed()` is raised. [D3]
- Weight revisions: TableFormer downloads `docling-project/docling-models` at
  `revision="v2.3.0"`. The default layout model (Heron) uses `revision="main"`.
  [D5]
- Licenses: Docling is MIT. `docling-models` declares `cdla-permissive-2.0` and
  `apache-2.0`. Heron is `apache-2.0`. [D7]

**Not established.** Arbitrary-angle label recovery. DMS or foot-mark recognition
by any Docling OCR engine. TableFormer behavior on rotated tables or
engineering-drawing tables. On-disk size of the model set.

## AWS Textract

**Established.**

- Rotation: "supports all in-plane document rotations, for example 45-degree
  in-plane rotation". "Horizontally arrayed text can be read regardless of the
  degree of rotation of a document", but vertical (CJK-style) text is not
  supported. [A1] `Geometry.RotationAngle` "corresponding to the rotation of the
  WORD block" takes only 0, 90, 180, or 270. [A2]
- Limits: minimum text height is 15 pixels (about 8 pt at 150 DPI). Images must
  be at most 10,000 px on all sides. Supported formats are JPEG, PNG, PDF, and
  TIFF. [A1]
- Characters: the documented list includes U+0027 `'`, U+0022 `"`, and U+00B0
  `°`. It lists no U+2032/U+2033 prime characters. [A1]
- Geometry: `BoundingBox` is "axis-aligned coarse". `Polygon` is "a fine-grained
  polygon around the recognized item". [A2, A3]
- Tables: `CELL` blocks carry `RowIndex` and `ColumnIndex` and always have span 1.
  `MERGED_CELL` blocks carry spans. Cells may be `COLUMN_HEADER`,
  `TABLE_SECTION_TITLE`, `TABLE_SUMMARY`, and more. Tables are
  `STRUCTURED_TABLE` or `SEMI_STRUCTURED_TABLE`. [A4] The FAQ says tables work
  best when "the text within the table is upright (e.g. not rotated relative to
  other text on the page)". [A6]
- Confidence: 0–100 on blocks, including WORD and CELL. [A3, A4, A6]
- Model identity: `AnalyzeDocument` returns `AnalyzeDocumentModelVersion`. The
  request has no base-model version selector, only an optional
  `AdaptersConfig.Adapters[].Version`. [A5]
- Data use: Service Terms §50.3 lets AWS "use and store AI Content … to develop
  and improve" Textract and "store such AI Content in an AWS region outside of
  the AWS region where you are using" it, unless an AWS Organizations AI services
  opt-out policy is set. [A7, A6] Opting out deletes stored improvement copies,
  "but any content that is used to provide the service to you is not deleted".
  [A8]
- Retention: async results "are encrypted and stored for 7 days in a Amazon
  Textract owned bucket by default" unless `OutputConfig` names your bucket. [A9]
- Transport: PrivateLink keeps VPC-to-Textract traffic on the AWS network. [A10]

**Not established.** Recognition of text at different arbitrary angles *within*
one page (the rotation statement is about the document). Whether `'`/`"` are
distinguished reliably as foot/inch marks. Whether the model behind
`AnalyzeDocumentModelVersion` changes without notice.

## Google Document AI

**Established.**

- Enterprise Document OCR offers "Rotation correction … to correct rotation
  issues" in images. Its "even for rotated texts" claim is scoped to extracting
  embedded text from digital PDFs. [G1] `Layout.orientation` is one of
  `PAGE_UP`, `PAGE_RIGHT`, `PAGE_DOWN`, or `PAGE_LEFT`. [G2]
- It accepts PDF, GIF, TIFF, JPEG, PNG, BMP, and WebP, and returns page-level
  image-quality scores "in eight dimensions" plus optional symbol-level output.
  [G1, G3]
- Geometry: `Layout.boundingPoly` holds "the bounding polygon vertices" and
  normalized vertices. `Layout.confidence` is in [0,1] "for a single token, a
  table, a visual element, etc." [G2]
- Tables: Form Parser "only recognizes conventional tables, those without cells
  that span rows or columns. So `rowSpan` and `colSpan` are always 1". [G3]
  Layout Parser `LayoutTableBlock` has header and body rows whose cells carry
  spans, and each block has a `boundingBox`. [G2] Layout Parser versions 1.5 and
  later are Gemini-based release candidates. [G4]
- Versions: "Earlier Google stable versions are deprecated six months after the
  newest stable version is released". Calls that specify a deprecated version
  fail. Google recommends "to not specify the processor version ID in your
  production requests". [G5] OCR v1.2 is a frozen snapshot "for up to 18 months"
  (Advanced versioning, Preview). [G1]
- Data: online requests are "processed in memory, encrypted in flight, and not
  persisted to disk". Batch documents are "typically deleted immediately after
  the processing, with a failsafe Time to live (TTL) of one day". "We never use
  customer data to improve our models." [G6] A regional or multi-regional
  location must be specified. [G7]

**Not established.** Whether `boundingPoly` follows rotated glyphs or is an
axis-aligned rectangle. Any per-label arbitrary-angle recovery. DMS or foot-mark
character support. Whether responses record the processor version that served
them. A stated residency guarantee on the regions page.

## Azure Document Intelligence (v4.0, `2024-11-30`)

**Established.**

- Read and Layout accept PDF, JPEG, PNG, BMP, TIFF, and HEIF from 50×50 to
  10,000×10,000 px. Minimum text height is 12 px on a 1024×768 image, about 8 pt
  at 150 DPI. [M2]
- Geometry: word, line, and cell `polygon` "vertices, clockwise from the left
  (-180 degrees inclusive) relative to the element orientation". `DocumentPage.angle`
  is "the general orientation of the content in clockwise direction, measured in
  degrees between (-180, 180]". [M3]
- Words carry `confidence` in [0,1]. [M2, M3]
- Tables: `rowCount` and `columnCount`, with cells carrying `rowIndex`,
  `columnIndex`, `rowSpan`, `columnSpan`, and `kind` (`content`, `rowHeader`,
  `columnHeader`, `stubHead`, `description`). "The model supports extracting
  tables that are rotated." [M1, M3]
- Identity: `AnalyzeResult` carries `apiVersion` and `modelId`
  (for example `prebuilt-layout`). [M3, M1] The FAQ states "We continually
  update and improve the Document Intelligence models." [M7]
- Data: "processed in the same region where the Document Intelligence resource
  was created". The service "stores submitted input data and analyze results for
  24 hours" and a Delete Analyze Result API purges earlier. [M4]
- Containers: v4.0 containers exist for Read and Layout. Layout needs at least 8
  cores and 16 GB, with 24 GB recommended. [M5] Connected containers "aren't
  licensed to run without being connected to Azure for metering" and "don't send
  customer data (for example, the image or text that is being analyzed) to
  Microsoft". [M5]
- Disconnected containers require an access request limited to strategic
  customers or partners, a commitment-tier purchase, and a license file with an
  expiration date, downloaded while connected. [M6]

**Not established.** Recovery of individual labels at arbitrary angles (the
angle is page-level, the polygon is element-relative). DMS or foot-mark
recognition. Whether `prebuilt-layout` under a fixed `apiVersion` is
behaviorally frozen. Whether customer content is used for model improvement (the
service privacy page is silent). Whether the v4.0 Layout image is approved for
disconnected use (that page's examples use 3.0 images).

## Implications for an auditable adapter

These follow from the sources above plus this repository's existing rules.
Processing behavior changes create a new Processing Definition. Model and prompt
changes are incompatible. Credentials, endpoints, storage locations, and machine
identities are runtime locators that must not enter the Cache Key.
[Markdown pipeline decision](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/docs/decisions/markdown-pipeline-and-evaluation-scope.md#non-deterministic-upstream-processing),
[`definition.py`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/src/docproc/definition.py). The existing
`VisionStageIdentity` names `llm_provider_id` and `llm_model_id` as plain
strings. [`vision/base.py`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/src/docproc/adapters/vision/base.py)

1. **Egress is binary per provider, and only local options pass by default.**
   - **Docling** passes, provided weights are prefetched through `artifacts_path`
     and `enable_remote_services` stays false. [D3]
   - **Ollama** passes, provided cloud features are disabled and no `remote_model`
     is selected. [O4, O5, O2]
   - **Textract, Document AI, and Azure's hosted API** all require the source
     bytes to reach the vendor. Region choice, PrivateLink, or opt-out narrows
     where the bytes go, not whether they leave. A hosted adapter can therefore
     only be an explicit per-deployment opt-in that is off by default and never
     exercised by tests or CI.
   - **Textract's default** permits cross-region storage for service improvement.
     Enabling it without an Organizations opt-out would contradict the rule on
     its own. [A7, A8]
   - **An Azure container** is the one vendor path whose documentation says
     analyzed content stays local. But the connected mode still requires
     metering egress, and disconnected mode is approval-gated. [M5, M6]
2. **A model name is not a model identity. Each provider needs a different
   identity field, and some cannot supply one.**
   - **Ollama:** the tag is a name. The Definition should carry the tag's SHA256
     `digest` [O2], and the run should record the Ollama version, because
     updates install automatically. [O4]
   - **Docling:** `docling==2.124.0` does not pin layout weights resolved from
     `main`. [D5] The Definition, or a recorded artifact digest, must cover the
     resolved weight revisions, OCR engine, and TableFormer mode. Bumping
     `PARSER_VERSION` only when the pip pin moves is not sufficient.
     [`docling_parser.py`](https://github.com/monocongo/docproc/blob/5b53f9a125b9ee94517aea3179a82007b4a28aaf/src/docproc/adapters/docling_parser.py)
   - **Textract:** the base model version is observable (`AnalyzeDocumentModelVersion`)
     but not selectable. [A5] A Definition cannot pin it. The adapter must persist
     the observed version per run, and a decision is needed on whether a changed
     observed version invalidates reuse.
   - **Document AI:** version IDs are selectable, but a pinned Definition stops
     executing after deprecation, and Google's recommended unpinned default is a
     mutable identity. [G5] Only an explicit version ID is compatible with
     Definition identity, and a lifetime limit must be expected.
   - **Azure:** `apiVersion` and `modelId` are selectable and echoed, [M3] but
     silent model updates under them are not ruled out. [M7] Containers need an
     image digest rather than a `:latest`-style tag. [M6]
   - **All hosted providers:** region and endpoint stay out of the Cache Key as
     runtime locators. They still belong in the run's authorization/provenance
     record, because they decide residency.
3. **Geometry must be stored as returned.** Textract `Polygon`, Azure `polygon`,
   and Document AI `boundingPoly` can express non-axis-aligned regions
   (Document AI's exact shape is not established). Docling and Qwen2.5-VL emit
   axis-aligned boxes. [A2, M3, G2, D6, Q2] The evidence model needs a polygon
   region with a provenance kind that separates detector-returned from
   model-proposed boxes, plus the transform back to source pixels. The Qwen
   resize and Docling's 72-DPI default both make that transform load-bearing.
   [Q1, D3] Today's axis-aligned `BBox` cannot hold any of this, as #108 found.
4. **Vendor confidence is necessary but not sufficient to highlight suspect
   values.** Per-word confidence is documented for Textract, Azure, and Document
   AI. Docling's is intermediate and off by default, and Ollama's is generation
   log-probability. [A3, M3, G2, D4, O2] No source ties any of these to
   punctuation-level errors. Textract's documented character set has no prime
   glyphs, so a foot/inch mark can only arrive as `'`/`"`. [A1] A
   repository-owned rule over the raw text must still flag unit and DMS
   punctuation for review, with vendor confidence as one input.
5. **Hosted table contracts are the strongest, but schema stays per-plat.**
   - **Explicit cell indices:** Textract and Azure return indices and header
     kinds, and Azure claims rotated tables. [A4, M1]
   - **Weaker grids:** Document AI's Form Parser has no spans, and Layout
     Parser's span-bearing tables come from Gemini-based release candidates.
     [G3, G4] Docling returns genuine cells but documents no rotated-table
     handling. [D6]
   - **Column meaning:** none maps header text to survey meaning. The per-plat,
     reviewer-confirmed column mapping from #108 remains the adapter's job.
6. **Association is unchanged.** No source for any candidate claims to link a
   text region or table row to drawn linework. Association stays a reviewable
   candidate whatever extractor is chosen.

## Sources

All retrieved 2026-09-10.

- **[Q1] Qwen**, [*Qwen2.5-VL-7B-Instruct* model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/README.md).
- **[Q2] Bai et al.**, [*Qwen2.5-VL Technical Report*, arXiv:2502.13923](https://arxiv.org/abs/2502.13923), §2.2.1.
- **[Q3] Qwen**, model licenses: [3B Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE); [7B card (`apache-2.0`)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/README.md); [32B card (`apache-2.0`)](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/blob/main/README.md); [72B Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/blob/main/LICENSE).
- **[O1] Ollama**, [`qwen2.5vl` library page](https://ollama.com/library/qwen2.5vl).
- **[O2] Ollama**, [OpenAPI specification](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/openapi.yaml) (`logprobs`, `top_logprobs`, `digest`, `remote_model`).
- **[O3] Ollama**, [macOS requirements](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/macos.mdx).
- **[O4] Ollama**, [FAQ](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/faq.mdx) (updates, model storage, disabling cloud).
- **[O5] Ollama**, [Cloud](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/cloud.mdx).
- **[O6] Ollama**, [Vision](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/capabilities/vision.mdx) and [Structured outputs](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/docs/capabilities/structured-outputs.mdx).
- **[O7] Ollama**, [LICENSE (MIT)](https://github.com/ollama/ollama/blob/b68b112bd8868d6278250d7d4bdfafa5cbf035c8/LICENSE).
- **[D1] Docling**, [model catalog, v2.124.0](https://github.com/docling-project/docling/blob/v2.124.0/docs/usage/model_catalog.md).
- **[D2] Docling**, [confidence scores, v2.124.0](https://github.com/docling-project/docling/blob/v2.124.0/docs/concepts/confidence_scores.md).
- **[D3] Docling**, [advanced options, v2.124.0](https://github.com/docling-project/docling/blob/v2.124.0/docs/usage/advanced_options.md) (offline models, remote services, image DPI).
- **[D4] Docling source, v2.124.0**: [`rapid_ocr_model.py` L533–556](https://github.com/docling-project/docling/blob/v2.124.0/docling/models/stages/ocr/rapid_ocr_model.py#L533-L556); [`ocr_mac_model.py` L121–135](https://github.com/docling-project/docling/blob/v2.124.0/docling/models/stages/ocr/ocr_mac_model.py#L121-L135); [`tesseract_ocr_cli_model.py` L320–397](https://github.com/docling-project/docling/blob/v2.124.0/docling/models/stages/ocr/tesseract_ocr_cli_model.py#L320-L397); [`utils/orientation.py`](https://github.com/docling-project/docling/blob/v2.124.0/docling/utils/orientation.py); [`datamodel/base_models.py`](https://github.com/docling-project/docling/blob/v2.124.0/docling/datamodel/base_models.py) (`Page.cells`, table `orientation` default); [`pipeline_options.py` `generate_parsed_pages`](https://github.com/docling-project/docling/blob/v2.124.0/docling/datamodel/pipeline_options.py#L2064-L2073).
- **[D5] Docling source, v2.124.0**: [`table_structure_model.py` L103–112](https://github.com/docling-project/docling/blob/v2.124.0/docling/models/stages/table_structure/table_structure_model.py#L103-L112); [`layout_model_specs.py` L82–86](https://github.com/docling-project/docling/blob/v2.124.0/docling/datamodel/layout_model_specs.py#L82-L86); [`pipeline_options.py` L1532 (Heron default)](https://github.com/docling-project/docling/blob/v2.124.0/docling/datamodel/pipeline_options.py#L1532).
- **[D6] docling-core source, v2.92.0**: [`page.py` (`BoundingRectangle`, `TextCell`)](https://github.com/docling-project/docling-core/blob/v2.92.0/docling_core/types/doc/page.py#L103-L320); [`reference.py` (`ProvenanceItem`)](https://github.com/docling-project/docling-core/blob/v2.92.0/docling_core/types/doc/common/reference.py#L182-L192); [`table_data.py` (`TableCell`, `Orientation`)](https://github.com/docling-project/docling-core/blob/v2.92.0/docling_core/types/doc/items/table/table_data.py#L20-L107).
- **[D7] Docling licenses**: [Docling LICENSE (MIT)](https://github.com/docling-project/docling/blob/v2.124.0/LICENSE); [`docling-models` card](https://huggingface.co/docling-project/docling-models/blob/main/README.md); [`docling-layout-heron` card](https://huggingface.co/docling-project/docling-layout-heron/blob/main/README.md).
- **[D8] Docling**, [supported formats, v2.124.0](https://github.com/docling-project/docling/blob/v2.124.0/docs/usage/supported_formats.md).
- **[D9] Nassar et al.**, [*TableFormer: Table Structure Understanding with Transformers*, arXiv:2203.01017](https://arxiv.org/abs/2203.01017).
- **[A1] AWS**, [Set quotas in Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/limits-document.html).
- **[A2] AWS**, [`Geometry` API reference](https://docs.aws.amazon.com/textract/latest/APIReference/API_Geometry.html).
- **[A3] AWS**, [Locating items on a document page](https://docs.aws.amazon.com/textract/latest/dg/text-location.html).
- **[A4] AWS**, [Tables](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html).
- **[A5] AWS**, [`AnalyzeDocument` API reference](https://docs.aws.amazon.com/textract/latest/APIReference/API_AnalyzeDocument.html).
- **[A6] AWS**, [Amazon Textract FAQs](https://aws.amazon.com/textract/faqs/).
- **[A7] AWS**, [AWS Service Terms](https://aws.amazon.com/service-terms/), §50 (last updated September 1, 2026).
- **[A8] AWS**, [AI services opt-out policies](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_ai-opt-out.html).
- **[A9] AWS**, [Calling Amazon Textract asynchronous operations](https://docs.aws.amazon.com/textract/latest/dg/api-async.html).
- **[A10] AWS**, [Amazon Textract and interface VPC endpoints](https://docs.aws.amazon.com/textract/latest/dg/vpc-interface-endpoints.html).
- **[G1] Google Cloud**, [Enterprise Document OCR](https://docs.cloud.google.com/document-ai/docs/enterprise-document-ocr).
- **[G2] Google Cloud**, [REST reference: `Document`](https://docs.cloud.google.com/document-ai/docs/reference/rest/v1/Document).
- **[G3] Google Cloud**, [Handle processing response](https://docs.cloud.google.com/document-ai/docs/handle-response).
- **[G4] Google Cloud**, [Process documents with Gemini layout parser](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk).
- **[G5] Google Cloud**, [Managing processor versions](https://docs.cloud.google.com/document-ai/docs/manage-processor-versions).
- **[G6] Google Cloud**, [Document AI security and compliance](https://docs.cloud.google.com/document-ai/docs/security).
- **[G7] Google Cloud**, [Regional and multi-regional support](https://docs.cloud.google.com/document-ai/docs/regions).
- **[M1] Microsoft**, [Document layout analysis (v4.0)](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0).
- **[M2] Microsoft**, [Read model OCR data extraction (v4.0)](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/read?view=doc-intel-4.0.0).
- **[M3] Microsoft**, [REST API 2024-11-30: Get Analyze Result](https://learn.microsoft.com/en-us/rest/api/aiservices/document-models/get-analyze-result?view=rest-aiservices-v4.0%20(2024-11-30)).
- **[M4] Microsoft**, [Data, privacy, and security for Document Intelligence](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/document-intelligence/data-privacy-security).
- **[M5] Microsoft**, [Install and run Document Intelligence containers (v4.0)](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-4.0.0).
- **[M6] Microsoft**, [Use Docker containers in disconnected environments](https://learn.microsoft.com/en-us/azure/ai-services/containers/disconnected-containers) and [Document Intelligence disconnected containers](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/disconnected?view=doc-intel-4.0.0).
- **[M7] Microsoft**, [Document Intelligence FAQ](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/faq?view=doc-intel-4.0.0).

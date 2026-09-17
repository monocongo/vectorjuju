"""Ingest source plats into one upright raster space and convert to CAD units.

Ingest normalizes a single-page PDF, JPG, or TIFF to one upright RGB raster
with the raster convention (top-left origin, y increasing downward). JPG and
TIFF pixels are used as-is after ``ImageOps.exif_transpose``; a PDF page is
rasterized at ``dpi``. ``px_to_cad``/``cad_to_px`` are the only coordinate
conversion point: raster pixels to CAD units (bottom-left origin, y up).

``convert()`` is the one end-to-end entry point: ingest -> trace -> OCR and
bind -> calibrate -> DXF and JSON sidecar. The heavy pieces are imported when
it runs, so ``import vectorjuju`` and ``import vectorjuju.pipeline`` stay free
of cv2, skimage, and docling.

There is no pixel-unit fallback: transforming before calibration raises.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from vectorjuju.errors import VectorjujuError

RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".tif", ".tiff"})
_RASTER_FORMATS = frozenset({"JPEG", "MPO", "TIFF"})
_PDF_SUFFIX = ".pdf"

# Pillow's MAX_IMAGE_PIXELS bounds decompression bombs, not process memory.
# Ingest peaks near 12 bytes/pixel (decoded, transposed, and RGB buffers), so
# cap accepted rasters at 256 MiB of peak allocations (~22M pixels).
MAX_INGEST_PIXELS = 256 * 1024 * 1024 // 12


class UnsupportedInputError(VectorjujuError):
    """Input is not a supported single-page PDF, JPG, or TIFF."""


def _require_fpp(fpp: float | None, caller: str) -> float:
    if fpp is None:
        raise ValueError(f"{caller} requires a calibrated fpp; pixel-unit output is not an option")
    if not math.isfinite(fpp) or fpp <= 0:
        raise ValueError(f"{caller} requires a finite fpp > 0, got {fpp!r}")
    return fpp


def _require_img_height(img_height: float, caller: str) -> float:
    if not math.isfinite(img_height) or img_height <= 0:
        raise ValueError(f"{caller} requires a finite img_height > 0, got {img_height!r}")
    return img_height


def load_raster(path: str | Path, dpi: int = 200) -> Image.Image:
    """Load a single-page PDF, JPG, or TIFF as one upright RGB raster.

    JPG/TIFF rasters are used as-is, with the EXIF orientation applied first
    (docling ignores EXIF, so it has to happen before anything downstream sees
    the image). A PDF page is rasterized at ``dpi``. The result keeps the
    raster convention: top-left origin, y increasing downward.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == _PDF_SUFFIX:
        return _load_pdf(path, dpi)
    if suffix in RASTER_SUFFIXES:
        return _load_image(path)
    raise UnsupportedInputError(f"unsupported input type {suffix or 'without extension'!r}: {path.name}")


def _load_pdf(path: Path, dpi: int) -> Image.Image:
    try:
        pdf = pdfium.PdfDocument(str(path))
    except MemoryError:
        raise
    except Exception as exc:
        raise UnsupportedInputError(f"cannot read PDF: {path.name}") from exc
    try:
        if len(pdf) != 1:
            raise UnsupportedInputError(f"PDF must have exactly one page, found {len(pdf)}: {path.name}")
        page = pdf[0]
        width_pt, height_pt = page.get_size()
        pixels = width_pt * height_pt * (dpi / 72) ** 2
        if pixels > MAX_INGEST_PIXELS:
            raise UnsupportedInputError(f"page too large to render at {dpi} dpi: {path.name}")
        return page.render(scale=dpi / 72).to_pil().convert("RGB")
    except UnsupportedInputError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise UnsupportedInputError(f"cannot render PDF: {path.name}") from exc
    finally:
        pdf.close()


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            if image.format not in _RASTER_FORMATS:
                raise UnsupportedInputError(f"expected a JPG or TIFF, decoded {image.format}: {path.name}")
            frames = getattr(image, "n_frames", 1)
            if frames != 1:
                raise UnsupportedInputError(f"image must have exactly one frame, found {frames}: {path.name}")
            if image.size[0] * image.size[1] > MAX_INGEST_PIXELS:
                raise UnsupportedInputError(f"image too large to decode: {path.name}")
            return ImageOps.exif_transpose(image).convert("RGB")
    except UnsupportedInputError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise UnsupportedInputError(f"cannot read image: {path.name}") from exc


def px_to_cad(pt: tuple[float, float], *, fpp: float | None = None, img_height: int) -> tuple[float, float]:
    """Raster px (top-left, y down) -> CAD units (bottom-left, y up)."""
    fpp = _require_fpp(fpp, "px_to_cad")
    img_height = _require_img_height(img_height, "px_to_cad")
    x, y = pt
    return x * fpp, (img_height - y) * fpp


def cad_to_px(pt: tuple[float, float], *, fpp: float | None = None, img_height: int) -> tuple[float, float]:
    """CAD units (bottom-left, y up) -> raster px (top-left, y down)."""
    fpp = _require_fpp(fpp, "cad_to_px")
    img_height = _require_img_height(img_height, "cad_to_px")
    x, y = pt
    return x / fpp, img_height - y / fpp


Units = Literal["us-survey-foot", "international-foot", "metre"]


def convert(
    input: str | Path,
    *,
    out: str | Path | None = None,
    dpi: int = 200,
    units: Units = "us-survey-foot",
    scale: float | None = None,
) -> Path:
    """Convert one survey plat sheet to a DXF drawing plus its JSON sidecar.

    ``input`` is a single-page PDF, JPG, or TIFF; a multi-page one, an
    unsupported extension, or a corrupt file raises ``UnsupportedInputError``.
    ``out`` names the DXF and defaults to ``input`` with a ``.dxf`` suffix; the
    sidecar is written beside it as ``<stem>.json``. ``dpi`` is the working
    resolution for both PDF rasterization and raster input, never read from
    image metadata. ``units`` is never inferred, and ``scale`` -- in units per
    pixel at ``dpi`` -- skips calibration and is recorded as
    ``scale.method == "override"``.

    When too little text parses into bound distance calls to calibrate,
    ``ScaleCalibrationError`` is raised and neither output is written: there is
    no pixel-unit fallback.

    Returns the DXF path; the sidecar is not part of the return value.
    """
    # Imported here, not at module scope: the ingest-only path must not need
    # docling/cv2/skimage, and this is the only caller that does.
    from vectorjuju.calibrate import Scale, calibrate_scale
    from vectorjuju.export import INSUNITS, write_outputs
    from vectorjuju.text import bind_calls, extract_text
    from vectorjuju.tracing import join_corners, trace_runs

    if units not in INSUNITS:
        raise ValueError(f"units must be one of {sorted(INSUNITS)}, got {units!r}")
    input = Path(input)
    image = load_raster(input, dpi=dpi)
    # Text is traced through, not masked out: a page-pass OCR box is an
    # axis-aligned bound on rotated call text, wide enough to blank the very
    # edge it labels (issue #8). trace_runs drops glyphs geometrically instead.
    traced = trace_runs(image, dpi)
    items = extract_text(image, traced, dpi=dpi)
    bound, unbound = bind_calls(traced, items, dpi=dpi)
    # The neighbour population is every traced run, not just the bound ones:
    # an unbound edge still closes the called edges around it, so calibration
    # must measure the geometry conversion emits, not a bound-only subset.
    calibration = calibrate_scale(bound, scale=scale, dpi=dpi, runs=traced)
    # Calls are read in feet, so RANSAC's fpp is feet per pixel. A metre
    # drawing's scale is metres per pixel: 1 US survey foot = 1200/3937 m
    # exactly. An explicit ``scale`` is already in the chosen unit, and
    # international feet differ by 2 ppm -- the drawing header is what tells
    # the two foot units apart.
    if units == "metre" and scale is None:
        calibration = Scale(value=calibration.value * 1200.0 / 3937.0, method=calibration.method)
    # Corner closure is emission geometry: calibration measured the traced runs
    # through the same closure, and the drawing gets the closed runs. Bound
    # calls are remapped onto them by identity so the writer's run-to-label map
    # survives.
    runs = join_corners(traced, dpi)
    remapped = {id(run): closed for run, closed in zip(traced, runs, strict=True)}
    bound = [replace(call, run=remapped[id(call.run)]) for call in bound]
    dxf = Path(out) if out is not None else input.with_suffix(".dxf")
    return write_outputs(dxf, runs, bound, unbound, scale=calibration, img_height=image.size[1], units=units, dpi=dpi)

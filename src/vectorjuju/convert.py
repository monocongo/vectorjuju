"""Ingest source plats into one upright raster space and convert to CAD units.

Ingest normalizes a single-page PDF, JPG, or TIFF to one upright RGB raster
with the raster convention (top-left origin, y increasing downward). JPG and
TIFF pixels are used as-is after ``ImageOps.exif_transpose``; a PDF page is
rasterized at ``dpi``. ``px_to_cad``/``cad_to_px`` are the only coordinate
conversion point: raster pixels to CAD units (bottom-left origin, y up).

There is no pixel-unit fallback: transforming before calibration raises.
"""

from __future__ import annotations

import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".tif", ".tiff"})
_RASTER_FORMATS = frozenset({"JPEG", "MPO", "TIFF"})
_PDF_SUFFIX = ".pdf"


class VectorjujuError(Exception):
    """Base class for vectorjuju errors."""


class UnsupportedInputError(VectorjujuError):
    """Input is not a supported single-page PDF, JPG, or TIFF."""


def _require_fpp(fpp: float | None, caller: str) -> float:
    if fpp is None:
        raise ValueError(f"{caller} requires a calibrated fpp; pixel-unit output is not an option")
    if not math.isfinite(fpp) or fpp <= 0:
        raise ValueError(f"{caller} requires a finite fpp > 0, got {fpp!r}")
    return fpp


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
        if pixels > Image.MAX_IMAGE_PIXELS:
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
            if image.size[0] * image.size[1] > Image.MAX_IMAGE_PIXELS:
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
    x, y = pt
    return x * fpp, (img_height - y) * fpp


def cad_to_px(pt: tuple[float, float], *, fpp: float | None = None, img_height: int) -> tuple[float, float]:
    """CAD units (bottom-left, y up) -> raster px (top-left, y down)."""
    fpp = _require_fpp(fpp, "cad_to_px")
    x, y = pt
    return x / fpp, img_height - y / fpp

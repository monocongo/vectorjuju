"""Ingest and pixel-to-CAD transform tests: gate U1 and the input half of A10."""

from __future__ import annotations

import math
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops, ImageStat
from reportlab.pdfgen import canvas

from vectorjuju.convert import UnsupportedInputError, VectorjujuError, cad_to_px, load_raster, px_to_cad
from vectorjuju.synthetic_plat import PAGE_H, PAGE_W, RENDER_DPI, generate_sheet

EXPECTED_SIZE = (round(PAGE_W * RENDER_DPI / 72), round(PAGE_H * RENDER_DPI / 72))  # 1700 x 2200
POINTS = [(0.0, 0.0), (37.5, 812.25), (1699.0, 2199.0), (850.0, 1100.0)]


@pytest.fixture(scope="module")
def sheet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sheet")
    generate_sheet(out)
    return out


@pytest.fixture(scope="module")
def bad_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bad")
    (out / "sheet.txt").write_text("not a plat", encoding="utf-8")
    (out / "corrupt.pdf").write_bytes(b"%PDF-1.4\nnot really a pdf")
    (out / "corrupt.tif").write_bytes(b"not a tiff")
    two_pages = canvas.Canvas(str(out / "two_page.pdf"), pagesize=(PAGE_W, PAGE_H))
    two_pages.showPage()
    two_pages.showPage()
    two_pages.save()
    first, second = Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "black")
    first.save(out / "two_frame.tif", save_all=True, append_images=[second])
    pdfium.PdfDocument.new().save(str(out / "zero_page.pdf"))
    return out


def test_all_media_load_to_the_same_upright_raster(sheet: Path) -> None:
    pdf = load_raster(sheet / "sheet.pdf")
    tif = load_raster(sheet / "sheet.tif")
    jpg = load_raster(sheet / "sheet.jpg")

    # An ignored EXIF orientation would leave the JPG stored 90 degrees rotated.
    assert pdf.size == tif.size == jpg.size == EXPECTED_SIZE
    assert ImageChops.difference(pdf, tif).getbbox() is None

    # JPEG is lossy; the decoded raster must still track the lossless one.
    assert ImageStat.Stat(ImageChops.difference(tif, jpg).convert("L")).mean[0] < 20


def test_unsupported_input_error_is_a_vectorjuju_error() -> None:
    assert issubclass(UnsupportedInputError, VectorjujuError)


@pytest.mark.parametrize(
    "media", ["sheet.txt", "corrupt.pdf", "corrupt.tif", "two_page.pdf", "two_frame.tif", "zero_page.pdf"]
)
def test_unsupported_inputs_raise_and_write_nothing(bad_inputs: Path, media: str) -> None:
    before = set(bad_inputs.iterdir())
    with pytest.raises(UnsupportedInputError):
        load_raster(bad_inputs / media)
    assert set(bad_inputs.iterdir()) == before


@pytest.mark.parametrize("fpp", [72 / (0.85 * 200), 0.129092])
def test_px_to_cad_round_trips(fpp: float) -> None:
    for img_height in (2200, 1100):
        for pt in POINTS:
            back = cad_to_px(px_to_cad(pt, fpp=fpp, img_height=img_height), fpp=fpp, img_height=img_height)
            assert math.dist(back, pt) <= 1e-6


def test_origin_flip_puts_the_top_of_the_raster_at_large_cad_y() -> None:
    fpp, img_height = 0.423529, 2200
    assert px_to_cad((0.0, 0.0), fpp=fpp, img_height=img_height) == (0.0, img_height * fpp)
    assert px_to_cad((0.0, img_height), fpp=fpp, img_height=img_height) == (0.0, 0.0)
    assert px_to_cad((10.0, 30.0), fpp=fpp, img_height=img_height) == (10.0 * fpp, (img_height - 30.0) * fpp)


def test_transform_refuses_pixel_units_without_calibration() -> None:
    with pytest.raises(ValueError):
        px_to_cad((1.0, 2.0), img_height=100)
    with pytest.raises(ValueError):
        cad_to_px((1.0, 2.0), img_height=100)

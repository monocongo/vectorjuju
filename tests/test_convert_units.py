"""Ingest, pixel-to-CAD transform, and text/binding tests: gates U1, U5, and the input half of A10."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops, ImageStat
from reportlab.pdfgen import canvas

from vectorjuju.convert import UnsupportedInputError, VectorjujuError, cad_to_px, load_raster, px_to_cad
from vectorjuju.synthetic_plat import PAGE_H, PAGE_W, PARCEL_FT, RENDER_DPI, bearing_distance, generate_sheet
from vectorjuju.text import TextItem, bind_calls, normalize_ocr, parse_call
from vectorjuju.tracing import Run

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
    (out / "empty.pdf").write_bytes(b"")
    two_pages = canvas.Canvas(str(out / "two_page.pdf"), pagesize=(PAGE_W, PAGE_H))
    two_pages.showPage()
    two_pages.showPage()
    two_pages.save()
    first, second = Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "black")
    first.save(out / "two_frame.tif", save_all=True, append_images=[second])
    pdfium.PdfDocument.new().save(str(out / "zero_page.pdf"))
    Image.new("RGB", (8, 8), "white").save(out / "mislabeled.jpg", format="PNG")
    return out


def test_all_media_load_to_the_same_upright_raster(sheet: Path) -> None:
    pdf = load_raster(sheet / "sheet.pdf")
    tif = load_raster(sheet / "sheet.tif")
    jpg = load_raster(sheet / "sheet.jpg")

    # An ignored EXIF orientation would leave the JPG stored 90 degrees rotated.
    assert pdf.size == tif.size == jpg.size == EXPECTED_SIZE
    assert ImageChops.difference(pdf, tif).getbbox() is None

    # JPEG is lossy; the decoded raster must still track the lossless one.
    # Measured mean-abs-diff on this fixture is ~5.6; 20 leaves headroom for
    # libjpeg/Pillow version drift across CI runners without masking a real bug.
    assert ImageStat.Stat(ImageChops.difference(tif, jpg).convert("L")).mean[0] < 20


def test_unsupported_input_error_is_a_vectorjuju_error() -> None:
    assert issubclass(UnsupportedInputError, VectorjujuError)


def test_uppercase_extension_still_loads(sheet: Path, tmp_path: Path) -> None:
    upper = tmp_path / "SHEET.PDF"
    upper.write_bytes((sheet / "sheet.pdf").read_bytes())
    assert load_raster(upper).size == EXPECTED_SIZE


def test_load_raster_dpi_scales_the_pdf_render(sheet: Path) -> None:
    half = load_raster(sheet / "sheet.pdf", dpi=100)
    assert half.size == (EXPECTED_SIZE[0] // 2, EXPECTED_SIZE[1] // 2)


def test_oversized_pdf_page_is_rejected(tmp_path: Path) -> None:
    huge = tmp_path / "huge.pdf"
    huge_page = canvas.Canvas(str(huge), pagesize=(20000, 20000))
    huge_page.showPage()
    huge_page.save()
    with pytest.raises(UnsupportedInputError):
        load_raster(huge)


def test_zero_dpi_raises_unsupported_input_error_instead_of_leaking(sheet: Path) -> None:
    with pytest.raises(UnsupportedInputError):
        load_raster(sheet / "sheet.pdf", dpi=0)


@pytest.mark.parametrize(
    "media",
    [
        "sheet.txt",
        "corrupt.pdf",
        "corrupt.tif",
        "empty.pdf",
        "missing.pdf",
        "two_page.pdf",
        "two_frame.tif",
        "zero_page.pdf",
        "mislabeled.jpg",
    ],
)
def test_unsupported_inputs_raise_and_write_nothing(bad_inputs: Path, media: str) -> None:
    before = set(bad_inputs.iterdir())
    with pytest.raises(UnsupportedInputError):
        load_raster(bad_inputs / media)
    assert set(bad_inputs.iterdir()) == before


@pytest.mark.parametrize("media", ["sheet.pdf", "sheet.jpg", "sheet.tif"])
def test_near_limit_rasters_are_rejected_by_the_ingest_budget(
    sheet: Path, media: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vectorjuju.convert.MAX_INGEST_PIXELS", 8)
    with pytest.raises(UnsupportedInputError):
        load_raster(sheet / media)


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


@pytest.mark.parametrize("img_height", [0, -1, float("nan"), float("inf")])
def test_transform_refuses_raster_heights_that_cannot_place_the_origin(img_height: float) -> None:
    with pytest.raises(ValueError):
        px_to_cad((1.0, 2.0), fpp=0.423529, img_height=img_height)
    with pytest.raises(ValueError):
        cad_to_px((1.0, 2.0), fpp=0.423529, img_height=img_height)


@pytest.mark.parametrize("fpp", [0.0, -0.42, float("nan"), float("inf")])
def test_transform_refuses_scales_that_cannot_be_a_calibration(fpp: float) -> None:
    with pytest.raises(ValueError):
        px_to_cad((1.0, 2.0), fpp=fpp, img_height=100)
    with pytest.raises(ValueError):
        cad_to_px((1.0, 2.0), fpp=fpp, img_height=100)


# --- U5: parsing and binding, no OCR ----------------------------------------

STRAIGHT_EDGES = [i for i in range(len(PARCEL_FT)) if i not in (2, 4)]  # CURVE_EDGES are 2 and 4


def _run(points: list[tuple[float, float]]) -> Run:
    return Run(points_px=np.array(points, dtype=float))


@pytest.mark.parametrize("edge", STRAIGHT_EDGES)
def test_parse_call_recovers_each_planted_straight_call(edge: int) -> None:
    n = len(PARCEL_FT)
    p0, p1 = PARCEL_FT[edge], PARCEL_FT[(edge + 1) % n]
    bearing, dist_ft = bearing_distance(p0, p1)
    raw = f"{bearing}  {dist_ft:.2f}'"  # synthetic_plat.py's own label format

    call = parse_call(raw)

    assert call is not None
    assert call.kind == "bearing_distance"
    assert math.isclose(call.distance_ft, round(dist_ft, 2), abs_tol=1e-9)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    expected_az = math.degrees(math.atan2(dx, dy)) % 360.0
    # dms() rounds to the nearest arcsecond before formatting; recovering the
    # bearing from that text can be off by up to half an arcsecond (~1.4e-4 deg).
    assert math.isclose(call.bearing_deg, expected_az, abs_tol=1e-3)
    assert call.suspect_tokens == ()


@pytest.mark.parametrize("raw", ["C1", "C2", "c1"])
def test_parse_call_reads_bare_curve_refs(raw: str) -> None:
    call = parse_call(raw)
    assert call is not None
    assert call.kind == "curve_ref"
    assert call.curve_id == raw.upper()


@pytest.mark.parametrize(
    "raw",
    [
        "PARCEL 5",
        "LOT 12",
        "24l",
        "L=150.00'",
        "R=150.00'",
        "RAD. 150",
        "N 45°30' 100.00'",  # historic space-capture bug: no [EW] present at all
        "190.00'",  # curve-table radius cell: no bearing
        "36°41'27\"",  # curve-table delta cell: no quadrant letters
        "SCALE: 1\" = 100'",
        "IPF",
        "IPS",
        "CURVE TABLE",
    ],
)
def test_parse_call_rejects_non_calls(raw: str) -> None:
    assert parse_call(raw) is None


def test_parse_call_accepts_a_whole_degree_bearing_with_no_minutes() -> None:
    call = parse_call("N 45° E 100.00'")
    assert call is not None
    assert math.isclose(call.bearing_deg, 45.0, abs_tol=1e-6)
    assert math.isclose(call.distance_ft, 100.0)


def test_parse_call_rejects_out_of_range_degrees() -> None:
    assert parse_call("N 999°99' E 100.00'") is None


@pytest.mark.parametrize(
    ("raw", "suspect"),
    [
        ('N 87°42\'34" E  200.16"', "foot-mark"),  # foot mark misread as inch mark
        ("N 87°42'34\" E  200.16*", "foot-mark"),  # foot mark dropped to a stray asterisk
        ("N 87 42'34\" E  200.16'", "degree-mark"),  # degree mark dropped
    ],
)
def test_parse_call_flags_a_corrupted_mark_as_suspect(raw: str, suspect: str) -> None:
    call = parse_call(raw)
    assert call is not None
    assert suspect in call.suspect_tokens


def test_normalize_ocr_folds_punctuation_confusables_before_case_folding() -> None:
    assert normalize_ocr("n 45º30′ e") == "N 45°30' E"


def test_bind_calls_prefers_the_nearer_of_two_runs() -> None:
    near = _run([(0.0, 0.0), (200.0, 0.0)])
    far = _run([(0.0, -40.0), (200.0, -40.0)])
    item = TextItem(text="N 90°00'00\" E  200.00'", box_px=(90.0, -3.0, 130.0, 3.0), source="page")

    bound, unbound = bind_calls([near, far], [item])

    assert len(bound) == 1
    assert bound[0].run is near
    assert unbound == []


def test_bind_calls_sends_off_gate_text_to_unbound_never_force_bound() -> None:
    boundary = _run([(0.0, 0.0), (200.0, 0.0)])
    far_item = TextItem(text="N 90°00'00\" E  200.00'", box_px=(90.0, 500.0, 130.0, 506.0), source="page")

    bound, unbound = bind_calls([boundary], [far_item])

    assert bound == []
    assert unbound == [far_item]


def test_bind_calls_never_binds_a_non_call() -> None:
    boundary = _run([(0.0, 0.0), (200.0, 0.0)])
    table_cell = TextItem(text="190.00'", box_px=(90.0, 2.0, 120.0, 8.0), source="page")

    bound, unbound = bind_calls([boundary], [table_cell])

    assert bound == []
    assert unbound == [table_cell]


def test_bind_calls_binds_each_run_at_most_once() -> None:
    boundary = _run([(0.0, 0.0), (200.0, 0.0)])
    first = TextItem(text="N 90°00'00\" E  200.00'", box_px=(90.0, 1.0, 130.0, 3.0), source="crop")
    second = TextItem(text="N 90°00'00\" E  200.00'", box_px=(95.0, 4.0, 135.0, 8.0), source="page")

    bound, unbound = bind_calls([boundary], [first, second])

    assert len(bound) == 1
    assert bound[0].item is first  # closer of the two (distance 2 px vs 6 px)
    assert unbound == [second]


def test_bind_calls_is_deterministic() -> None:
    boundary = _run([(0.0, 0.0), (200.0, 0.0)])
    other = _run([(0.0, 300.0), (200.0, 300.0)])
    items = [
        TextItem(text="N 90°00'00\" E  200.00'", box_px=(90.0, 1.0, 130.0, 3.0), source="crop"),
        TextItem(text="C1", box_px=(90.0, 298.0, 110.0, 302.0), source="crop"),
    ]

    first_bound, first_unbound = bind_calls([boundary, other], items)
    second_bound, second_unbound = bind_calls([boundary, other], items)

    assert [(b.run is boundary, b.call.raw_text) for b in first_bound] == [
        (b.run is boundary, b.call.raw_text) for b in second_bound
    ]
    assert first_unbound == second_unbound

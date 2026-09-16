"""Ingest, pixel-to-CAD transform, and text/binding tests: gates U1, U5, and the input half of A10."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops, ImageStat
from reportlab.pdfgen import canvas

from vectorjuju import text as text_module
from vectorjuju.convert import UnsupportedInputError, VectorjujuError, cad_to_px, load_raster, px_to_cad
from vectorjuju.synthetic_plat import PAGE_H, PAGE_W, PARCEL_FT, RENDER_DPI, bearing_distance, generate_sheet
from vectorjuju.text import TextItem, bind_calls, extract_text, normalize_ocr, parse_call
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


def test_parse_call_accepts_minutes_with_no_seconds() -> None:
    call = parse_call("N 45°30' E 100.00'")
    assert call is not None
    assert math.isclose(call.bearing_deg, 45.5, abs_tol=1e-9)


def test_parse_call_reads_the_south_east_quadrant() -> None:
    # none of the fixture's planted edges land in SE; the quadrant math is
    # otherwise untested for this branch of _azimuth.
    call = parse_call("S 30°00'00\" E 50.00'")
    assert call is not None
    assert math.isclose(call.bearing_deg, 150.0, abs_tol=1e-9)  # 180 - 30


def test_parse_call_rejects_out_of_range_degrees() -> None:
    assert parse_call("N 999°99' E 100.00'") is None


def test_parse_call_accepts_exactly_ninety_degrees() -> None:
    call = parse_call("N 90°00'00\" E 10.00'")
    assert call is not None
    assert math.isclose(call.bearing_deg, 90.0, abs_tol=1e-9)


def test_parse_call_rejects_ninety_degrees_plus_any_minutes() -> None:
    # each field validates in range on its own (0-90, 0-59, 0-59), but a
    # quadrant bearing tops out at exactly 90 degrees combined.
    assert parse_call("N 90°30'00\" E 10.00'") is None


def test_parse_call_rejects_a_bearing_with_no_distance() -> None:
    assert parse_call("N 45° E") is None


def test_parse_call_rejects_a_distance_not_immediately_after_the_bearing() -> None:
    # the historic \b-based bug let a distance regex skip past a glued/
    # garbled read and match a stray digit further into the string --
    # confirmed here as the "1" in a neighbouring curve ref.
    assert parse_call("N 45° E 100.00'C1") is None


@pytest.mark.parametrize(
    "raw",
    [
        "NOTE: N 45° E 100.00' TYPICAL",  # call-shaped substring inside an annotation
        "SEE N 45° E 100.00'",
        "N 45° E 100.00' PER PLAT",
    ],
)
def test_parse_call_rejects_a_call_embedded_in_surrounding_text(raw: str) -> None:
    # A bearing+distance must be the read's entire text -- otherwise this
    # defeats the non-call filtering test_parse_call_rejects_non_calls checks
    # and can attach an authoritative call to the wrong run.
    assert parse_call(raw) is None


def test_parse_call_tolerates_a_stray_character_glued_onto_the_bearing() -> None:
    # Measured OCR noise (a misread tick mark fused onto the quadrant
    # letter, no whitespace gap) must still parse -- unlike a genuine
    # separate annotation word, which the isolation check above rejects.
    call = parse_call("WN 35°09'59\" E 107.65'")
    assert call is not None
    assert math.isclose(call.distance_ft, 107.65)


@pytest.mark.parametrize(
    ("raw", "suspect"),
    [
        ('N 87°42\'34" E  200.16"', "foot-mark"),  # foot mark misread as inch mark
        ("N 87°42'34\" E  200.16*", "foot-mark"),  # foot mark dropped to a stray asterisk
        ("N 87 42'34\" E  200.16'", "degree-mark"),  # degree mark dropped
        ("N 87°42 34\" E  200.16'", "minute-mark"),  # minute mark dropped
        ("N 87°42'34 E  200.16'", "second-mark"),  # second mark dropped
    ],
)
def test_parse_call_flags_a_corrupted_mark_as_suspect(raw: str, suspect: str) -> None:
    call = parse_call(raw)
    assert call is not None
    assert suspect in call.suspect_tokens


@pytest.mark.parametrize("raw", ["C1", " C1 ", "c1\n"])
def test_curve_ref_regex_tolerates_only_whitespace_padding(raw: str) -> None:
    call = parse_call(raw)
    assert call is not None
    assert call.curve_id == "C1"


@pytest.mark.parametrize("raw", ["SEE C1", "C1 REF", "C1A"])
def test_curve_ref_regex_rejects_embedded_non_whitespace_content(raw: str) -> None:
    # fullmatch, deliberately: a ref merged with adjacent OCR text fails safe
    # into unbound_text rather than guessing which part is the real one.
    assert parse_call(raw) is None


def test_normalize_ocr_folds_punctuation_confusables_before_case_folding() -> None:
    assert normalize_ocr("n 45º30′ e") == "N 45°30' E"


def test_convert_path_logs_a_conversion_failure_instead_of_hiding_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # A dependency/init failure must stay observable, not collapse into the
    # same empty result a legitimate no-text page returns.
    class _BrokenConverter:
        def convert(self, _path: str):
            raise RuntimeError("model assets unavailable")

    monkeypatch.setattr(text_module, "_converter", lambda: _BrokenConverter())

    with caplog.at_level(logging.WARNING, logger=text_module.__name__):
        result = text_module._convert_path(tmp_path / "page.png")

    assert result is None
    assert "docling conversion failed" in caplog.text


def test_extract_text_rejects_non_finite_or_nonpositive_dpi() -> None:
    for bad_dpi in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="dpi"):
            extract_text(object(), [], dpi=bad_dpi)


def test_extract_text_rejects_too_many_runs() -> None:
    runs = [Run(points_px=np.zeros((2, 2)))] * (text_module._MAX_RUNS_FOR_TEXT + 1)
    with pytest.raises(ValueError, match="runs"):
        extract_text(object(), runs, dpi=200.0)


def test_extract_text_reuses_supplied_page_items_instead_of_a_second_page_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called(_image: object) -> list[TextItem]:
        raise AssertionError("extract_text must not re-run the page pass when page_items is supplied")

    monkeypatch.setattr(text_module, "_page_items", _fail_if_called)
    supplied = [TextItem(text="C1", box_px=(0.0, 0.0, 1.0, 1.0), source="page")]

    items = extract_text(object(), [], dpi=200.0, page_items=supplied)

    assert items == supplied


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


def test_bind_calls_optimises_globally_instead_of_claiming_greedily() -> None:
    """Two runs, two items, crossing distances: A's only candidate is run1
    (10px); B is nearer to run1 (2px) but also has a real run2 option
    (50px). Greedy-by-nearest-distance claims run1 for B first and strands
    A (its only candidate already taken) -- the exact failure docs/prior-art/
    20-implementation-plan.md's "assignment, not greedy" requirement names.
    Minimising total distance instead finds A->run1, B->run2 (cost 60,
    versus greedy's B->run1 + A unbound), binding both.
    """
    run1 = _run([(0.0, 0.0), (100.0, 0.0)])
    run2 = _run([(150.0, 0.0), (150.0, 100.0)])
    item_a = TextItem(text="N 90°00'00\" E  10.00'", box_px=(48.0, 8.0, 52.0, 12.0), source="page")  # centre (50,10)
    item_b = TextItem(text="N 90°00'00\" E  10.00'", box_px=(98.0, 0.0, 102.0, 4.0), source="page")  # centre (100,2)

    bound, unbound = bind_calls([run1, run2], [item_a, item_b])

    assert unbound == []
    by_item = {b.item: b.run for b in bound}
    assert by_item[item_a] is run1
    assert by_item[item_b] is run2


def test_bind_calls_leaves_a_loser_unbound_not_force_bound_elsewhere() -> None:
    """Three items competing for two runs: the loser has no other in-gate
    run and must come back unbound, never pushed onto a farther one."""
    run1 = _run([(0.0, 0.0), (100.0, 0.0)])
    run2 = _run([(150.0, 0.0), (150.0, 100.0)])
    item_a = TextItem(text="N 90°00'00\" E  5.00'", box_px=(48.0, 3.0, 52.0, 7.0), source="page")  # run1 @ 5, only
    item_b = TextItem(text="N 90°00'00\" E  3.00'", box_px=(48.0, 1.0, 52.0, 5.0), source="page")  # run1 @ 3, only
    item_c = TextItem(text="N 90°00'00\" E  5.00'", box_px=(148.0, 3.0, 152.0, 7.0), source="page")  # run2 @ 5, only

    bound, unbound = bind_calls([run1, run2], [item_a, item_b, item_c])

    by_item = {b.item: b.run for b in bound}
    assert by_item[item_b] is run1  # nearer of the two run1 candidates
    assert by_item[item_c] is run2
    assert unbound == [item_a]  # run1's loser: no other in-gate run existed


def test_bind_calls_penalty_exceeds_a_whole_valid_matching_not_just_one_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a penalty larger only than the biggest single distance let
    the solver buy cheap rows with one penalty cell and beat a complete in-gate
    matching on total cost -- the gate check then dropped a bindable row.

    All spans are infinite (invalid) except the 59px diagonal (the only
    complete in-gate matching, 236px total) and a 0.1px chain that leaves one
    row needing a penalty cell. The penalty must exceed 236px, so the 60px
    gate is the floor: 60 * 4 + 1. The old max-distance + 2*gate + 1 = 180px
    penalty lost (180.3 < 236) and returned one item unbound.
    """
    gate = 60.0  # _BIND_RADIUS_PX at the 200dpi reference
    size = 4
    distances = np.full((size, size), np.inf)
    for i in range(size):
        distances[i, i] = gate - 1.0
    for i in range(1, size):
        distances[i, i - 1] = 0.1
    runs = [_run([(0.0, 0.0), (1.0, 0.0)]) for _ in range(size)]
    items = [
        TextItem(text="N 90°00'00\" E  200.00'", box_px=(float(i), 0.0, float(i) + 1.0, 1.0), source="page")
        for i in range(size)
    ]
    columns = {id(run): index for index, run in enumerate(runs)}
    monkeypatch.setattr(
        text_module,
        "_point_run_distance",
        lambda point, run: float(distances[int(point[0])][columns[id(run)]]),
    )

    bound, unbound = bind_calls(runs, items, dpi=200.0)

    assert unbound == []
    assert [b.item for b in bound] == items
    assert [b.run for b in bound] == runs


def test_bind_calls_collapses_duplicate_reads_of_one_call() -> None:
    """Regression: the page pass and the band crop both read one label, and
    binding both let the duplicate claim run2 while run2's real call (whose
    read is only in gate of run2) was stranded in unbound. A collapsed
    duplicate must also not resurface in unbound_text as a second call."""
    call = "N 90°00'00\" E  100.00'"
    run1 = _run([(0.0, 0.0), (100.0, 0.0)])
    run2 = _run([(100.0, 0.0), (100.0, 100.0)])
    page = TextItem(text=call, box_px=(68.0, 0.0, 72.0, 4.0), source="page")
    crop = TextItem(text=call, box_px=(68.0, 0.0, 72.0, 4.0), source="crop")
    other = TextItem(text="S 45°00'00\" W  50.00'", box_px=(133.0, 48.0, 137.0, 52.0), source="page")

    bound, unbound = bind_calls([run1, run2], [page, crop, other], dpi=200.0)

    assert unbound == []
    assert len(bound) == 2
    by_call = {bc.call.raw_text: bc.run for bc in bound}
    assert by_call[call] is run1  # page geometry kept; crop duplicate dropped
    assert by_call[other.text] is run2


def test_bind_calls_collapses_a_chain_of_overlapping_duplicate_reads() -> None:
    """Regression: three reads of one label where only consecutive boxes
    overlap (page -> crop -> crop). The third read overlaps the merged crop but
    not the page representative; checking overlap against representatives alone
    left it distinct, letting it claim run2 and strand run2's real call."""
    call = "N 90°00'00\" E  100.00'"
    run1 = _run([(0.0, 0.0), (100.0, 0.0)])
    run2 = _run([(100.0, 0.0), (100.0, 100.0)])
    page = TextItem(text=call, box_px=(68.0, 0.0, 72.0, 4.0), source="page")
    crop1 = TextItem(text=call, box_px=(70.0, 0.0, 74.0, 4.0), source="crop")
    crop2 = TextItem(text=call, box_px=(72.5, 0.0, 76.5, 4.0), source="crop")
    other = TextItem(text="S 45°00'00\" W  50.00'", box_px=(133.0, 48.0, 137.0, 52.0), source="page")

    bound, unbound = bind_calls([run1, run2], [page, crop1, crop2, other], dpi=200.0)

    assert unbound == []
    assert len(bound) == 2
    by_call = {bc.call.raw_text: bc.run for bc in bound}
    assert by_call[call] is run1  # page geometry kept; both crop duplicates dropped
    assert by_call[other.text] is run2


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

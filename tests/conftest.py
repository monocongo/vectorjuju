"""Session fixtures for the acceptance suite (issue #22).

One generated synthetic sheet per session, one ``convert()`` per medium into
its own output directory, and one overlay artifact per medium for the CI job
to upload. The heavy OCR runs happen once here; the gates only read what they
produced.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from acceptance_helpers import write_overlay
from PIL import Image
from reportlab.pdfgen import canvas

from vectorjuju.pipeline import convert, load_raster
from vectorjuju.synthetic_plat import PAGE_H, PAGE_W, RENDER_DPI, generate_sheet

MEDIA = ("pdf", "tif", "jpg")


@dataclass(frozen=True)
class SyntheticSheet:
    """The generated sheet, its media, and its planted ground truth."""

    directory: Path
    truth: dict

    def media(self, medium: str) -> Path:
        return self.directory / f"sheet.{medium}"


@dataclass(frozen=True)
class Conversion:
    """One full ``convert()`` run over one medium."""

    medium: str
    out: Path
    sidecar: dict
    img_height: int
    rerun_out: Path
    rerun_sidecar: dict

    @property
    def entities(self) -> list[dict]:
        return self.sidecar["entities"]

    @property
    def labeled(self) -> list[dict]:
        return [entry for entry in self.entities if entry["label"] is not None]


@pytest.fixture(scope="session")
def artifacts() -> Path:
    """Where the CI job collects overlays; ``VECTORJUJU_ARTIFACTS`` overrides."""
    directory = Path(os.environ.get("VECTORJUJU_ARTIFACTS", "artifacts"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def synthetic_sheet(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSheet:
    directory = tmp_path_factory.mktemp("sheet")
    return SyntheticSheet(directory=directory, truth=generate_sheet(directory))


@pytest.fixture(scope="session")
def conversions(
    synthetic_sheet: SyntheticSheet, tmp_path_factory: pytest.TempPathFactory, artifacts: Path
) -> dict[str, Conversion]:
    """Every medium converted once (plus a repeat run for determinism)."""
    made: dict[str, Conversion] = {}
    for medium in MEDIA:
        input_path = synthetic_sheet.media(medium)
        out = convert(input_path, out=tmp_path_factory.mktemp(f"converted-{medium}") / "sheet.dxf")
        rerun = convert(input_path, out=tmp_path_factory.mktemp(f"rerun-{medium}") / "sheet.dxf")
        sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        page = load_raster(input_path, dpi=RENDER_DPI)
        write_overlay(page, out, sidecar, artifacts / f"overlay-{medium}.png")
        made[medium] = Conversion(
            medium=medium,
            out=out,
            sidecar=sidecar,
            img_height=page.size[1],
            rerun_out=rerun,
            rerun_sidecar=json.loads(rerun.with_suffix(".json").read_text(encoding="utf-8")),
        )
    return made


@pytest.fixture(scope="session", params=MEDIA, ids=MEDIA)
def conversion(request: pytest.FixtureRequest, conversions: dict[str, Conversion]) -> Conversion:
    return conversions[request.param]


@pytest.fixture(scope="session")
def bad_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every input class that must fail, plus the uncalibratable page."""
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

    # Text but no distance calls anywhere: calibration has nothing to earn.
    text_only = canvas.Canvas(str(out / "text_only.pdf"), pagesize=(PAGE_W, PAGE_H))
    text_only.setFont("Helvetica", 10)
    text_only.drawString(72, 700, "PARCEL 5")
    text_only.drawString(72, 680, "LOT 12")
    text_only.drawString(72, 660, "CURVE TABLE")
    text_only.showPage()
    text_only.save()
    return out

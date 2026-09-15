import json
import math

from PIL import Image

from vectorjuju.synthetic_plat import (
    angle_bucket,
    arc_points,
    bearing_distance,
    dms,
    ft_to_pt,
    generate_sheet,
    label_rotation,
)


def test_dms_formats_degrees_minutes_seconds():
    assert dms(45.5075) == "45°30'27\""


def test_dms_rounds_seconds_and_minutes_overflow_into_degrees():
    # 44.999916 deg: seconds round to 60 -> carries into minutes -> carries into degrees.
    assert dms(44.999916) == "45°00'00\""


def test_bearing_distance_due_north():
    bearing, distance = bearing_distance((0.0, 0.0), (0.0, 100.0))
    assert bearing == "N 0°00'00\" E"
    assert math.isclose(distance, 100.0)


def test_bearing_distance_due_east():
    bearing, distance = bearing_distance((0.0, 0.0), (100.0, 0.0))
    assert bearing == "N 90°00'00\" E"
    assert math.isclose(distance, 100.0)


def test_label_rotation_stays_in_range():
    assert label_rotation((0.0, 0.0), (0.0, 100.0)) == 90.0
    assert label_rotation((0.0, 0.0), (100.0, 0.0)) == 0.0


def test_angle_bucket_thresholds():
    assert angle_bucket(0.0) == "flat"
    assert angle_bucket(30.0) == "shallow"
    assert angle_bucket(60.0) == "steep"
    assert angle_bucket(89.0) == "near-vertical"


def test_arc_points_starts_and_ends_at_chord_endpoints():
    p0, p1 = (0.0, 0.0), (100.0, 0.0)
    pts, radius, delta, arc_len, chord = arc_points(p0, p1, radius_ft=80.0)
    assert math.isclose(pts[0][0], p0[0], abs_tol=1e-6)
    assert math.isclose(pts[0][1], p0[1], abs_tol=1e-6)
    assert math.isclose(pts[-1][0], p1[0], abs_tol=1e-6)
    assert math.isclose(pts[-1][1], p1[1], abs_tol=1e-6)
    assert radius >= 80.0
    assert delta > 0.0
    assert arc_len > 0.0
    assert math.isclose(chord, 100.0)


def test_ft_to_pt_applies_scale_and_origin():
    assert ft_to_pt((0.0, 0.0)) == (60.0, 300.0)


def test_generate_sheet_writes_expected_files(tmp_path):
    ground_truth = generate_sheet(tmp_path)

    assert (tmp_path / "sheet.pdf").exists()
    assert (tmp_path / "sheet.tif").exists()
    assert (tmp_path / "sheet.jpg").exists()

    on_disk = json.loads((tmp_path / "ground_truth.json").read_text())
    # generate_sheet's return value holds tuples (e.g. start_pt); JSON only has
    # arrays, so compare both sides through a JSON round trip.
    assert on_disk == json.loads(json.dumps(ground_truth))


def test_generate_sheet_ground_truth_has_straight_and_curve_calls(tmp_path):
    gt = generate_sheet(tmp_path)

    assert any(seg["kind"] == "straight" for seg in gt["segments"])
    assert any(seg["kind"] == "curve" for seg in gt["segments"])
    assert len(gt["curves"]) >= 1
    assert len(gt["curve_table_cells"]) == len(gt["curves"]) + 1  # header row


def test_generate_sheet_ground_truth_has_rotated_labels(tmp_path):
    gt = generate_sheet(tmp_path)

    rotations = {lab["rotation_deg"] for lab in gt["labels"]}
    assert any(abs(r) > 1.0 for r in rotations), "expected at least one non-flat label rotation"


def test_generate_sheet_jpg_is_exif_rotated(tmp_path):
    generate_sheet(tmp_path)

    with Image.open(tmp_path / "sheet.jpg") as img:
        exif = img.getexif()
        tiff_orientation_tag = 274
        assert exif.get(tiff_orientation_tag) == 6

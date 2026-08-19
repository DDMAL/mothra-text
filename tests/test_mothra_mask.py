"""Tests for steps.mothra_mask.MothraImageMask."""

import json

from PIL import Image

from steps.mothra_mask import MothraImageMask


def _write_mothra_json(tmp_path, annotations):
    path = tmp_path / "mothra.json"
    path.write_text(json.dumps({"annotations": annotations}))
    return str(path)


def _pixel(img, x, y):
    return img.getpixel((x, y))


class TestInitFiltersToClassId1:
    def test_keeps_only_classid_1_bboxes(self, tmp_path):
        path = _write_mothra_json(tmp_path, [
            {"classId": 0, "bbox": [1, 1, 2, 2]},
            {"classId": 1, "bbox": [3, 3, 4, 4]},
            {"classId": 2, "bbox": [5, 5, 6, 6]},
            {"classId": 1, "bbox": [7, 7, 8, 8]},
        ])
        masker = MothraImageMask(path)
        assert masker._bboxes == [[3, 3, 4, 4], [7, 7, 8, 8]]

    def test_no_classid_1_annotations_gives_empty_bboxes(self, tmp_path):
        path = _write_mothra_json(tmp_path, [
            {"classId": 0, "bbox": [1, 1, 2, 2]},
            {"classId": 2, "bbox": [5, 5, 6, 6]},
        ])
        masker = MothraImageMask(path)
        assert masker._bboxes == []


class TestApplyMasking:
    def test_blacks_out_regions_outside_bbox(self, tmp_path):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        path = _write_mothra_json(tmp_path, [
            {"classId": 1, "bbox": [8, 8, 4, 4]},
        ])
        masker = MothraImageMask(path, padding_px=0)
        result = masker.apply(img)
        # Inside the bbox: original (white) pixel preserved.
        assert _pixel(result, 9, 9) == (255, 255, 255)
        # Far outside the bbox: blacked out.
        assert _pixel(result, 0, 0) == (0, 0, 0)
        assert _pixel(result, 19, 19) == (0, 0, 0)

    def test_padding_expands_visible_region(self, tmp_path):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        path = _write_mothra_json(tmp_path, [
            {"classId": 1, "bbox": [8, 8, 4, 4]},
        ])
        # Pixel (6,9) is just outside the raw bbox (x in [8,12)) but within
        # a 2px padding -> should become visible with padding, unlike
        # the zero-padding case above.
        masker = MothraImageMask(path, padding_px=2)
        result = masker.apply(img)
        assert _pixel(result, 6, 9) == (255, 255, 255)

    def test_padding_clamped_to_image_bounds(self, tmp_path):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        path = _write_mothra_json(tmp_path, [
            {"classId": 1, "bbox": [0, 0, 2, 2]},
        ])
        # Padding far larger than the image; must clamp rather than error
        # or produce an invalid (negative/out-of-range) draw rectangle.
        masker = MothraImageMask(path, padding_px=1000)
        result = masker.apply(img)
        assert _pixel(result, 0, 0) == (255, 255, 255)
        assert _pixel(result, 9, 9) == (255, 255, 255)

    def test_no_bboxes_produces_fully_black_image(self, tmp_path):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        path = _write_mothra_json(tmp_path, [])
        masker = MothraImageMask(path)
        result = masker.apply(img)
        assert _pixel(result, 5, 5) == (0, 0, 0)

    def test_confidence_field_is_not_read_or_filtered(self, tmp_path):
        # Pitfall regression guard (see DEEP_DIVE.md #13): apply() must
        # keep classId-1 boxes regardless of any confidence/score field —
        # it has no concept of a confidence threshold at all. If this
        # test ever starts failing because MothraImageMask began filtering
        # on confidence, update DEEP_DIVE.md's Pitfalls entry alongside it.
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        path = _write_mothra_json(tmp_path, [
            {"classId": 1, "bbox": [1, 1, 2, 2], "confidence": 0.01},
            {"classId": 1, "bbox": [10, 10, 2, 2]},
        ])
        masker = MothraImageMask(path, padding_px=0)
        assert len(masker._bboxes) == 2
        result = masker.apply(img)
        assert _pixel(result, 1, 1) == (255, 255, 255)
        assert _pixel(result, 10, 10) == (255, 255, 255)

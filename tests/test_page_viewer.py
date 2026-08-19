"""Tests for page_viewer.py's pure helper functions (not the Tkinter app)."""

import json
import xml.etree.ElementTree as ET

import pytest

from page_viewer import (
    _classify_arg,
    _coords,
    _parse_points,
    _point_in_polygon,
    _point_near_polyline,
    _resolve_image,
    _text_equiv,
    load_annotation_file,
    parse_kraken_json,
    parse_page_xml,
)

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

PAGE_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{NS}">
  <Page imageFilename="folio.jpg" imageWidth="100" imageHeight="200">
    <TextRegion id="r1" type="paragraph">
      <Coords points="0,0 10,0 10,10 0,10"/>
      <TextLine id="r1_l1">
        <Coords points="1,1 9,1 9,5 1,5"/>
        <Baseline points="1,3 9,3"/>
        <Word id="r1_l1_w1">
          <Coords points="1,1 4,1 4,5 1,5"/>
          <TextEquiv><Unicode>foo</Unicode></TextEquiv>
          <Glyph id="r1_l1_w1_g1">
            <Coords points="1,1 2,1 2,5 1,5"/>
            <TextEquiv><Unicode>f</Unicode></TextEquiv>
          </Glyph>
        </Word>
        <TextEquiv><Unicode>foo</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""

KRAKEN_JSON = {
    "type": "baselines",
    "imagename": "folio.jpg",
    "regions": {
        "paragraph": [
            {"id": "r1", "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        ],
    },
    "lines": [
        {
            "id": "l1",
            "boundary": [[1, 1], [9, 1], [9, 5], [1, 5]],
            "baseline": [[1, 3], [9, 3]],
            "text": "foo",
            "tags": "default",
            "regions": ["r1"],
        },
    ],
}


class TestPointInPolygon:
    def test_point_inside_square(self):
        square = [0, 0, 10, 0, 10, 10, 0, 10]
        assert _point_in_polygon(5, 5, square) is True

    def test_point_outside_square(self):
        square = [0, 0, 10, 0, 10, 10, 0, 10]
        assert _point_in_polygon(50, 50, square) is False


class TestPointNearPolyline:
    def test_point_on_segment_is_near(self):
        line = [0, 0, 10, 0]
        assert _point_near_polyline(5, 0, line) is True

    def test_point_far_from_segment_is_not_near(self):
        line = [0, 0, 10, 0]
        assert _point_near_polyline(5, 100, line) is False

    def test_point_within_threshold_of_segment(self):
        line = [0, 0, 10, 0]
        assert _point_near_polyline(5, 5, line, threshold=6.0) is True
        assert _point_near_polyline(5, 7, line, threshold=6.0) is False

    def test_degenerate_zero_length_segment(self):
        # Both endpoints identical -> seg_sq == 0 -> falls back to plain
        # point-to-point distance instead of dividing by zero.
        line = [3, 3, 3, 3]
        assert _point_near_polyline(3, 3, line, threshold=1.0) is True
        assert _point_near_polyline(10, 10, line, threshold=1.0) is False


class TestParsePoints:
    def test_parses_space_separated_pairs(self):
        assert _parse_points("10,20 30,40") == [(10, 20), (30, 40)]

    def test_truncates_float_like_coordinates(self):
        assert _parse_points("10.7,20.2") == [(10, 20)]

    def test_empty_string_gives_empty_list(self):
        assert _parse_points("") == []


class TestTextEquivAndCoords:
    def test_text_equiv_present(self):
        el = ET.fromstring(
            f'<Word xmlns="{NS}" id="w1">'
            f'<TextEquiv><Unicode> hello </Unicode></TextEquiv></Word>'
        )
        assert _text_equiv(el, NS) == "hello"

    def test_text_equiv_absent_returns_empty_string(self):
        el = ET.fromstring(f'<Word xmlns="{NS}" id="w1"/>')
        assert _text_equiv(el, NS) == ""

    def test_coords_present(self):
        el = ET.fromstring(
            f'<Word xmlns="{NS}" id="w1"><Coords points="1,1 2,2"/></Word>'
        )
        assert _coords(el, NS) == [(1, 1), (2, 2)]

    def test_coords_absent_returns_empty_list(self):
        el = ET.fromstring(f'<Word xmlns="{NS}" id="w1"/>')
        assert _coords(el, NS) == []


class TestParsePageXml:
    def test_full_structure(self, tmp_path):
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(PAGE_XML)
        result = parse_page_xml(str(xml_path))

        assert result["image_filename"] == "folio.jpg"
        assert result["image_width"] == 100
        assert result["image_height"] == 200

        assert len(result["regions"]) == 1
        region = result["regions"][0]
        assert region["id"] == "r1"
        assert region["type"] == "paragraph"
        assert region["parent_id"] is None
        assert region["coords"] == [(0, 0), (10, 0), (10, 10), (0, 10)]

        assert len(result["lines"]) == 1
        line = result["lines"][0]
        assert line["parent_id"] == "r1"
        assert line["text"] == "foo"

        assert len(result["baselines"]) == 1
        assert result["baselines"][0]["parent_id"] == "r1_l1"
        assert result["baselines"][0]["coords"] == [(1, 3), (9, 3)]

        assert len(result["words"]) == 1
        word = result["words"][0]
        assert word["parent_id"] == "r1_l1"
        assert word["text"] == "foo"

        assert len(result["glyphs"]) == 1
        glyph = result["glyphs"][0]
        assert glyph["parent_id"] == "r1_l1_w1"
        assert glyph["text"] == "f"

    def test_missing_page_element_raises(self, tmp_path):
        xml_path = tmp_path / "bad.xml"
        xml_path.write_text(f'<PcGts xmlns="{NS}"></PcGts>')
        with pytest.raises(ValueError, match="No <Page>"):
            parse_page_xml(str(xml_path))


class TestParseKrakenJson:
    def test_full_structure(self, tmp_path):
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(KRAKEN_JSON))
        result = parse_kraken_json(str(path))

        assert result["image_filename"] == "folio.jpg"
        assert result["image_width"] == 0
        assert result["image_height"] == 0

        assert len(result["regions"]) == 1
        assert result["regions"][0]["type"] == "paragraph"
        assert result["regions"][0]["coords"] == [
            (0, 0), (10, 0), (10, 10), (0, 10)
        ]

        assert len(result["lines"]) == 1
        line = result["lines"][0]
        assert line["parent_id"] == "r1"
        assert line["text"] == "foo"
        assert line["attrs"] == {"tags": "default"}

        assert len(result["baselines"]) == 1
        assert result["baselines"][0]["coords"] == [(1, 3), (9, 3)]

    def test_unrecognised_type_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"type": "something-else"}))
        with pytest.raises(ValueError, match="Unrecognised Kraken JSON type"):
            parse_kraken_json(str(path))

    def test_missing_regions_and_lines_gives_empty_lists(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps({"type": "baselines"}))
        result = parse_kraken_json(str(path))
        assert result["regions"] == []
        assert result["lines"] == []
        assert result["baselines"] == []


class TestLoadAnnotationFile:
    def test_detects_kraken_json_by_content(self, tmp_path):
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(KRAKEN_JSON))
        data, label = load_annotation_file(str(path))
        assert label == "Kraken JSON"
        assert data["image_filename"] == "folio.jpg"

    def test_detects_page_xml_by_content(self, tmp_path):
        path = tmp_path / "ann.xml"
        path.write_text(PAGE_XML)
        data, label = load_annotation_file(str(path))
        assert label == "PAGE XML"
        assert data["image_filename"] == "folio.jpg"


class TestResolveImage:
    def test_finds_image_next_to_xml(self, tmp_path):
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(PAGE_XML)
        (tmp_path / "folio.jpg").write_bytes(b"")
        result = _resolve_image("folio.jpg", str(xml_path))
        assert result == tmp_path / "folio.jpg"

    def test_falls_back_to_basename_match(self, tmp_path):
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(PAGE_XML)
        (tmp_path / "folio2.jpg").write_bytes(b"")
        result = _resolve_image("some/nested/folio2.jpg", str(xml_path))
        assert result == tmp_path / "folio2.jpg"

    def test_absolute_path_checked_first(self, tmp_path):
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(PAGE_XML)
        abs_dir = tmp_path / "elsewhere"
        abs_dir.mkdir()
        abs_image = abs_dir / "folio.jpg"
        abs_image.write_bytes(b"")
        result = _resolve_image(str(abs_image), str(xml_path))
        assert result == abs_image

    def test_returns_none_when_not_found(self, tmp_path):
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(PAGE_XML)
        assert _resolve_image("nowhere.jpg", str(xml_path)) is None


class TestClassifyArg:
    def test_image_extension(self):
        assert _classify_arg("folio.jpg") == "image"
        assert _classify_arg("folio.PNG") == "image"

    def test_annotation_extension(self):
        assert _classify_arg("ann.xml") == "xml"
        assert _classify_arg("ann.json") == "xml"

    def test_unknown_extension(self):
        assert _classify_arg("readme.md") == "unknown"

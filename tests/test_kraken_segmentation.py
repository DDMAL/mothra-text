"""Tests for steps.kraken_segmentation."""

import logging
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from htrflow.volume.volume import Collection

from steps.kraken_segmentation import KrakenSegmentation


def _make_test_image(path: str, h: int = 100, w: int = 200) -> None:
    """Write a small BGR image to path."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.imwrite(path, img)


def _fake_segmentation(boundaries):
    """Build a minimal Kraken Segmentation mock."""
    seg = MagicMock()
    lines = []
    for boundary in boundaries:
        line = MagicMock()
        line.boundary = boundary
        lines.append(line)
    seg.lines = lines
    return seg


class TestKrakenSegmentation:
    def test_lines_become_nodes(self, tmp_path):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        boundary = [(0, 0), (100, 0), (100, 20), (0, 20)]
        fake_seg = _fake_segmentation([boundary, boundary])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            step = KrakenSegmentation(device="cpu")
            collection = step.run(collection)

        nodes = list(collection.active_leaves())
        assert len(nodes) == 2

    def test_none_boundary_lines_are_skipped(self, tmp_path):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        boundary = [(0, 0), (100, 0), (100, 20), (0, 20)]
        fake_seg = _fake_segmentation([boundary, None])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            step = KrakenSegmentation(device="cpu")
            collection = step.run(collection)

        nodes = list(collection.active_leaves())
        assert len(nodes) == 1

    def test_all_none_boundaries_produces_no_segment_nodes(self, tmp_path):
        # When every line has None boundary, no SegmentNode children are
        # created. HTRflow leaves the PageNode as the only active leaf.
        from htrflow.volume.volume import PageNode, SegmentNode
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        fake_seg = _fake_segmentation([None, None, None])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            step = KrakenSegmentation(device="cpu")
            collection = step.run(collection)

        nodes = list(collection.active_leaves())
        assert len(nodes) == 1
        assert isinstance(nodes[0], PageNode)
        assert not any(isinstance(n, SegmentNode) for n in nodes)

    def test_none_boundary_warning_is_logged(self, tmp_path, caplog):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        boundary = [(0, 0), (100, 0), (100, 20), (0, 20)]
        fake_seg = _fake_segmentation([boundary, None])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            with caplog.at_level(
                logging.WARNING, logger="steps.kraken_segmentation"
            ):
                collection = Collection([img_path])
                KrakenSegmentation(device="cpu").run(collection)

        assert caplog.records

    def test_node_has_bbox(self, tmp_path):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path, h=100, w=200)

        boundary = [(10, 5), (150, 5), (150, 30), (10, 30)]
        fake_seg = _fake_segmentation([boundary])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            KrakenSegmentation(device="cpu").run(collection)

        node = list(collection.active_leaves())[0]
        assert node.bbox is not None
        assert node.bbox.area > 0

    def test_returns_collection(self, tmp_path):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        fake_seg = _fake_segmentation([[(0, 0), (50, 0), (50, 10), (0, 10)]])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            result = KrakenSegmentation(device="cpu").run(collection)

        assert isinstance(result, Collection)

    def test_device_passed_to_blla(self, tmp_path):
        img_path = str(tmp_path / "folio.jpg")
        _make_test_image(img_path)

        fake_seg = _fake_segmentation([])

        with patch("steps.kraken_segmentation.blla") as mock_blla:
            mock_blla.segment.return_value = fake_seg
            collection = Collection([img_path])
            KrakenSegmentation(device="mps").run(collection)

        _, kwargs = mock_blla.segment.call_args
        assert kwargs.get("device") == "mps"

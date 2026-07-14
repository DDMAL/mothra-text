"""Tests for steps.kraken_recognition."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from steps.kraken_recognition import KrakenRecognition


def _make_node(label: str = "region0_line0") -> MagicMock:
    node = MagicMock()
    node.label = label
    node.image = np.ones((30, 200, 3), dtype=np.uint8) * 128
    return node


def _make_collection(nodes: list) -> MagicMock:
    col = MagicMock()
    col.active_leaves.return_value = iter(nodes)
    return col


class TestKrakenRecognitionStubMode:
    def test_raises_without_allow_stub(self):
        collection = _make_collection([_make_node()])
        with pytest.raises(ValueError, match="stub mode was not explicitly requested"):
            KrakenRecognition(model=None, allow_stub=False).run(collection)

    def test_stub_sets_empty_text_on_all_nodes(self):
        nodes = [_make_node("r0_l0"), _make_node("r0_l1"), _make_node("r0_l2")]
        collection = _make_collection(nodes)

        KrakenRecognition(model=None, allow_stub=True).run(collection)

        collection.update.assert_called_once()
        (results,) = collection.update.call_args.args
        assert len(results) == 3
        for r in results:
            assert r.texts == [""]

    def test_stub_logs_warning(self, caplog):
        collection = _make_collection([_make_node()])
        with caplog.at_level(logging.WARNING, logger="steps.kraken_recognition"):
            KrakenRecognition(model=None, allow_stub=True).run(collection)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_stub_returns_same_collection(self):
        collection = _make_collection([_make_node()])
        returned = KrakenRecognition(model=None, allow_stub=True).run(collection)
        assert returned is collection

    def test_stub_empty_collection_no_update(self):
        collection = _make_collection([])
        KrakenRecognition(model=None, allow_stub=True).run(collection)
        collection.update.assert_not_called()


class TestKrakenRecognitionModelPath:
    def test_model_path_loads_model_once(self):
        nodes = [_make_node("r0_l0"), _make_node("r0_l1")]
        collection = _make_collection(nodes)

        fake_record = MagicMock()
        fake_record.prediction = "alleluia"
        fake_record.confidences = [0.9, 0.8, 0.9]

        with patch("kraken.lib.models.load_any", return_value=MagicMock()) as mock_load, \
             patch("kraken.rpred.rpred", return_value=iter([fake_record])):
            KrakenRecognition(model="some/model", device="cpu").run(collection)

        mock_load.assert_called_once_with("some/model", device="cpu")

    def test_model_path_text_from_record_prediction(self):
        nodes = [_make_node("r0_l0")]
        collection = _make_collection(nodes)

        fake_record = MagicMock()
        fake_record.prediction = "dominus"
        fake_record.confidences = [0.95]

        with patch("kraken.lib.models.load_any", return_value=MagicMock()), \
             patch("kraken.rpred.rpred", return_value=iter([fake_record])):
            KrakenRecognition(model="some/model").run(collection)

        (results,) = collection.update.call_args.args
        assert results[0].texts == ["dominus"]

    def test_model_path_rpred_called_once_per_node(self):
        nodes = [_make_node("r0_l0"), _make_node("r0_l1")]
        collection = _make_collection(nodes)

        fake_record = MagicMock()
        fake_record.prediction = "kyrie"
        fake_record.confidences = [0.9]

        with patch("kraken.lib.models.load_any", return_value=MagicMock()), \
             patch("kraken.rpred.rpred", return_value=iter([fake_record])) as mock_rpred:
            KrakenRecognition(model="some/model").run(collection)

        assert mock_rpred.call_count == 2

    def test_model_path_empty_collection_no_rpred(self):
        collection = _make_collection([])

        with patch("kraken.lib.models.load_any", return_value=MagicMock()), \
             patch("kraken.rpred.rpred") as mock_rpred:
            KrakenRecognition(model="some/model").run(collection)

        mock_rpred.assert_not_called()
        collection.update.assert_not_called()

"""Tests for run_chain.py: folio contiguity checking and the chaining loop."""

import sys
import types
from unittest.mock import MagicMock

import pytest

import run_chain
from run_chain import _are_contiguous, _parse_folio_id, _run_one


class TestParseFolioId:
    def test_zero_padded_recto(self):
        assert _parse_folio_id("001r") == (1, "r")

    def test_non_padded_verso(self):
        assert _parse_folio_id("1v") == (1, "v")

    def test_number_only(self):
        assert _parse_folio_id("1") == (1, None)

    def test_case_insensitive(self):
        assert _parse_folio_id("001R") == (1, "r")

    def test_unrecognized_format_returns_none_none(self):
        assert _parse_folio_id("3bis") == (None, None)

    def test_empty_string_returns_none_none(self):
        assert _parse_folio_id("") == (None, None)


class TestAreContiguous:
    def test_recto_to_verso_same_number_is_contiguous(self):
        assert _are_contiguous("001r", "001v") is True

    def test_verso_to_next_recto_is_contiguous(self):
        assert _are_contiguous("001v", "002r") is True

    def test_no_suffix_sequential_numbers_is_contiguous(self):
        assert _are_contiguous("1", "2") is True

    def test_recto_to_recto_same_number_is_not_contiguous(self):
        assert _are_contiguous("001r", "001r") is False

    def test_verso_to_verso_same_number_is_not_contiguous(self):
        assert _are_contiguous("001v", "001v") is False

    def test_non_adjacent_numbers_not_contiguous(self):
        assert _are_contiguous("001r", "005r") is False

    def test_unrecognized_folio_format_not_contiguous(self):
        # _parse_folio_id returns (None, None) for either side -> always False,
        # matching the documented fallback of "always run without chaining"
        # for an unusual naming convention.
        assert _are_contiguous("3bis", "4bis") is False
        assert _are_contiguous("001r", "3bis") is False


class _FakeFolioState:
    def __init__(self, remaining_words=None, fully_consumed=True):
        self.remaining_words = remaining_words or []
        self.fully_consumed = fully_consumed


class TestRunOne:
    def _common_kwargs(self, run_mock, read_folio_state_mock, **overrides):
        args = MagicMock()
        args.source_id = 123
        args.csv = None
        args.segmentation_model = None
        args.device = "cpu"
        args.column_bimodal_threshold = 0.5
        args.debug_ocr = False
        args.column_count = None
        kwargs = dict(
            idx=0,
            total=1,
            image_path="folio.jpg",
            folio="001r",
            prev_state=None,
            infer_continuation=True,
            export_json_path=None,
            mei_json_path=None,
            folio_label="001r",
            folio_states_dir=None,
            recognition_model=None,
            mothra_json_path=None,
            padding=15,
            args=args,
            run=run_mock,
            export_json=MagicMock(),
            read_folio_state=read_folio_state_mock,
            build_pipeline_payload=MagicMock(),
            write_mei_json=MagicMock(),
        )
        kwargs.update(overrides)
        return kwargs

    def test_passes_infer_continuation_through_to_run(self):
        run_mock = MagicMock(return_value=(MagicMock(), {}))
        read_mock = MagicMock(return_value=_FakeFolioState())
        kwargs = self._common_kwargs(
            run_mock, read_mock, infer_continuation=False
        )
        _run_one(**kwargs)
        assert run_mock.call_args.kwargs["infer_continuation"] is False

    def test_cleans_up_temp_folio_state_file_on_success(self):
        run_mock = MagicMock(return_value=(MagicMock(), {}))
        read_mock = MagicMock(return_value=_FakeFolioState())
        kwargs = self._common_kwargs(run_mock, read_mock)
        _run_one(**kwargs)
        tmp_path = read_mock.call_args.args[0]
        assert not __import__("pathlib").Path(tmp_path).exists()

    def test_cleans_up_temp_file_and_reraises_on_run_failure(self):
        run_mock = MagicMock(side_effect=RuntimeError("boom"))
        read_mock = MagicMock(return_value=_FakeFolioState())
        kwargs = self._common_kwargs(run_mock, read_mock)
        with pytest.raises(RuntimeError, match="boom"):
            _run_one(**kwargs)
        read_mock.assert_not_called()

    def test_export_json_called_when_export_path_given(self, tmp_path):
        run_mock = MagicMock(return_value=(MagicMock(), {"a": "b"}))
        read_mock = MagicMock(return_value=_FakeFolioState())
        export_mock = MagicMock()
        export_path = str(tmp_path / "out" / "001r.json")
        kwargs = self._common_kwargs(
            run_mock, read_mock,
            export_json_path=export_path, export_json=export_mock,
        )
        _run_one(**kwargs)
        export_mock.assert_called_once()
        assert export_mock.call_args.args[3] == export_path

    def test_mei_json_written_when_mei_path_given(self):
        run_mock = MagicMock(return_value=(MagicMock(), {}))
        read_mock = MagicMock(return_value=_FakeFolioState())
        build_payload_mock = MagicMock(return_value={"payload": True})
        write_mei_mock = MagicMock()
        kwargs = self._common_kwargs(
            run_mock, read_mock,
            mei_json_path="out.json",
            build_pipeline_payload=build_payload_mock,
            write_mei_json=write_mei_mock,
        )
        _run_one(**kwargs)
        build_payload_mock.assert_called_once()
        write_mei_mock.assert_called_once_with({"payload": True}, "out.json")

    def test_folio_state_copied_to_states_dir_when_given(self, tmp_path):
        run_mock = MagicMock(return_value=(MagicMock(), {}))
        read_mock = MagicMock(return_value=_FakeFolioState())
        states_dir = tmp_path / "states"
        states_dir.mkdir()
        kwargs = self._common_kwargs(
            run_mock, read_mock,
            folio_states_dir=str(states_dir), folio="001r",
        )
        _run_one(**kwargs)
        assert (states_dir / "state_001r.json").exists()


class TestMainContiguityWiring:
    """Integration-level tests for main()'s per-folio loop: a failed
    contiguity check resets prev_state, but (mothra-text#58) leaves
    infer_continuation True so build_flat_text_and_anchors' hop-aware
    CSV-scan (mothra-text#55/#56) can still supply the correct continuation
    when the true predecessor has its own CSV row - exactly as a standalone
    run of that folio would. Exercised via main() itself (not just
    _are_contiguous in isolation) since the wiring lives in the loop, not in
    _are_contiguous.
    """

    def _install_fake_run_pipeline_modules(
        self, monkeypatch, run_mock, folio_state_mock=None
    ):
        fake_run_pipeline = types.ModuleType("run_pipeline")
        fake_run_pipeline.run = run_mock
        fake_run_pipeline.export_json = MagicMock()
        fake_run_pipeline._build_pipeline_payload = MagicMock()
        fake_run_pipeline._write_mei_json = MagicMock()

        fake_nw = types.ModuleType("steps.nw_chant_allocator")
        fake_nw.read_folio_state = folio_state_mock or MagicMock(
            return_value=_FakeFolioState()
        )

        fake_gt = types.ModuleType("steps.gt_manifest")
        fake_gt.fetch_cantus_csv = MagicMock()
        fake_gt.load_local_csv = MagicMock()
        fake_gt.make_output_stem = MagicMock()

        monkeypatch.setitem(sys.modules, "run_pipeline", fake_run_pipeline)
        monkeypatch.setitem(sys.modules, "steps.nw_chant_allocator", fake_nw)
        monkeypatch.setitem(sys.modules, "steps.gt_manifest", fake_gt)

    def _make_images(self, tmp_path, *names):
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"")
            paths.append(str(p))
        return paths

    def test_non_contiguous_folio_resets_state_but_keeps_infer_continuation_true(
        self, tmp_path, monkeypatch
    ):
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return MagicMock(), {}

        self._install_fake_run_pipeline_modules(monkeypatch, fake_run)
        images = self._make_images(tmp_path, "001r.jpg", "005r.jpg")
        monkeypatch.setattr(sys, "argv", [
            "run_chain.py",
            "--images", *images,
            "--folios", "001r", "005r",
            "--source-id", "123",
            "--stub-mode",
        ])

        run_chain.main()

        assert len(calls) == 2
        assert calls[0]["infer_continuation"] is True
        assert calls[1]["infer_continuation"] is True
        assert calls[1]["prev_folio_state"] is None

    def test_contiguous_folios_keep_infer_continuation_true(
        self, tmp_path, monkeypatch
    ):
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return MagicMock(), {}

        self._install_fake_run_pipeline_modules(monkeypatch, fake_run)
        images = self._make_images(tmp_path, "001r.jpg", "001v.jpg")
        monkeypatch.setattr(sys, "argv", [
            "run_chain.py",
            "--images", *images,
            "--folios", "001r", "001v",
            "--source-id", "123",
            "--stub-mode",
        ])

        run_chain.main()

        assert len(calls) == 2
        assert calls[0]["infer_continuation"] is True
        assert calls[1]["infer_continuation"] is True

    def test_chain_aborts_on_failure_and_reports_partial_progress(
        self, tmp_path, monkeypatch, capsys
    ):
        # main() calls logging.basicConfig(..., force=True), which replaces
        # pytest's caplog handler on the root logger with its own
        # StreamHandler -> assert on captured stderr text instead of
        # caplog.records, which force=True would otherwise silently drop.
        call_count = {"n": 0}

        def fake_run(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("segmentation failed")
            return MagicMock(), {}

        self._install_fake_run_pipeline_modules(monkeypatch, fake_run)
        images = self._make_images(
            tmp_path, "001r.jpg", "001v.jpg", "002r.jpg"
        )
        monkeypatch.setattr(sys, "argv", [
            "run_chain.py",
            "--images", *images,
            "--folios", "001r", "001v", "002r",
            "--source-id", "123",
            "--stub-mode",
        ])

        with pytest.raises(SystemExit) as exc_info:
            run_chain.main()

        assert exc_info.value.code == 1
        assert call_count["n"] == 2
        stderr = capsys.readouterr().err
        assert "Completed 1/3" in stderr

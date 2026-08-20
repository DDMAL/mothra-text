import { describe, it, expect, vi } from "vitest";
import { act } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ImageCanvas } from "./ImageCanvas";
import { samplePipeline } from "../test/fixtures/samplePipeline";
import type { LayerKey } from "../types";

vi.mock("openseadragon");
import OpenSeadragonMock from "openseadragon";
import { __getLastViewer } from "../../__mocks__/openseadragon";

const allLayersOn: Record<LayerKey, boolean> = { lines: true, words: true, syllables: true };

function renderCanvas(layers: Record<LayerKey, boolean> = allLayersOn) {
  const utils = render(
    <ImageCanvas imageUrl="blob:test-image" data={samplePipeline} layers={layers} />
  );
  const viewer = __getLastViewer();
  if (!viewer) throw new Error("expected a fake OpenSeadragon viewer to have been created");
  act(() => {
    viewer.__fireEvent("open");
  });
  return { ...utils, viewer };
}

describe("ImageCanvas", () => {
  it("constructs the OpenSeadragon viewer with the given image URL", () => {
    renderCanvas();
    expect(OpenSeadragonMock).toHaveBeenCalledWith(
      expect.objectContaining({
        tileSources: { type: "image", url: "blob:test-image" },
      })
    );
  });

  it("renders all line, word, and syllable shapes when every layer is on", () => {
    renderCanvas();
    expect(screen.getByTestId("line-line_0")).toBeInTheDocument();
    expect(screen.getByTestId("line-line_1")).toBeInTheDocument();
    expect(screen.getByTestId("word-line_0_word_0")).toBeInTheDocument();
    expect(screen.getByTestId("word-line_0_word_1")).toBeInTheDocument();
    expect(screen.getByTestId("word-line_1_word_0")).toBeInTheDocument();
    expect(screen.getByTestId("syllable-line_0_word_0_syl_0")).toBeInTheDocument();
    expect(screen.getByTestId("syllable-line_0_word_0_syl_1")).toBeInTheDocument();
  });

  it("hides only line shapes when the lines layer is off", () => {
    renderCanvas({ ...allLayersOn, lines: false });
    expect(screen.queryByTestId("line-line_0")).not.toBeInTheDocument();
    expect(screen.getByTestId("word-line_0_word_0")).toBeInTheDocument();
    expect(screen.getByTestId("syllable-line_0_word_0_syl_0")).toBeInTheDocument();
  });

  it("hides only word shapes when the words layer is off", () => {
    renderCanvas({ ...allLayersOn, words: false });
    expect(screen.getByTestId("line-line_0")).toBeInTheDocument();
    expect(screen.queryByTestId("word-line_0_word_0")).not.toBeInTheDocument();
    expect(screen.getByTestId("syllable-line_0_word_0_syl_0")).toBeInTheDocument();
  });

  it("hides only syllable shapes when the syllables layer is off", () => {
    renderCanvas({ ...allLayersOn, syllables: false });
    expect(screen.getByTestId("line-line_0")).toBeInTheDocument();
    expect(screen.getByTestId("word-line_0_word_0")).toBeInTheDocument();
    expect(screen.queryByTestId("syllable-line_0_word_0_syl_0")).not.toBeInTheDocument();
  });

  it("colors words by source: gt is teal, fallback is rose", () => {
    renderCanvas();
    const gtWord = screen.getByTestId("word-line_0_word_0").querySelector("polygon");
    const fallbackWord = screen.getByTestId("word-line_0_word_1").querySelector("polygon");
    expect(gtWord).toHaveAttribute("stroke", "rgb(45,212,191)");
    expect(fallbackWord).toHaveAttribute("stroke", "rgb(251,113,133)");
  });

  it("shows the selected line's concatenated word text on click, and hides it again on a second click", () => {
    renderCanvas();
    const linePolygon = screen.getByTestId("line-line_0");

    fireEvent.click(linePolygon);
    expect(screen.getByText("Alleluia laus")).toBeInTheDocument();

    fireEvent.click(linePolygon);
    expect(screen.queryByText("Alleluia laus")).not.toBeInTheDocument();
  });

  it("destroys the OpenSeadragon viewer on unmount", () => {
    const { unmount, viewer } = renderCanvas();
    unmount();
    expect(viewer.destroy).toHaveBeenCalledTimes(1);
  });
});

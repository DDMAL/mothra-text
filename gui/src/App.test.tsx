import { describe, it, expect, vi } from "vitest";
import { act } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { samplePipeline } from "./test/fixtures/samplePipeline";

vi.mock("openseadragon");
import { __getLastViewer } from "../__mocks__/openseadragon";

function makeImageFile() {
  return new File(["fake-image-bytes"], "folio.png", { type: "image/png" });
}

function makeJsonFile(contents: unknown = samplePipeline) {
  return new File([JSON.stringify(contents)], "pipeline.json", { type: "application/json" });
}

describe("App", () => {
  it("shows the drop-zone UI on initial render, with no layer buttons yet", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Pipeline Inspector" })).toBeInTheDocument();
    expect(screen.getByLabelText(/Folio image/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Pipeline JSON/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Lines/ })).not.toBeInTheDocument();
  });

  it("transitions to the ready view after selecting an image and a pipeline JSON via the file inputs", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/Folio image/), makeImageFile());
    await user.upload(screen.getByLabelText(/Pipeline JSON/), makeJsonFile());

    expect(await screen.findByRole("button", { name: "Lines (2)" })).toBeInTheDocument();
  });

  it("transitions to the ready view after dropping an image and a pipeline JSON onto the drop zone", async () => {
    render(<App />);
    const dropZone = screen.getByTestId("drop-zone");

    fireEvent.drop(dropZone, { dataTransfer: { files: [makeImageFile(), makeJsonFile()] } });

    expect(await screen.findByRole("button", { name: "Lines (2)" })).toBeInTheDocument();
  });

  it("shows an error banner and stays on the drop-zone view when the JSON is malformed", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/Pipeline JSON/), makeJsonFile({ not: "a pipeline" }));

    expect(
      await screen.findByText('Error: Pipeline JSON must have a "lines" array')
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Lines/ })).not.toBeInTheDocument();
  });

  it("resets back to the drop-zone view when Open is clicked, clearing any error", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/Folio image/), makeImageFile());
    await user.upload(screen.getByLabelText(/Pipeline JSON/), makeJsonFile());
    expect(await screen.findByRole("button", { name: "Lines (2)" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open" }));

    expect(screen.getByRole("heading", { name: "Pipeline Inspector" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Lines/ })).not.toBeInTheDocument();
  });

  it("threads layer state down to ImageCanvas: toggling Lines in TopBar hides line shapes", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/Folio image/), makeImageFile());
    await user.upload(screen.getByLabelText(/Pipeline JSON/), makeJsonFile());
    await screen.findByRole("button", { name: "Lines (2)" });

    const viewer = __getLastViewer();
    if (!viewer) throw new Error("expected a fake OpenSeadragon viewer to have been created");
    act(() => {
      viewer.__fireEvent("open");
    });

    expect(screen.getByTestId("line-line_0")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Lines (2)" }));

    expect(screen.queryByTestId("line-line_0")).not.toBeInTheDocument();
    // unaffected: words layer stays on
    expect(screen.getByTestId("word-line_0_word_0")).toBeInTheDocument();
  });
});

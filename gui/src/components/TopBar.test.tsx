import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TopBar } from "./TopBar";
import { samplePipeline } from "../test/fixtures/samplePipeline";
import type { LayerKey } from "../types";

const allLayersOn: Record<LayerKey, boolean> = { lines: true, words: true, syllables: true };

describe("TopBar", () => {
  it("shows layer buttons with live counts derived from the data", () => {
    render(
      <TopBar data={samplePipeline} layers={allLayersOn} onToggle={vi.fn()} onOpenFiles={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: "Lines (2)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Words (3)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Syllables (2)" })).toBeInTheDocument();
  });

  it("calls onToggle with the correct key for each layer button", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <TopBar data={samplePipeline} layers={allLayersOn} onToggle={onToggle} onOpenFiles={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: "Lines (2)" }));
    expect(onToggle).toHaveBeenLastCalledWith("lines");

    await user.click(screen.getByRole("button", { name: "Words (3)" }));
    expect(onToggle).toHaveBeenLastCalledWith("words");

    await user.click(screen.getByRole("button", { name: "Syllables (2)" }));
    expect(onToggle).toHaveBeenLastCalledWith("syllables");

    expect(onToggle).toHaveBeenCalledTimes(3);
  });

  it("shows the GT/OCR fallback legend only when the words layer is active", () => {
    const { rerender } = render(
      <TopBar
        data={samplePipeline}
        layers={{ ...allLayersOn, words: false }}
        onToggle={vi.fn()}
        onOpenFiles={vi.fn()}
      />
    );
    expect(screen.queryByText("GT")).not.toBeInTheDocument();
    expect(screen.queryByText("OCR fallback")).not.toBeInTheDocument();

    rerender(
      <TopBar data={samplePipeline} layers={allLayersOn} onToggle={vi.fn()} onOpenFiles={vi.fn()} />
    );
    expect(screen.getByText("GT")).toBeInTheDocument();
    expect(screen.getByText("OCR fallback")).toBeInTheDocument();
  });

  it("shows the folio badge only when data is present", () => {
    const { rerender } = render(
      <TopBar data={samplePipeline} layers={allLayersOn} onToggle={vi.fn()} onOpenFiles={vi.fn()} />
    );
    expect(screen.getByText("001r")).toBeInTheDocument();

    rerender(<TopBar data={null} layers={allLayersOn} onToggle={vi.fn()} onOpenFiles={vi.fn()} />);
    expect(screen.queryByText("001r")).not.toBeInTheDocument();
  });

  it("hides layer buttons, folio badge, and legend when data is null, but keeps Open working", async () => {
    const user = userEvent.setup();
    const onOpenFiles = vi.fn();
    render(<TopBar data={null} layers={allLayersOn} onToggle={vi.fn()} onOpenFiles={onOpenFiles} />);

    expect(screen.queryByRole("button", { name: /Lines/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Words/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Syllables/ })).not.toBeInTheDocument();
    expect(screen.queryByText("GT")).not.toBeInTheDocument();

    const openButton = screen.getByRole("button", { name: "Open" });
    expect(openButton).toBeInTheDocument();
    await user.click(openButton);
    expect(onOpenFiles).toHaveBeenCalledTimes(1);
  });
});

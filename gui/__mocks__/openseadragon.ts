import { vi } from "vitest";

type Handler = () => void;

function createFakeViewer() {
  const handlers: Record<string, Handler[]> = {};
  return {
    addHandler: vi.fn((event: string, fn: Handler) => {
      (handlers[event] ??= []).push(fn);
    }),
    destroy: vi.fn(),
    viewport: {
      imageToViewportCoordinates: vi.fn((x: number, y: number) => ({
        x: x / 1000,
        y: y / 1000,
      })),
      viewportToViewerElementCoordinates: vi.fn(
        (pt: { x: number; y: number }) => ({ x: pt.x * 500, y: pt.y * 500 })
      ),
    },
    __fireEvent: (event: string) => {
      (handlers[event] ?? []).forEach((fn) => fn());
    },
  };
}

let lastViewer: ReturnType<typeof createFakeViewer> | null = null;

const OpenSeadragon = vi.fn(() => {
  lastViewer = createFakeViewer();
  return lastViewer;
});

export function __getLastViewer() {
  return lastViewer;
}

export default OpenSeadragon;

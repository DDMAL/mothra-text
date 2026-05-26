import OpenSeadragon from "openseadragon";
import { useEffect, useRef, useState } from "react";
import type { LayerKey, LineEntry, PipelineData, WordEntry } from "../types";

interface Props {
  imageUrl: string;
  data: PipelineData;
  layers: Record<LayerKey, boolean>;
}

// Recompute all polygon/rect points in SVG element space.
function toSvgPoints(
  viewer: OpenSeadragon.Viewer,
  imgPoints: [number, number][]
): string {
  return imgPoints
    .map(([x, y]) => {
      const vp = viewer.viewport.imageToViewportCoordinates(x, y);
      const el = viewer.viewport.viewportToViewerElementCoordinates(vp);
      return `${el.x},${el.y}`;
    })
    .join(" ");
}

function bboxToPoints(bbox: [number, number, number, number]): [number, number][] {
  const [x0, y0, x1, y1] = bbox;
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}

interface SvgState {
  linePolygons: { key: string; points: string; line: LineEntry }[];
  wordRects: { key: string; points: string; word: WordEntry; cx: number; cy: number; width: number }[];
}

export function ImageCanvas({ imageUrl, data, layers }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const [svgState, setSvgState] = useState<SvgState>({ linePolygons: [], wordRects: [] });
  const [svgSize, setSvgSize] = useState({ w: 0, h: 0 });

  // Init OSD viewer
  useEffect(() => {
    if (!containerRef.current) return;
    const viewer = OpenSeadragon({
      element: containerRef.current,
      tileSources: { type: "image", url: imageUrl },
      showNavigationControl: false,
      animationTime: 0.3,
      minZoomLevel: 0.5,
    });
    viewerRef.current = viewer;

    const recompute = () => {
      const el = containerRef.current;
      if (!el) return;
      setSvgSize({ w: el.clientWidth, h: el.clientHeight });

      const linePolygons = data.lines.map((line) => ({
        key: line.label,
        points: toSvgPoints(viewer, line.polygon),
        line,
      }));

      const wordRects = data.lines.flatMap((line) =>
        line.words.map((word) => {
          const pts = bboxToPoints(word.bbox);
          const svgPts = pts.map(([x, y]) => {
            const vp = viewer.viewport.imageToViewportCoordinates(x, y);
            return viewer.viewport.viewportToViewerElementCoordinates(vp);
          });
          // top-left and width in element space for label positioning
          const minX = Math.min(...svgPts.map((p) => p.x));
          const minY = Math.min(...svgPts.map((p) => p.y));
          const maxX = Math.max(...svgPts.map((p) => p.x));
          const w = maxX - minX;
          return {
            key: word.label,
            points: pts.map(([x, y]) => {
              const vp = viewer.viewport.imageToViewportCoordinates(x, y);
              const el2 = viewer.viewport.viewportToViewerElementCoordinates(vp);
              return `${el2.x},${el2.y}`;
            }).join(" "),
            word,
            cx: minX,
            cy: minY,
            width: w,
          };
        })
      );

      setSvgState({ linePolygons, wordRects });
    };

    viewer.addHandler("open", recompute);
    viewer.addHandler("update-viewport", recompute);
    viewer.addHandler("resize", recompute);

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [imageUrl, data]);

  return (
    <div className="relative flex-1 overflow-hidden bg-gray-900">
      {/* OSD target */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* SVG overlay — pointer-events none so OSD pan/zoom still works */}
      <svg
        className="absolute inset-0 pointer-events-none"
        width={svgSize.w}
        height={svgSize.h}
      >
        {/* Line polygons */}
        {layers.lines &&
          svgState.linePolygons.map(({ key, points, line }) => (
            <g key={key}>
              <polygon
                points={points}
                fill="rgba(168,85,247,0.10)"
                stroke="rgb(168,85,247)"
                strokeWidth={1.5}
              />
              <title>{line.label}</title>
            </g>
          ))}

        {/* Word rects */}
        {layers.words &&
          svgState.wordRects.map(({ key, points, word, cy, width }) => (
            <g key={key} style={{ pointerEvents: "all", cursor: "default" }}>
              <polygon
                points={points}
                fill="rgba(45,212,191,0.10)"
                stroke="rgb(45,212,191)"
                strokeWidth={1}
              />
              <title>{word.text || "(empty)"}</title>
              {/* Text label — hidden when rect is too narrow */}
              {width > 24 && word.text && (
                <text
                  x={parseFloat(points.split(" ")[0].split(",")[0]) + 2}
                  y={cy - 2}
                  fontSize={9}
                  fontFamily="ui-monospace, monospace"
                  fill="rgb(45,212,191)"
                  style={{ userSelect: "none" }}
                >
                  {word.text}
                </text>
              )}
            </g>
          ))}
      </svg>
    </div>
  );
}

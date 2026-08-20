import OpenSeadragon from "openseadragon";
import { useEffect, useRef, useState } from "react";
import type { LayerKey, LineEntry, PipelineData, SyllableEntry, WordEntry } from "../types";

interface Props {
  imageUrl: string;
  data: PipelineData;
  layers: Record<LayerKey, boolean>;
}

function toSvgPointsArray(
  viewer: OpenSeadragon.Viewer,
  imgPoints: [number, number][]
): { x: number; y: number }[] {
  return imgPoints.map(([x, y]) => {
    const vp = viewer.viewport.imageToViewportCoordinates(x, y);
    return viewer.viewport.viewportToViewerElementCoordinates(vp);
  });
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
  linePolygons: {
    key: string;
    points: string;
    line: LineEntry;
  }[];
  wordRects: {
    key: string;
    points: string;
    word: WordEntry;
    cx: number;
    cy: number;
    width: number;
    height: number;
  }[];
  syllableRects: {
    key: string;
    points: string;
    syllable: SyllableEntry;
    cx: number;
    cy: number;
    width: number;
    height: number;
  }[];
}

export function ImageCanvas({ imageUrl, data, layers }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const [svgState, setSvgState] = useState<SvgState>({ linePolygons: [], wordRects: [], syllableRects: [] });
  const [svgSize, setSvgSize] = useState({ w: 0, h: 0 });
  const [selectedLineKey, setSelectedLineKey] = useState<string | null>(null);

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

      const linePolygons = data.lines.map((line) => {
        const svgPts = toSvgPointsArray(viewer, line.polygon);
        return {
          key: line.label,
          points: svgPts.map((p) => `${p.x},${p.y}`).join(" "),
          line,
        };
      });

      const wordRects = data.lines.flatMap((line) =>
        line.words.map((word) => {
          const pts = bboxToPoints(word.bbox);
          const svgPts = pts.map(([x, y]) => {
            const vp = viewer.viewport.imageToViewportCoordinates(x, y);
            return viewer.viewport.viewportToViewerElementCoordinates(vp);
          });
          const minX = Math.min(...svgPts.map((p) => p.x));
          const minY = Math.min(...svgPts.map((p) => p.y));
          const maxX = Math.max(...svgPts.map((p) => p.x));
          const maxY = Math.max(...svgPts.map((p) => p.y));
          return {
            key: word.label,
            points: svgPts.map((p) => `${p.x},${p.y}`).join(" "),
            word,
            cx: minX,
            cy: minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        })
      );

      const syllableRects = data.lines.flatMap((line) =>
        line.words.flatMap((word) =>
          (word.syllables ?? []).map((syl) => {
            const pts = bboxToPoints(syl.bbox);
            const svgPts = pts.map(([x, y]) => {
              const vp = viewer.viewport.imageToViewportCoordinates(x, y);
              return viewer.viewport.viewportToViewerElementCoordinates(vp);
            });
            const minX = Math.min(...svgPts.map((p) => p.x));
            const minY = Math.min(...svgPts.map((p) => p.y));
            const maxX = Math.max(...svgPts.map((p) => p.x));
            const maxY = Math.max(...svgPts.map((p) => p.y));
            return {
              key: syl.label,
              points: svgPts.map((p) => `${p.x},${p.y}`).join(" "),
              syllable: syl,
              cx: minX,
              cy: minY,
              width: maxX - minX,
              height: maxY - minY,
            };
          })
        )
      );

      setSvgState({ linePolygons, wordRects, syllableRects });
    };

    viewer.addHandler("open", recompute);
    viewer.addHandler("update-viewport", recompute);
    viewer.addHandler("resize", recompute);

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [imageUrl, data]);

  const selectedLine = selectedLineKey
    ? svgState.linePolygons.find((lp) => lp.key === selectedLineKey) ?? null
    : null;

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
        {/* Line polygons — click to show words for that line */}
        {layers.lines &&
          svgState.linePolygons.map(({ key, points, line }) => {
            const selected = selectedLineKey === key;
            return (
              <polygon
                key={key}
                data-testid={`line-${key}`}
                points={points}
                fill={selected ? "rgba(168,85,247,0.20)" : "rgba(168,85,247,0.10)"}
                stroke={selected ? "rgb(216,180,254)" : "rgb(168,85,247)"}
                strokeWidth={selected ? 2 : 1.5}
                style={{ pointerEvents: "all", cursor: "pointer" }}
                onClick={() =>
                  setSelectedLineKey((prev) => (prev === line.label ? null : line.label))
                }
              />
            );
          })}

        {/* Syllable rects — amber, rendered below word boxes */}
        {layers.syllables &&
          svgState.syllableRects.map(({ key, points, syllable, cx, cy, width, height }) => {
            const fs = Math.max(9, Math.round(height * 0.4));
            return (
              <g key={key} data-testid={`syllable-${key}`}>
                <polygon
                  points={points}
                  fill="rgba(251,146,60,0.10)"
                  stroke="rgb(251,146,60)"
                  strokeWidth={1}
                />
                {syllable.text && syllable.text.length * fs * 0.4 <= width && (
                  <text
                    x={cx + 2}
                    y={cy - 2}
                    fontSize={fs}
                    fontFamily="ui-monospace, monospace"
                    fill="rgb(251,146,60)"
                    style={{ userSelect: "none", pointerEvents: "none" }}
                  >
                    {syllable.text}
                  </text>
                )}
              </g>
            );
          })}

        {/* Word rects — teal for GT words, rose for OCR fallback */}
        {layers.words &&
          svgState.wordRects.map(({ key, points, word, cx, cy, width, height }) => {
            const fs = Math.max(9, Math.round(height * 0.4));
            const isGt = word.source === "gt";
            const color = isGt ? "rgb(45,212,191)" : "rgb(251,113,133)";
            const fillColor = isGt ? "rgba(45,212,191,0.10)" : "rgba(251,113,133,0.10)";
            return (
              <g key={key} data-testid={`word-${key}`}>
                <polygon
                  points={points}
                  fill={fillColor}
                  stroke={color}
                  strokeWidth={1}
                />
                {word.text && word.text.length * fs * 0.4 <= width && (
                  <text
                    x={cx + 2}
                    y={cy - 2}
                    fontSize={fs}
                    fontFamily="ui-monospace, monospace"
                    fill={color}
                    style={{ userSelect: "none", pointerEvents: "none" }}
                  >
                    {word.text}
                  </text>
                )}
              </g>
            );
          })}
      </svg>

      {/* Line text panel — shown when a line polygon is clicked */}
      {selectedLine && (
        <div className="absolute top-2 right-2 max-w-xs bg-gray-900/95 border border-purple-500 rounded px-3 py-2 text-xs text-purple-200 pointer-events-none z-10">
          <div className="text-gray-500 font-mono text-[10px] mb-1 leading-none">
            {selectedLineKey}
          </div>
          <div>
            {selectedLine.line.words
              .map((w) => w.text)
              .filter(Boolean)
              .join(" ") || "(no text)"}
          </div>
        </div>
      )}
    </div>
  );
}

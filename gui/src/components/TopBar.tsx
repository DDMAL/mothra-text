import type { LayerKey, PipelineData } from "../types";

interface Props {
  data: PipelineData | null;
  layers: Record<LayerKey, boolean>;
  onToggle: (key: LayerKey) => void;
  onOpenFiles: () => void;
}

const LAYER_STYLES: Record<LayerKey, { active: string; inactive: string; label: string }> = {
  lines: {
    active: "bg-purple-600 text-white border-purple-500",
    inactive: "bg-gray-700 text-gray-300 border-gray-600 hover:bg-gray-600",
    label: "Lines",
  },
  words: {
    active: "bg-teal-600 text-white border-teal-500",
    inactive: "bg-gray-700 text-gray-300 border-gray-600 hover:bg-gray-600",
    label: "Words",
  },
};

export function TopBar({ data, layers, onToggle, onOpenFiles }: Props) {
  const wordCount = data?.lines.reduce((n, l) => n + l.words.length, 0) ?? 0;
  const counts: Record<LayerKey, number> = {
    lines: data?.lines.length ?? 0,
    words: wordCount,
  };

  return (
    <div className="flex items-center h-12 px-4 gap-4 bg-gray-800 border-b border-gray-700 shrink-0">
      {/* Branding */}
      <span className="text-gray-100 text-sm font-semibold tracking-tight select-none">
        Pipeline Inspector
      </span>

      {/* Folio name */}
      {data && (
        <span className="text-gray-400 text-xs font-mono truncate max-w-xs">
          {data.folio}
        </span>
      )}

      <div className="flex-1" />

      {/* Layer toggles */}
      {data && (
        <div className="flex items-center gap-1.5">
          {(["lines", "words"] as LayerKey[]).map((key) => {
            const cfg = LAYER_STYLES[key];
            const active = layers[key];
            return (
              <button
                key={key}
                onClick={() => onToggle(key)}
                className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                  active ? cfg.active : cfg.inactive
                }`}
              >
                {cfg.label} ({counts[key]})
              </button>
            );
          })}
        </div>
      )}

      {/* Open */}
      <button
        onClick={onOpenFiles}
        className="px-2.5 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors"
      >
        Open
      </button>
    </div>
  );
}

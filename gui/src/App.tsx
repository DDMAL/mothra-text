import { useCallback, useState } from "react";
import { ImageCanvas } from "./components/ImageCanvas";
import { TopBar } from "./components/TopBar";
import { loadPipelineJson } from "./data/loader";
import type { LayerKey, PipelineData } from "./types";

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [data, setData] = useState<PipelineData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    lines: true,
    words: true,
    syllables: true,
  });

  const handleToggle = useCallback((key: LayerKey) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleImageFile = useCallback((file: File) => {
    setImageUrl(URL.createObjectURL(file));
    setError(null);
  }, []);

  const handleJsonFile = useCallback(async (file: File) => {
    try {
      const parsed = await loadPipelineJson(file);
      setData(parsed);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const handleOpenFiles = useCallback(() => {
    setImageUrl(null);
    setData(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      for (const file of Array.from(e.dataTransfer.files)) {
        if (file.type.startsWith("image/")) handleImageFile(file);
        else if (file.name.endsWith(".json")) handleJsonFile(file);
      }
    },
    [handleImageFile, handleJsonFile]
  );

  const ready = imageUrl !== null && data !== null;

  return (
    <div className="flex flex-col h-full bg-gray-900 text-gray-100">
      <TopBar
        data={data}
        layers={layers}
        onToggle={handleToggle}
        onOpenFiles={handleOpenFiles}
      />

      {ready ? (
        <ImageCanvas imageUrl={imageUrl} data={data} layers={layers} />
      ) : (
        <div
          className="flex flex-1 items-center justify-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <div className="w-full max-w-md p-8 rounded-lg bg-gray-800 border border-gray-700 flex flex-col gap-6">
            <h1 className="text-lg font-semibold text-gray-100 text-center">
              Pipeline Inspector
            </h1>
            <p className="text-xs text-gray-400 text-center">
              Load a folio image and its pipeline JSON to inspect line and word
              bounding boxes. You can also drag and drop both files anywhere on
              this screen.
            </p>

            <label className="flex flex-col gap-1.5">
              <span className="text-xs text-gray-300 font-medium">Folio image</span>
              <div
                className={`flex items-center gap-2 px-3 py-2 rounded border text-xs cursor-pointer transition-colors ${
                  imageUrl
                    ? "bg-gray-700 border-purple-500 text-purple-300"
                    : "bg-gray-700 border-gray-600 text-gray-400 hover:border-gray-500"
                }`}
              >
                <span>{imageUrl ? "Image loaded" : "Choose JPEG / PNG…"}</span>
              </div>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleImageFile(f);
                }}
              />
            </label>

            <label className="flex flex-col gap-1.5">
              <span className="text-xs text-gray-300 font-medium">Pipeline JSON</span>
              <div
                className={`flex items-center gap-2 px-3 py-2 rounded border text-xs cursor-pointer transition-colors ${
                  data
                    ? "bg-gray-700 border-teal-500 text-teal-300"
                    : "bg-gray-700 border-gray-600 text-gray-400 hover:border-gray-500"
                }`}
              >
                <span>
                  {data
                    ? `${data.folio} — ${data.lines.length} lines`
                    : "Choose pipeline .json…"}
                </span>
              </div>
              <input
                type="file"
                accept=".json,application/json"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleJsonFile(f);
                }}
              />
            </label>

            {error && (
              <p className="text-xs text-red-400 bg-red-950 border border-red-800 rounded px-3 py-2">
                {error}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

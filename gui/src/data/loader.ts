import type { PipelineData } from "../types";

export function parsePipelineJson(raw: unknown): PipelineData {
  if (typeof raw !== "object" || raw === null) {
    throw new Error("Pipeline JSON must be an object");
  }
  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.lines)) {
    throw new Error('Pipeline JSON must have a "lines" array');
  }
  if (typeof obj.image_width !== "number" || typeof obj.image_height !== "number") {
    throw new Error('Pipeline JSON must have numeric "image_width" and "image_height"');
  }
  return raw as PipelineData;
}

export async function loadPipelineJson(file: File): Promise<PipelineData> {
  const text = await file.text();
  const raw: unknown = JSON.parse(text);
  return parsePipelineJson(raw);
}

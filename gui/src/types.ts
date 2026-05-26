export interface WordEntry {
  label: string;
  text: string;
  bbox: [number, number, number, number]; // xmin, ymin, xmax, ymax — absolute image px
  source: "gt" | "fallback";
}

export interface LineEntry {
  label: string;
  bbox: [number, number, number, number];
  polygon: [number, number][]; // [[x, y], ...] — absolute image px
  text: string;
  words: WordEntry[];
}

export interface PipelineData {
  folio: string;
  image_width: number;
  image_height: number;
  lines: LineEntry[];
}

export type LayerKey = "lines" | "words";

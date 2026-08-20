export interface SyllableEntry {
  label: string;
  text: string;
  bbox: [number, number, number, number]; // xmin, ymin, xmax, ymax — absolute image px
}

export interface WordEntry {
  label: string;
  text: string;
  bbox: [number, number, number, number]; // xmin, ymin, xmax, ymax — absolute image px
  source: "gt" | "fallback";
  syllables?: SyllableEntry[];
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
  mode: "cantus_aligned" | "ocr_only";
  image_width: number;
  image_height: number;
  lines: LineEntry[];
}

export type LayerKey = "lines" | "words" | "syllables";

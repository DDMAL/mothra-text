import type { PipelineData } from "../../types";

export const samplePipeline: PipelineData = {
  folio: "001r",
  mode: "cantus_aligned",
  image_width: 2000,
  image_height: 3000,
  lines: [
    {
      label: "line_0",
      bbox: [100, 200, 1800, 260],
      polygon: [
        [100, 200],
        [1800, 200],
        [1800, 260],
        [100, 260],
      ],
      text: "Alleluia laus",
      words: [
        {
          label: "line_0_word_0",
          text: "Alleluia",
          bbox: [100, 200, 400, 260],
          source: "gt",
          syllables: [
            { label: "line_0_word_0_syl_0", text: "Al-", bbox: [100, 200, 200, 260] },
            { label: "line_0_word_0_syl_1", text: "le-lu-ia", bbox: [200, 200, 400, 260] },
          ],
        },
        {
          label: "line_0_word_1",
          text: "laus",
          bbox: [420, 200, 600, 260],
          source: "fallback",
        },
      ],
    },
    {
      label: "line_1",
      bbox: [100, 300, 1800, 360],
      polygon: [
        [100, 300],
        [1800, 300],
        [1800, 360],
        [100, 360],
      ],
      text: "Deo",
      words: [
        { label: "line_1_word_0", text: "Deo", bbox: [100, 300, 300, 360], source: "gt" },
      ],
    },
  ],
};

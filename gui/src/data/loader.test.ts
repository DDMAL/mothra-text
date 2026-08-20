import { describe, it, expect } from "vitest";
import { parsePipelineJson, loadPipelineJson } from "./loader";
import { samplePipeline } from "../test/fixtures/samplePipeline";

describe("parsePipelineJson", () => {
  it("throws when the input is null", () => {
    expect(() => parsePipelineJson(null)).toThrow("Pipeline JSON must be an object");
  });

  it("throws when the input is not an object", () => {
    expect(() => parsePipelineJson("not an object")).toThrow(
      "Pipeline JSON must be an object"
    );
  });

  it("throws a lines-array error for a bare array (passes the object check, fails the lines check)", () => {
    expect(() => parsePipelineJson([])).toThrow('Pipeline JSON must have a "lines" array');
  });

  it("throws when lines is not an array", () => {
    expect(() => parsePipelineJson({ lines: "not-an-array" })).toThrow(
      'Pipeline JSON must have a "lines" array'
    );
  });

  it("throws when image_width/image_height are not numeric", () => {
    expect(() =>
      parsePipelineJson({ lines: [], image_width: "2000", image_height: 3000 })
    ).toThrow('Pipeline JSON must have numeric "image_width" and "image_height"');
  });

  it("returns the same object reference on the happy path", () => {
    const result = parsePipelineJson(samplePipeline);
    expect(result).toBe(samplePipeline);
  });
});

describe("loadPipelineJson", () => {
  it("parses a well-formed pipeline JSON file", async () => {
    const file = new File([JSON.stringify(samplePipeline)], "pipeline.json", {
      type: "application/json",
    });
    const result = await loadPipelineJson(file);
    expect(result).toEqual(samplePipeline);
  });

  it("rejects when the file contains invalid JSON syntax", async () => {
    const file = new File(["{not valid json"], "pipeline.json", {
      type: "application/json",
    });
    await expect(loadPipelineJson(file)).rejects.toThrow();
  });
});

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}

// React 19's act() checks this flag before deciding whether to warn about
// state updates outside act(); RTL no longer sets it for us automatically.
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

if (typeof URL.createObjectURL !== "function") {
  Object.defineProperty(URL, "createObjectURL", {
    value: () => "blob:mock-url",
    writable: true,
  });
}

// jsdom deliberately doesn't implement Blob/File body-reading methods
// (https://github.com/jsdom/jsdom/issues/2555); polyfill .text() via
// FileReader, which jsdom does implement, since loader.ts calls file.text().
if (typeof Blob.prototype.text !== "function") {
  Blob.prototype.text = function (this: Blob) {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

afterEach(() => {
  cleanup();
});

import { describe, expect, it } from "vitest";
import type { DetectedQuad } from "../gpu/contour";
import { frameHasDuplicateDecodedTagIds } from "./calibrationQuality";

function bareQuad(tagId?: number): DetectedQuad {
  return {
    corners: [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ],
    label: 1,
    count: 1000,
    aspectRatio: 1,
    area: 10000,
    gridCells: null,
    pattern: null,
    hasCorners: true,
    cornerDebug: { failureCode: 0, edgePixelCount: 100, minR2: 0.99, intersectionCount: 6 },
    ...(typeof tagId === "number" ? { decodedTagId: tagId } : {}),
  };
}

describe("frameHasDuplicateDecodedTagIds", () => {
  it("returns false when all ids unique", () => {
    expect(frameHasDuplicateDecodedTagIds([bareQuad(1), bareQuad(2)])).toBe(false);
  });

  it("returns true when same decoded id twice", () => {
    expect(frameHasDuplicateDecodedTagIds([bareQuad(5), bareQuad(5)])).toBe(true);
  });
});

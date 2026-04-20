import { describe, expect, it } from "vitest";
import { binarySearchMaxPassing, gridMaxPassing } from "./decodeStressSearch";

describe("decodeStressSearch", () => {
  it("binarySearchMaxPassing tightens a monotone pass band", () => {
    const passes = (t: number) => t < 0.3;
    const { maxPass, failHi } = binarySearchMaxPassing(passes, 0, 1, 48);
    expect(maxPass).toBeLessThan(0.301);
    expect(maxPass).toBeGreaterThan(0.29);
    expect(failHi).toBeGreaterThan(maxPass);
  });

  it("gridMaxPassing finds largest passing sample when pass→fail→pass exists", () => {
    const passes = (s: number) => s < 0.1 || s > 0.2;
    const { best, recoveries } = gridMaxPassing(passes, { hi: 0.35, step: 0.05 });
    expect(recoveries).toBeGreaterThan(0);
    expect(best).toBeGreaterThanOrEqual(0.2);
  });
});

import type { StripCorners } from "../../shared/types";
import { cornerErrorStats } from "./corners";

export function improvementCornerMaxPx(
  gt: StripCorners,
  initStrip: StripCorners,
  refinedStrip: StripCorners,
): { initMaxPx: number; refinedMaxPx: number; deltaMaxPx: number } {
  const init = cornerErrorStats(gt, initStrip);
  const ref = cornerErrorStats(gt, refinedStrip);
  return {
    initMaxPx: init.maxPx,
    refinedMaxPx: ref.maxPx,
    deltaMaxPx: init.maxPx - ref.maxPx,
  };
}

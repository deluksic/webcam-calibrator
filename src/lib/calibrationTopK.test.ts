import { describe, expect, it } from 'vitest';
import type { CalibrationSample } from './calibrationTypes';
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationSamplesTopK } from './calibrationTopK';
import type { Point } from './geometry';

const corners = (): Point[] =>
  Array.from({ length: 49 }, (_, i) => ({ x: i, y: 0 }));

function sample(
  frameId: number,
  tagId: number,
  score: number,
): CalibrationSample {
  return {
    frameId,
    tagId,
    rotation: 0,
    innerCorners: corners(),
    score,
  };
}

describe('mergeCalibrationSamplesTopK', () => {
  it('evicts lowest score when over K', () => {
    const k = 3;
    const a = mergeCalibrationSamplesTopK(
      [],
      [sample(1, 1, 0.5), sample(1, 2, 0.9), sample(1, 3, 0.7)],
      k,
    );
    expect(a.next.length).toBe(3);
    const b = mergeCalibrationSamplesTopK(a.next, [sample(2, 4, 0.2)], k);
    expect(b.evicted).toBe(1);
    expect(b.next.length).toBe(3);
    expect(b.next.some((s) => s.tagId === 4)).toBe(false);
    expect(b.next.some((s) => s.tagId === 1)).toBe(true);
  });

  it('respects DEFAULT_CALIBRATION_TOP_K export', () => {
    expect(DEFAULT_CALIBRATION_TOP_K).toBeGreaterThan(1000);
  });
});

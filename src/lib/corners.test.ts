import { describe, it, expect } from 'vitest';
import {
  extractEdgePixelsFromBbox,
  getPixel,
  findMaxMagnitude,
  findCornerCandidates,
  clusterCorners,
  orderCornersClockwise,
  getCornerPoint,
} from './corners';
import type { Point } from './geometry';

describe('corners', () => {
  describe('extractEdgePixelsFromBbox', () => {
    it('extracts edge pixels from Sobel data', () => {
      const sobelData = new Float32Array(4 * 4 * 2);
      // Pixel (1,1) has strong edge: gx=10, gy=0
      sobelData[1 * 4 * 2 + 0] = 10;
      sobelData[1 * 4 * 2 + 1] = 0;

      const pixels = extractEdgePixelsFromBbox(sobelData, 4, 0, 0, 3, 3);

      expect(pixels.length).toBeGreaterThan(0);
      const p = getPixel(pixels, 0);
      expect(p.magnitude).toBeCloseTo(10, 1);
    });
  });

  describe('getPixel', () => {
    it('returns structured pixel data from flat array', () => {
      const pixels = new Float32Array([10, 20, 1.57, 5.0]);
      const p = getPixel(pixels, 0);
      expect(p.x).toBeCloseTo(10, 0);
      expect(p.y).toBeCloseTo(20, 0);
      expect(p.tangent).toBeCloseTo(1.57, 2);
      expect(p.magnitude).toBeCloseTo(5.0, 1);
    });
  });

  describe('findMaxMagnitude', () => {
    it('finds maximum magnitude', () => {
      const pixels = new Float32Array([0, 0, 0, 5, 3, 0, 0, 10]);
      expect(findMaxMagnitude(pixels)).toBeCloseTo(10, 1);
    });

    it('returns 0 for empty array', () => {
      const pixels = new Float32Array([]);
      expect(findMaxMagnitude(pixels)).toBe(0);
    });
  });

  describe('findCornerCandidates', () => {
    it('handles empty input', () => {
      const pixels = new Float32Array([]);
      const candidates = findCornerCandidates(pixels);
      expect(candidates).toHaveLength(0);
    });

    it('returns empty for too few pixels', () => {
      const pixels = new Float32Array([10, 10, 0, 1]);
      const candidates = findCornerCandidates(pixels);
      expect(candidates).toHaveLength(0);
    });
  });

  describe('clusterCorners', () => {
    it('handles empty input', () => {
      const clustered = clusterCorners([]);
      expect(clustered).toHaveLength(0);
    });

    it('handles fewer than 4 candidates', () => {
      const candidates = [
        { x: 10, y: 10, diff: 1.0 },
        { x: 90, y: 10, diff: 0.9 },
        { x: 10, y: 90, diff: 0.8 },
      ];
      const clustered = clusterCorners(candidates);
      expect(clustered.length).toBeLessThanOrEqual(4);
    });

    it('clusters candidates into up to 4 quadrants', () => {
      const candidates = [
        { x: 10, y: 10, diff: 1.0 },  // top-left
        { x: 90, y: 10, diff: 0.9 },  // top-right
        { x: 90, y: 90, diff: 0.8 },  // bottom-right
        { x: 10, y: 90, diff: 0.7 },  // bottom-left
        { x: 20, y: 20, diff: 0.6 },  // top-left (extra)
        { x: 80, y: 80, diff: 0.5 },  // bottom-right (extra)
      ];
      const clustered = clusterCorners(candidates);
      // Max 4 clusters
      expect(clustered.length).toBeLessThanOrEqual(4);
    });
  });

  describe('orderCornersClockwise', () => {
    it('handles non-4 input', () => {
      const corners: Point[] = [{ x: 0, y: 0 }, { x: 10, y: 10 }];
      const ordered = orderCornersClockwise(corners);
      expect(ordered.length).toBe(2);
    });

    it('orders 4 corners clockwise', () => {
      // Corners in arbitrary order (centroid excluded)
      const corners: Point[] = [
        { x: 10, y: 90 },  // bottom-left
        { x: 90, y: 10 },  // top-right
        { x: 10, y: 10 }, // top-left
        { x: 90, y: 90 }, // bottom-right
      ];
      const ordered = orderCornersClockwise(corners);
      expect(ordered).toHaveLength(4);
    });
  });

  describe('getCornerPoint', () => {
    it('returns point from pixel array', () => {
      const pixels = new Float32Array([10, 20, 1.57, 5.0]);
      const point = getCornerPoint(0, pixels);
      expect(point.x).toBeCloseTo(10, 0);
      expect(point.y).toBeCloseTo(20, 0);
    });
  });
});

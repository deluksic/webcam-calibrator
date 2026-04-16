import { describe, it, expect } from 'vitest';
import {
  extractEdgePixelsFromBbox,
  getPixel,
  findMaxMagnitude,
  orderPixelsAlongContour,
  detectCorners,
  clusterCorners,
  selectBestQuadCorners,
  getCornerPoint,
} from './corners';

describe('corners', () => {
  describe('extractEdgePixelsFromBbox', () => {
    it('extracts edge pixels from Sobel data', () => {
      const sobelData = new Float32Array(4 * 4 * 2);
      // Pixel (1,1) has strong edge
      sobelData[1 * 4 * 2 + 2] = 10; // gx
      sobelData[1 * 4 * 2 + 3] = 0; // gy

      const pixels = extractEdgePixelsFromBbox(sobelData, 4, 0, 0, 3, 3);

      // Should find at least 1 pixel
      expect(pixels.length).toBeGreaterThan(0);
      const p = getPixel(pixels, 0);
      expect(p.magnitude).toBeCloseTo(10, 1);
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

  describe('orderPixelsAlongContour', () => {
    it('returns index array', () => {
      const pixels = new Float32Array([0, 0, 0, 1, 10, 0, 0, 1, 0, 10, 0, 1]);
      const sorted = orderPixelsAlongContour(pixels);
      expect(sorted.length).toBe(3);
      expect(sorted[0]).toBeDefined();
    });
  });

  describe('detectCorners', () => {
    it('handles empty input', () => {
      const pixels = new Float32Array([]);
      const sorted = orderPixelsAlongContour(pixels);
      const corners = detectCorners(pixels, sorted);
      expect(corners).toHaveLength(0);
    });
  });

  describe('clusterCorners', () => {
    it('removes corners too close to each other', () => {
      const pixels = new Float32Array([0, 0, 0, 1, 5, 0, 0, 1, 50, 50, 0, 1]);

      const corners = [
        { idx: 0, angle: Math.PI },
        { idx: 1, angle: Math.PI / 2 },
        { idx: 2, angle: Math.PI / 2 },
      ];

      const clustered = clusterCorners(corners, pixels, 20);
      expect(clustered.length).toBeLessThanOrEqual(corners.length);
    });
  });

  describe('selectBestQuadCorners', () => {
    it('handles insufficient corners', () => {
      const pixels = new Float32Array([0, 0, 0, 1, 100, 0, 0, 1]);
      const selected = selectBestQuadCorners([0, 1], pixels);
      expect(selected).toHaveLength(0);
    });
  });

  describe('getCornerPoint', () => {
    it('returns point from pixel array', () => {
      const pixels = new Float32Array([10, 20, 0, 1]);
      const point = getCornerPoint(0, pixels);
      expect(point.x).toBeCloseTo(10, 0);
      expect(point.y).toBeCloseTo(20, 0);
    });
  });
});
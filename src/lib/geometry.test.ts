import { describe, it, expect } from 'vitest';
import {
  lineFromPoints,
  lineIntersection,
  pointLineDistance,
  fitLine,
  subdivideSegment,
  quadAspectRatio,
} from './geometry';

describe('geometry', () => {
  describe('lineFromPoints', () => {
    it('computes line from two points', () => {
      const line = lineFromPoints({ x: 0, y: 0 }, { x: 1, y: 1 });
      expect(line).not.toBeNull();
      // Line y = -x should have a = -b, c = 0
      expect(line!.a).toBeCloseTo(-0.707, 3);
      expect(line!.b).toBeCloseTo(0.707, 3);
    });

    it('returns null for coincident points', () => {
      const line = lineFromPoints({ x: 1, y: 1 }, { x: 1, y: 1 });
      expect(line).toBeNull();
    });
  });

  describe('lineIntersection', () => {
    it('intersects two non-parallel lines', () => {
      // Horizontal line y = 0 and vertical line x = 0
      const l1 = { a: 0, b: 1, c: 0 }; // y = 0
      const l2 = { a: 1, b: 0, c: 0 }; // x = 0
      const intersection = lineIntersection(l1, l2);
      expect(intersection).not.toBeNull();
      expect(intersection!.x).toBeCloseTo(0, 5);
      expect(intersection!.y).toBeCloseTo(0, 5);
    });

    it('returns null for parallel lines', () => {
      const l1 = { a: 1, b: 0, c: 0 }; // x = 0
      const l2 = { a: 2, b: 0, c: 1 }; // 2x + 1 = 0 → x = -0.5 (parallel!)
      const intersection = lineIntersection(l1, l2);
      expect(intersection).toBeNull();
    });
  });

  describe('pointLineDistance', () => {
    it('computes perpendicular distance', () => {
      const line = { a: 0, b: 1, c: 0 }; // y = 0
      const dist = pointLineDistance({ x: 3, y: 4 }, line);
      expect(dist).toBeCloseTo(4, 5);
    });
  });

  describe('fitLine', () => {
    it('fits line to collinear points', () => {
      const points = [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
        { x: 2, y: 2 },
        { x: 3, y: 3 },
      ];
      const line = fitLine(points);
      expect(line).not.toBeNull();
      // Line should pass through origin
      expect(line!.c).toBeCloseTo(0, 5);
    });

    it('returns null for insufficient points', () => {
      const points = [{ x: 0, y: 0 }];
      expect(fitLine(points)).toBeNull();
    });
  });

  describe('subdivideSegment', () => {
    it('subdivides into equal segments', () => {
      const points = subdivideSegment({ x: 0, y: 0 }, { x: 6, y: 0 }, 6);
      expect(points).toHaveLength(7);
      expect(points[1]).toEqual({ x: 1, y: 0 });
      expect(points[3]).toEqual({ x: 3, y: 0 });
      expect(points[6]).toEqual({ x: 6, y: 0 });
    });

    it('returns endpoints when divisions = 1', () => {
      const points = subdivideSegment({ x: 0, y: 0 }, { x: 5, y: 5 }, 1);
      expect(points).toHaveLength(2);
      expect(points[0]).toEqual({ x: 0, y: 0 });
      expect(points[1]).toEqual({ x: 5, y: 5 });
    });
  });

  describe('quadAspectRatio', () => {
    it('computes aspect ratio of square', () => {
      const corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
      ];
      expect(quadAspectRatio(corners)).toBeCloseTo(1, 2);
    });

    it('returns 0 for non-quad', () => {
      expect(quadAspectRatio([{ x: 0, y: 0 }])).toBe(0);
    });
  });
});
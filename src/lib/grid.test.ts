import { describe, it, expect } from 'vitest';
import { buildTagGrid, decodeCell, GridCell } from './grid';
import { Point } from './geometry';

describe('grid', () => {
  describe('buildTagGrid', () => {
    it('builds 6x6 grid from square corners', () => {
      const corners: [Point, Point, Point, Point] = [
        { x: 0, y: 0 },   // TL
        { x: 100, y: 0 }, // TR
        { x: 100, y: 100 }, // BR
        { x: 0, y: 100 },  // BL
      ];

      const grid = buildTagGrid(corners, 6);

      expect(grid.outerCorners).toEqual(corners);
      expect(grid.cells.length).toBe(36); // 6x6
      expect(grid.innerCorners.length).toBe(49); // 7x7
    });

    it('builds grid from perspective quad', () => {
      // Simulate perspective view of square
      const corners: [Point, Point, Point, Point] = [
        { x: 10, y: 0 },   // TL (shifted left)
        { x: 90, y: 0 },   // TR (shifted right)
        { x: 100, y: 100 }, // BR
        { x: 0, y: 100 },  // BL
      ];

      const grid = buildTagGrid(corners, 6);

      expect(grid.cells.length).toBe(36);
      // Inner corners should form a grid even with perspective
      expect(grid.innerCorners.length).toBe(49);
    });

    it('has correct number of cells', () => {
      const corners: [Point, Point, Point, Point] = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
      ];

      const grid = buildTagGrid(corners, 6);

      // Should have 36 cells (6x6)
      expect(grid.cells.length).toBe(36);
    });
  });

  describe('decodeCell', () => {
    it('returns -1 for insufficient samples', () => {
      const samples = [{ mag: 1, tangent: 0 }];
      expect(decodeCell(samples)).toBe(-1);
    });

    it('returns -1 for low magnitude samples (solid interior)', () => {
      const samples = [
        { mag: 0.01, tangent: 0 },
        { mag: 0.02, tangent: 0.1 },
        { mag: 0.01, tangent: 0.2 },
        { mag: 0.02, tangent: 0.3 },
      ];
      expect(decodeCell(samples)).toBe(-1);
    });

    it('returns 0 or 1 for strong edge samples', () => {
      // Strong vertical edge
      const samples = [
        { mag: 1, tangent: 0 },
        { mag: 1, tangent: 0.05 },
        { mag: 1, tangent: -0.05 },
        { mag: 1, tangent: 0.1 },
      ];
      const result = decodeCell(samples);
      expect(result).toBeGreaterThanOrEqual(0);
    });
  });

  describe('grid cell access', () => {
    it('cells are indexed row by row', () => {
      const corners: [Point, Point, Point, Point] = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
      ];

      const grid = buildTagGrid(corners, 6);

      // First row, first column
      expect(grid.cells[0].row).toBe(0);
      expect(grid.cells[0].col).toBe(0);

      // First row, second column
      expect(grid.cells[1].row).toBe(0);
      expect(grid.cells[1].col).toBe(1);

      // Second row, first column
      expect(grid.cells[6].row).toBe(1);
      expect(grid.cells[6].col).toBe(0);
    });
  });
});
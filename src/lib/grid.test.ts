import { describe, it, expect } from 'vitest';
import {
  buildDecodeEdgeMask,
  buildTagGrid,
  decodeCell,
  fillUnknownNeighbors6,
  type CellSobelSample,
  type GridCell,
} from './grid';
import type { TagPattern } from './tag36h11';
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
    /** Axis-aligned cell quad: x = 100u, y = 100v (TL,TR,BR,BL). */
    const axisCell: GridCell = {
      row: 0,
      col: 0,
      corners: [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
      ],
      center: { x: 50, y: 50 },
    };

    const base = (over: Partial<CellSobelSample>): CellSobelSample => ({
      mag: 1,
      tangent: 0,
      gx: 0,
      gy: 0,
      u: 0.5,
      v: 0.5,
      ...over,
    });

    it('returns -1 for insufficient samples', () => {
      const samples: CellSobelSample[] = [base({ mag: 1 })];
      expect(decodeCell(axisCell, samples)).toBe(-1);
    });

    it('returns -1 for low magnitude samples (solid interior)', () => {
      const samples: CellSobelSample[] = Array.from({ length: 11 }, (_, i) =>
        base({ mag: 0.001, tangent: i * 0.1, u: 0.2 + i * 0.06, v: 0.5 }),
      );
      expect(decodeCell(axisCell, samples)).toBe(-1);
    });

    it('votes black when UV gradient aligns with outward radial (dark interior)', () => {
      const samples: CellSobelSample[] = Array.from({ length: 11 }, (_, i) =>
        base({
          mag: 1,
          gx: 1,
          gy: 0,
          u: 0.65 + i * 0.02,
          v: 0.5,
        }),
      );
      expect(decodeCell(axisCell, samples)).toBe(1);
    });

    it('votes white when UV gradient opposes outward radial', () => {
      const samples: CellSobelSample[] = Array.from({ length: 11 }, (_, i) =>
        base({
          mag: 1,
          gx: -1,
          gy: 0,
          u: 0.65 + i * 0.02,
          v: 0.5,
        }),
      );
      expect(decodeCell(axisCell, samples)).toBe(0);
    });

    it('returns -2 when pos/neg tie with enough votes', () => {
      const samples: CellSobelSample[] = [
        ...Array.from({ length: 5 }, (_, i) =>
          base({ mag: 1, gx: 1, gy: 0, u: 0.65 + i * 0.02, v: 0.5 }),
        ),
        ...Array.from({ length: 5 }, (_, i) =>
          base({ mag: 1, gx: -1, gy: 0, u: 0.65 + i * 0.02, v: 0.5 }),
        ),
      ];
      expect(decodeCell(axisCell, samples)).toBe(-2);
    });
  });

  describe('fillUnknownNeighbors6', () => {
    it('fills -1 from unanimous neighbors but leaves -2 unchanged', () => {
      const weak = Array(36).fill(0) as unknown as TagPattern;
      // Cell (2,2) idx 14; four cardinals all 1
      weak[8] = weak[20] = weak[13] = weak[15] = 1;
      weak[14] = -1;
      fillUnknownNeighbors6(weak);
      expect(weak[14]).toBe(1);

      const tie = Array(36).fill(0) as unknown as TagPattern;
      tie[8] = tie[20] = tie[13] = tie[15] = 1;
      tie[14] = -2;
      fillUnknownNeighbors6(tie);
      expect(tie[14]).toBe(-2);
    });
  });

  describe('buildDecodeEdgeMask', () => {
    it('marks only matching label with non-zero Sobel', () => {
      const w = 8;
      const h = 4;
      const labelData = new Uint32Array(w * h).fill(2);
      labelData[10] = 7;
      labelData[11] = 7;
      const sobel = new Float32Array(w * h * 2);
      sobel[10 * 2] = 0.1;
      sobel[10 * 2 + 1] = 0;
      sobel[11 * 2] = 0;
      sobel[11 * 2 + 1] = 0;
      const mask = buildDecodeEdgeMask(labelData, sobel, w, h, 7, 1, 1, 4, 2, 0);
      expect(mask[10]).toBe(1);
      expect(mask[11]).toBe(0);
      expect(mask[0]).toBe(0);
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
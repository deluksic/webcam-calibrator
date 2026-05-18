import { describe, it, expect } from 'vitest'

import type { Corners } from '@/lib/geometry'
import { buildTagGrid } from '@/lib/grid'

describe('grid', () => {
  describe('buildTagGrid', () => {
    it('builds 6x6 grid from square corners', () => {
      const corners: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]

      const grid = buildTagGrid(corners, 6)

      expect(grid.outerCorners).toEqual(corners)
      expect(grid.cells.length).toBe(36)
      expect(grid.innerCorners.length).toBe(49)
    })

    it('builds grid from perspective quad', () => {
      const corners: Corners = [
        { x: 10, y: 0 },
        { x: 90, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]

      const grid = buildTagGrid(corners, 6)

      expect(grid.cells.length).toBe(36)
      expect(grid.innerCorners.length).toBe(49)
    })

    it('cells are indexed row by row', () => {
      const corners: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]

      const grid = buildTagGrid(corners, 6)

      expect(grid.cells[0]!.row).toBe(0)
      expect(grid.cells[0]!.col).toBe(0)
      expect(grid.cells[1]!.row).toBe(0)
      expect(grid.cells[1]!.col).toBe(1)
      expect(grid.cells[6]!.row).toBe(1)
      expect(grid.cells[6]!.col).toBe(0)
    })
  })
})

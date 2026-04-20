// AprilTag grid generator for calibration targets
// Generates SVG with configurable NxM tag grid, spacing, and checkerboard
// Uses real tag36h11 dictionary from AprilRobotics/apriltag

import { codeToPattern, TAG36H11_CODES, TAG36H11_COUNT } from '@/lib/tag36h11'

export interface TagGridOptions {
  cols: number // Number of tags horizontally
  rows: number // Number of tags vertically
  tagSize: number // Tag side length in mm (or any unit)
  spacing: number // Spacing between tags, as multiple of tagSize (e.g., 1.5)
  checkerboard: boolean // Include checkerboard squares between tags
  margin: number // Margin around grid, as multiple of tagSize
  /** Array of tag IDs to display (one per grid cell). If not provided, uses indices. */
  tagIds?: number[]
}

/** Select N unique random tag IDs from the dictionary. */
export function selectRandomTags(count: number, seed?: number): number[] {
  // Simple deterministic shuffle using seed
  const rng = seed !== undefined ? seededRng(seed) : Math.random
  const indices = Array.from({ length: TAG36H11_COUNT }, (_, i) => i)
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  return indices.slice(0, count)
}

function seededRng(seed: number) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

const DEFAULT_OPTIONS: TagGridOptions = {
  cols: 4,
  rows: 3,
  tagSize: 40,
  spacing: 1.5,
  checkerboard: true,
  margin: 1,
}

/**
 * Generate a single AprilTag cell pattern as 6x6 SVG rects.
 * Uses real tag36h11 dictionary encoding via codeToPattern.
 * White background with black tag cells.
 */
function generateTagSVG(tagSize: number, offsetX: number, offsetY: number, pattern: (0 | 1 | -1)[]): string {
  const cellSize = tagSize / 8
  let svg = ''

  for (let my = 0; my < 8; my++) {
    for (let mx = 0; mx < 8; mx++) {
      const x = offsetX + mx * cellSize
      const y = offsetY + my * cellSize
      const border = mx === 0 || mx === 7 || my === 0 || my === 7
      if (border) {
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="black"/>\n`
        continue
      }
      const value = pattern[(my - 1) * 6 + (mx - 1)]
      if (value === 1) {
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="black"/>\n`
      }
    }
  }

  return svg
}

/**
 * Generate checkerboard squares at intersections of 4 tags.
 * Only draws squares at points where both a horizontal gap AND vertical gap exist.
 * For NxM tags: (N-1) × (M-1) squares at the intersections.
 */
function generateCheckerboard(
  tagSize: number,
  spacing: number,
  cols: number,
  rows: number,
  startX: number,
  startY: number,
): string {
  const gapSize = (spacing - 1) * tagSize
  const cols_1 = cols - 1
  const rows_1 = rows - 1

  let svg = ''

  // Squares only at intersections of horizontal and vertical gaps
  for (let r = 0; r < rows_1; r++) {
    for (let c = 0; c < cols_1; c++) {
      // Intersection of horizontal gap (row r) and vertical gap (col c)
      const x = startX + (c + 1) * tagSize + c * gapSize
      const y = startY + (r + 1) * tagSize + r * gapSize
      svg += `  <rect x="${x}" y="${y}" width="${gapSize}" height="${gapSize}" fill="black"/>\n`
    }
  }

  return svg
}

/**
 * Generate complete tag grid SVG.
 */
export function generateTagGridSVG(options: Partial<TagGridOptions> = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options }
  const { cols, rows, tagSize, spacing, checkerboard, margin, tagIds } = opts

  const boardSize = (spacing - 1) * tagSize

  const gridWidth = cols * tagSize + (cols - 1) * boardSize
  const gridHeight = rows * tagSize + (rows - 1) * boardSize
  const marginSize = margin * tagSize

  const svgWidth = gridWidth + 2 * marginSize
  const svgHeight = gridHeight + 2 * marginSize

  // White background with crispEdges to prevent sub-pixel gaps
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" shape-rendering="crispEdges">\n`
  svg += `  <rect width="100%" height="100%" fill="white"/>\n`

  const startX = marginSize
  const startY = marginSize

  // Generate tags
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c
      const tagId = tagIds && tagIds[idx] !== undefined ? tagIds[idx] : idx % TAG36H11_COUNT
      const pattern = codeToPattern(TAG36H11_CODES[tagId])
      const tagX = startX + c * (tagSize + boardSize)
      const tagY = startY + r * (tagSize + boardSize)
      svg += generateTagSVG(tagSize, tagX, tagY, pattern as (0 | 1 | -1)[])
    }
  }

  // Generate checkerboard between tags
  if (checkerboard && cols > 1 && rows > 1) {
    svg += generateCheckerboard(tagSize, spacing, cols, rows, startX, startY)
  }

  svg += '</svg>'
  return svg
}

/**
 * Generate a single AprilTag for standalone use.
 */
export function generateSingleTagSVG(tagSize: number, tagId: number = 0): string {
  const cellSize = tagSize / 8
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${tagSize}" height="${tagSize}" viewBox="0 0 ${tagSize} ${tagSize}" shape-rendering="crispEdges">\n`
  svg += `  <rect width="100%" height="100%" fill="white"/>\n`

  const pattern = codeToPattern(TAG36H11_CODES[tagId % TAG36H11_COUNT])
  for (let my = 0; my < 8; my++) {
    for (let mx = 0; mx < 8; mx++) {
      const x = mx * cellSize
      const y = my * cellSize
      const border = mx === 0 || mx === 7 || my === 0 || my === 7
      if (border) {
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="black"/>\n`
        continue
      }
      if (pattern[(my - 1) * 6 + (mx - 1)] === 1) {
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="black"/>\n`
      }
    }
  }

  svg += '</svg>'
  return svg
}

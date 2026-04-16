// AprilTag grid generator for calibration targets
// Generates SVG with configurable NxM tag grid, spacing, and checkerboard

export interface TagGridOptions {
  cols: number;      // Number of tags horizontally
  rows: number;      // Number of tags vertically
  tagSize: number;   // Tag side length in mm (or any unit)
  spacing: number;   // Spacing between tags, as multiple of tagSize (e.g., 1.5)
  checkerboard: boolean; // Include black checkerboard squares between tags
  margin: number;    // Margin around grid, as multiple of tagSize
}

const DEFAULT_OPTIONS: TagGridOptions = {
  cols: 4,
  rows: 3,
  tagSize: 40,
  spacing: 1.5,
  checkerboard: true,
  margin: 1,
};

/**
 * Generate a single AprilTag cell pattern as 6x6 SVG rects.
 * Border cells (row 0/5, col 0/5) are always black.
 * Interior alternates based on cell parity.
 */
function generateTagPattern(tagSize: number, offsetX: number, offsetY: number): string {
  const cellSize = tagSize / 6;
  let svg = '';

  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      // Border is always black
      if (row === 0 || row === 5 || col === 0 || col === 5) {
        // Black cell - no need to draw, background is black
        continue;
      }

      // Interior: alternate based on cell parity
      const isWhite = (row + col) % 2 === 1;
      if (isWhite) {
        const x = offsetX + col * cellSize;
        const y = offsetY + row * cellSize;
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="white"/>\n`;
      }
    }
  }

  return svg;
}

/**
 * Generate checkerboard pattern between tags.
 * Black squares in alternating positions.
 */
function generateCheckerboard(
  tagSize: number,
  spacing: number,
  cols: number,
  rows: number,
  startX: number,
  startY: number,
): string {
  const cellSize = tagSize / 6;
  const boardCellSize = cellSize; // Same as tag cell for visual consistency
  const boardWidth = (spacing - 1) * tagSize;
  const boardHeight = (spacing - 1) * tagSize;
  const cols_1 = cols - 1;
  const rows_1 = rows - 1;

  let svg = '';

  // Horizontal boards (between rows of tags)
  for (let r = 0; r < rows_1; r++) {
    const y = startY + (r + 1) * tagSize + r * boardHeight;
    for (let c = 0; c < cols; c++) {
      const x = startX + c * tagSize;
      // Draw board - all black
      svg += `  <rect x="${x}" y="${y}" width="${tagSize * cols}" height="${boardHeight}" fill="black"/>\n`;
    }
  }

  // Vertical boards (between columns of tags)
  // Only between columns, not between rows (already covered)
  for (let r = 0; r < rows; r++) {
    const y = startY + r * (tagSize + boardHeight) + (r > 0 ? boardHeight : 0);
    for (let c = 0; c < cols_1; c++) {
      const x = startX + (c + 1) * tagSize + c * boardWidth;
      svg += `  <rect x="${x}" y="${y}" width="${boardWidth}" height="${tagSize}" fill="black"/>\n`;
    }
  }

  return svg;
}

/**
 * Generate complete tag grid SVG.
 */
export function generateTagGridSVG(options: Partial<TagGridOptions> = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const { cols, rows, tagSize, spacing, checkerboard, margin } = opts;

  const cellSize = tagSize / 6;
  const boardSize = (spacing - 1) * tagSize;

  const gridWidth = cols * tagSize + (cols - 1) * boardSize;
  const gridHeight = rows * tagSize + (rows - 1) * boardSize;
  const marginSize = margin * tagSize;

  const svgWidth = gridWidth + 2 * marginSize;
  const svgHeight = gridHeight + 2 * marginSize;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}">\n`;
  svg += `  <rect width="100%" height="100%" fill="black"/>\n`;

  const startX = marginSize;
  const startY = marginSize;

  // Generate tags
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tagX = startX + c * (tagSize + boardSize);
      const tagY = startY + r * (tagSize + boardSize);
      svg += generateTagPattern(tagSize, tagX, tagY);
    }
  }

  // Generate checkerboard between tags
  if (checkerboard && cols > 1 && rows > 1) {
    svg += generateCheckerboard(tagSize, spacing, cols, rows, startX, startY);
  }

  svg += '</svg>';
  return svg;
}

/**
 * Generate a single AprilTag for standalone use.
 */
export function generateSingleTagSVG(tagSize: number): string {
  const cellSize = tagSize / 6;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${tagSize}" height="${tagSize}" viewBox="0 0 ${tagSize} ${tagSize}">\n`;
  svg += `  <rect width="100%" height="100%" fill="black"/>\n`;

  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      if (row === 0 || row === 5 || col === 0 || col === 5) continue;
      if ((row + col) % 2 === 1) {
        const x = col * cellSize;
        const y = row * cellSize;
        svg += `  <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="white"/>\n`;
      }
    }
  }

  svg += '</svg>';
  return svg;
}
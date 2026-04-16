// AprilTag tag36h11 — dictionary + decode from AprilRobotics/apriltag reference
// https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c
// BSD-3-Clause license

import codesRaw from './tag36h11.json';

// Parse JSON strings as BigInt (needed because JSON can't store 64-bit integers)
const TAG36H11_CODES: readonly bigint[] = (codesRaw as unknown as string[]).map(s => BigInt(s));
export { TAG36H11_CODES };
export const TAG36H11_COUNT = TAG36H11_CODES.length;

// LUT: each of 36 bits maps to (x, y) in 1-indexed 10×10 tag coords.
// The 10×10 grid has 8×8 inner data area (coords 1–8). Border = 0 and 9.
// For tag36h11, the data bits (1-indexed 6×6) are at coords 1–6 in both axes.
const BIT_X = [
  1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3,
] as const;
const BIT_Y = [
  1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4,
] as const;

export type TagPattern = (0 | 1 | -1)[];

/**
 * Decode a 36-bit tag36h11 code into a 6×6 binary grid (0/1 values).
 * Border cells (row 0, 5 or col 0, 5) are always 0 (black).
 * Interior cells use the AprilTag spiral-quad encoding via the bit_x/bit_y LUT.
 * Coordinate mapping: 10×10 tag grid, values stored at (bit_y, bit_x),
 * 6×6 pattern extracts rows 1–6, cols 1–6 (1-indexed).
 */
export function codeToPattern(code: bigint): TagPattern {
  const pattern = new Array(36).fill(0) as TagPattern;
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit];
    const y = BIT_Y[bit];
    // 10×10 coords (1-indexed) → 6×6 0-indexed pattern
    const row = y - 1; // 0–5
    const col = x - 1; // 0–5
    pattern[row * 6 + col] = (code >> BigInt(35 - bit)) & 1n ? 1 : 0;
  }
  return pattern;
}

/**
 * Fast popcount for 64-bit BigInt.
 */
function popcount64(x: bigint): number {
  let count = 0;
  while (x) { count++; x &= x - 1n; }
  return count;
}

/**
 * Compute Hamming distance between two 36-bit codes.
 */
export function hammingDistance(code1: bigint, code2: bigint): number {
  return popcount64(code1 ^ code2);
}

/**
 * Convert 6×6 binary pattern (row-major) to 36-bit BigInt.
 * Uses LUT to find code bit positions for each pattern cell.
 * This is the inverse of codeToPattern.
 */
export function patternToCode(pattern: TagPattern): bigint {
  if (pattern.length !== 36) return -1 as unknown as bigint;
  let code = 0n;
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit];
    const y = BIT_Y[bit];
    const row = y - 1;
    const col = x - 1;
    if (pattern[row * 6 + col] === 1) {
      code |= (1n << BigInt(35 - bit));
    }
  }
  return code;
}

/**
 * Match detected pattern against tag36h11 dictionary.
 * Returns tag ID if found within error threshold, or -1 if no match.
 *
 * @param pattern 6×6 binary pattern (36 bits)
 * @param maxError Maximum Hamming distance to consider a match
 * @returns Tag family ID (0–586) or -1
 */
export function decodeTag36h11(
  pattern: TagPattern | null,
  maxError: number = 5,
): number {
  if (!pattern || pattern.length !== 36) return -1;

  const unknownCount = pattern.filter(v => v === -1).length;
  if (unknownCount > maxError * 2) return -1; // Too many unknown cells

  // Build 36-bit code from pattern, tracking which bits are known
  let code = 0n;
  let knownMask = 0n;
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit];
    const y = BIT_Y[bit];
    // Map to 6×6 pattern position
    const pRow = y - 1;
    const pCol = x - 1;
    if (pRow < 0 || pRow > 5 || pCol < 0 || pCol > 5) continue; // border cell
    const pIdx = pRow * 6 + pCol;
    const val = pattern[pIdx];
    if (val === 1) {
      code |= (1n << BigInt(35 - bit));
      knownMask |= (1n << BigInt(35 - bit));
    } else if (val === 0) {
      knownMask |= (1n << BigInt(35 - bit));
    }
    // val === -1 → unknown, skip
  }

  // Find best match in dictionary
  let bestId = -1;
  let bestDist = maxError + 1;

  for (let id = 0; id < TAG36H11_COUNT; id++) {
    const dictCode = TAG36H11_CODES[id];
    // Hamming distance considering only known bits
    const diff = (code ^ dictCode) & knownMask;
    const dist = popcount64(diff);

    if (dist < bestDist) {
      bestDist = dist;
      bestId = id;
      if (bestDist === 0) break;
    }
  }

  return bestId;
}

/**
 * Rotate a 6×6 pattern 90 degrees clockwise.
 */
export function rotatePattern(pattern: TagPattern): TagPattern {
  const result: TagPattern = new Array(36) as TagPattern;
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const srcIdx = row * 6 + col;
      const dstRow = col;
      const dstCol = 5 - row;
      const dstIdx = dstRow * 6 + dstCol;
      result[dstIdx] = pattern[srcIdx];
    }
  }
  return result;
}

/**
 * Decode tag with all 4 rotations, return best match.
 */
export function decodeTag36h11AnyRotation(
  pattern: TagPattern | null,
  maxError: number = 5,
): { id: number; rotation: number } | null {
  if (!pattern) return null;

  let currentPattern = [...pattern];
  for (let r = 0; r < 4; r++) {
    const id = decodeTag36h11(currentPattern, maxError);
    if (id !== -1) {
      return { id, rotation: r };
    }
    currentPattern = rotatePattern(currentPattern);
  }

  return null;
}

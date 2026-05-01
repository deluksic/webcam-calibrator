// AprilTag tag36h11 — dictionary + decode from AprilRobotics/apriltag reference
// https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c
// BSD-3-Clause license

import codesRaw from '@/lib/tag36h11.json'

// Parse JSON strings as BigInt (needed because JSON can't store 64-bit integers)
const TAG36H11_CODES: bigint[] = (codesRaw as unknown as string[]).map((s) => BigInt(s))
export { TAG36H11_CODES }
export const TAG36H11_COUNT = TAG36H11_CODES.length

// LUT: each of 36 bits maps to (x, y) in 1-indexed 10×10 tag coords.
// The 10×10 grid has 8×8 inner data area (coords 1–8). Border = 0 and 9.
// For tag36h11, the data bits (1-indexed 6×6) are at coords 1–6 in both axes.
const BIT_X = [
  1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3,
] as const
const BIT_Y = [
  1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4,
] as const

// Precompute base code patterns (no rotations) for wildcard search
const BASE_CODE_PATTERNS: bigint[] = TAG36H11_CODES

/** `-1` = no / weak directional signal; `-2` = enough votes but pos/neg tie (ambiguous). */
export type TagPattern = (0 | 1 | -1 | -2)[]

/**
 * Decode a 36-bit tag36h11 code into a 6×6 binary grid (0/1 values).
 * Border cells (row 0, 5 or col 0, 5) are always 0 (black).
 * Interior cells use the AprilTag spiral-quad encoding via the bit_x/bit_y LUT.
 * Coordinate mapping: 10×10 tag grid, values stored at (bit_y, bit_x),
 * 6×6 pattern extracts rows 1–6, cols 1–6 (1-indexed).
 */
export function tag36h11Code(id: number): bigint {
  const code = TAG36H11_CODES[id]
  if (code === undefined) {
    throw new RangeError(`tag36h11 id ${id} out of range [0, ${TAG36H11_COUNT})`)
  }
  return code
}

export function codeToPattern(code: bigint): TagPattern {
  const pattern: TagPattern = Array.from({ length: 36 }, () => 0)
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    // 10×10 coords (1-indexed) → 6×6 0-indexed pattern
    const row = y - 1 // 0–5
    const col = x - 1 // 0–5
    pattern[row * 6 + col] = (code >> BigInt(35 - bit)) & 1n ? 1 : 0
  }
  return pattern
}

/** Row-major 6×6 0/1 interior pattern for tag36h11 `id` (border is implicit). */
export function tagIdPattern(id: number): TagPattern {
  return codeToPattern(tag36h11Code(id))
}

/**
 * Fast popcount for 64-bit BigInt.
 */
function popcount64(x: bigint): number {
  let count = 0
  while (x) {
    count++
    x &= x - 1n
  }
  return count
}

/**
 * Compute Hamming distance between two 36-bit codes.
 */
export function hammingDistance(code1: bigint, code2: bigint): number {
  return popcount64(code1 ^ code2)
}

/**
 * Convert 6×6 binary pattern (row-major) to 36-bit BigInt.
 * Uses LUT to find code bit positions for each pattern cell.
 * This is the inverse of codeToPattern.
 */
export function patternToCode(pattern: TagPattern): bigint {
  if (pattern.length !== 36) {
    return -1n
  }
  let code = 0n
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    const row = y - 1
    const col = x - 1
    if (pattern[row * 6 + col] === 1) {
      code |= 1n << BigInt(35 - bit)
    }
  }
  return code
}

export interface DictionaryMatch {
  id: number
  /** Hamming distance on known bits; `maxError + 1` if no codeword within `maxError`. */
  dist: number
}

/**
 * Best tag36h11 codeword for this pattern (known bits only).
 * @returns `id === -1` when no tag has `dist <= maxError`.
 */
export function decodeTag36h11Best(pattern: TagPattern | undefined, maxError: number = 5): DictionaryMatch {
  if (!pattern || pattern.length !== 36) {
    return { id: -1, dist: maxError + 1 }
  }

  const unknownCount = pattern.filter((v) => v === -1 || v === -2).length
  if (unknownCount > maxError * 2) {
    return { id: -1, dist: maxError + 1 }
  }

  // Collect positions of unknown bits (-1) for wildcard optimization
  const unknownPositions: number[] = []
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    const pRow = y - 1
    const pCol = x - 1
    if (pRow >= 0 && pRow <= 5 && pCol >= 0 && pCol <= 5) {
      const pIdx = pRow * 6 + pCol
      if (pattern[pIdx] === -1) {
        unknownPositions.push(bit)
      }
    }
  }

  const u = unknownPositions.length

  // Generate all 2^U combinations of unknown bits
  function generateMasks(pos: number, mask: bigint): void {
    if (pos >= u) {
      wildcardMasks.push(mask)
      return
    }
    const bitIndex = unknownPositions[pos]!
    generateMasks(pos + 1, mask) // 0
    generateMasks(pos + 1, mask | (1n << BigInt(bitIndex))) // 1
  }

  const wildcardMasks: bigint[] = []
  generateMasks(0, 0n)

  // Pre-compute pattern bits that are known (0 or 1)
  let patternKnown = 0n
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    const pRow = y - 1
    const pCol = x - 1
    if (pRow < 0 || pRow > 5 || pCol < 0 || pCol > 5) {
      continue
    }
    const pIdx = pRow * 6 + pCol
    if (pattern[pIdx] === 1) {
      patternKnown |= 1n << BigInt(35 - bit)
    }
  }

  let bestId = -1
  let bestDist = maxError + 1

  for (const baseCode of BASE_CODE_PATTERNS) {
    for (const wildMask of wildcardMasks) {
      const diff = (patternKnown ^ (baseCode ^ wildMask)) & 0xffffffffn
      const dist = popcount64(diff)
      if (dist < bestDist) {
        bestDist = dist
        bestId = TAG36H11_CODES.indexOf(baseCode)
        if (bestDist === 0) {
          break
        }
      }
    }
    if (bestDist === 0) {
      break
    }
  }

  return bestDist > maxError ? { id: -1, dist: bestDist } : { id: bestId, dist: bestDist }
}

/**
 * Match detected pattern against tag36h11 dictionary.
 * Returns tag ID if found within error threshold, or -1 if no match.
 */
export function decodeTag36h11(pattern: TagPattern | undefined, maxError: number = 5): number {
  return decodeTag36h11Best(pattern, maxError).id
}

/**
 * Rotate a 6×6 pattern 90 degrees clockwise.
 */
export function rotatePattern(pattern: TagPattern): TagPattern {
  const result = [] as TagPattern
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const srcIdx = row * 6 + col
      const dstRow = col
      const dstCol = 5 - row
      const dstIdx = dstRow * 6 + dstCol
      result[dstIdx] = pattern[srcIdx]!
    }
  }
  return result
}

/**
 * Decode tag across four 90° CW rotations — picks the rotation with **lowest Hamming** `dist`
 * (then lowest `id` on ties) among rotations with `dist <= maxError`.
 */
export function decodeTag36h11AnyRotation(
  pattern: TagPattern | undefined,
  maxError: number,
): { id: number; rotation: number } | undefined {
  if (!pattern) {
    return undefined
  }

  let best: { id: number; rotation: number; dist: number } | undefined = undefined
  let currentPattern = [...pattern]
  for (let r = 0; r < 4; r++) {
    const { id, dist } = decodeTag36h11Best(currentPattern, maxError)
    if (id !== -1) {
      if (!best || dist < best.dist || (dist === best.dist && id < best.id)) {
        best = { id, rotation: r, dist }
      }
    }
    currentPattern = rotatePattern(currentPattern)
  }

  return best ? { id: best.id, rotation: best.rotation } : undefined
}

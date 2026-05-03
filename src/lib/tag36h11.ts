// AprilTag tag36h11 — dictionary + decode from AprilRobotics/apriltag reference
// https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c
// BSD-3-Clause license

import codesRaw from '@/lib/tag36h11.json'

import {
  TAG_MODULE_CELL,
  type TagPattern,
  patternIsFullyBinary,
} from '@/lib/tagModuleCell'

export type { TagPattern } from '@/lib/tagModuleCell'

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
  const pattern: TagPattern = Array.from({ length: 36 }, () => TAG_MODULE_CELL.black)
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    // 10×10 coords (1-indexed) → 6×6 0-indexed pattern
    const row = y - 1 // 0–5
    const col = x - 1 // 0–5
    pattern[row * 6 + col] = (code >> BigInt(35 - bit)) & 1n ? TAG_MODULE_CELL.white : TAG_MODULE_CELL.black
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
    if (pattern[row * 6 + col] === TAG_MODULE_CELL.white) {
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

function collectWildcardUnknownPositions(pattern: TagPattern): number[] {
  const unknownPositions: number[] = []
  for (let bit = 0; bit < 36; bit++) {
    const x = BIT_X[bit]!
    const y = BIT_Y[bit]!
    const pRow = y - 1
    const pCol = x - 1
    if (pRow >= 0 && pRow <= 5 && pCol >= 0 && pCol <= 5) {
      const pIdx = pRow * 6 + pCol
      if (pattern[pIdx] === TAG_MODULE_CELL.weak) {
        unknownPositions.push(bit)
      }
    }
  }
  return unknownPositions
}

function generateWildcardMasks(u: number, unknownPositions: number[]): bigint[] {
  const wildcardMasks: bigint[] = []
  function generateMasks(pos: number, mask: bigint): void {
    if (pos >= u) {
      wildcardMasks.push(mask)
      return
    }
    const bitIndex = unknownPositions[pos]!
    generateMasks(pos + 1, mask)
    generateMasks(pos + 1, mask | (1n << BigInt(bitIndex)))
  }
  generateMasks(0, 0n)
  return wildcardMasks
}

function patternKnownBits(pattern: TagPattern): bigint {
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
    if (pattern[pIdx] === TAG_MODULE_CELL.white) {
      patternKnown |= 1n << BigInt(35 - bit)
    }
  }
  return patternKnown
}

/**
 * Best Hamming match of `pattern` to a single 36-bit codeword (wildcard expansion on weak cells only).
 */
export function decodeBestAgainstCodeword(
  pattern: TagPattern | undefined,
  codeword: bigint,
  maxError: number,
): { dist: number } {
  if (!pattern || pattern.length !== 36) {
    return { dist: maxError + 1 }
  }

  // Cap ambiguous cells before 2^U wildcard expansion. `2 * maxError` matches historical behavior
  // (noise often yields 4–6 weak cells while Hamming match still succeeds within `maxError`).
  const unknownCount = pattern.filter((v) => v === TAG_MODULE_CELL.weak || v === TAG_MODULE_CELL.tie).length
  if (unknownCount > maxError * 2) {
    return { dist: maxError + 1 }
  }

  const unknownPositions = collectWildcardUnknownPositions(pattern)
  const u = unknownPositions.length
  const wildcardMasks = generateWildcardMasks(u, unknownPositions)
  const patternKnown = patternKnownBits(pattern)

  let bestDist = maxError + 1
  for (const wildMask of wildcardMasks) {
    const diff = (patternKnown ^ (codeword ^ wildMask)) & 0xffffffffn
    const dist = popcount64(diff)
    if (dist < bestDist) {
      bestDist = dist
      if (bestDist === 0) {
        break
      }
    }
  }

  return { dist: bestDist }
}

/**
 * Best tag36h11 codeword for this pattern (known bits only).
 * @returns `id === -1` when no tag has `dist <= maxError`.
 */
export function decodeTag36h11Best(pattern: TagPattern | undefined, maxError: number = 5): DictionaryMatch {
  if (!pattern || pattern.length !== 36) {
    return { id: -1, dist: maxError + 1 }
  }

  const unknownCount = pattern.filter((v) => v === TAG_MODULE_CELL.weak || v === TAG_MODULE_CELL.tie).length
  if (unknownCount > maxError * 2) {
    return { id: -1, dist: maxError + 1 }
  }

  const unknownPositions = collectWildcardUnknownPositions(pattern)
  const u = unknownPositions.length
  const wildcardMasks = generateWildcardMasks(u, unknownPositions)
  const patternKnown = patternKnownBits(pattern)

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
 * Encode canonical 36-bit payload as fixed-length base64url (no padding).
 */
export function canonicalCodeToLabel(code: bigint): string {
  let x = code
  const bytes = new Uint8Array(5)
  for (let i = 4; i >= 0; i--) {
    bytes[i] = Number(x & 0xffn)
    x >>= 8n
  }
  let bin = ''
  for (let i = 0; i < 5; i++) {
    bin += String.fromCharCode(bytes[i]!)
  }
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Inverse of {@link canonicalCodeToLabel} for valid custom payloads; returns undefined if invalid. */
export function labelToCanonicalCode(label: string): bigint | undefined {
  const pad = 4 - (label.length % 4 || 4)
  const b64 = label.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(pad === 4 ? 0 : pad)
  try {
    const bin = atob(b64)
    if (bin.length !== 5) {
      return undefined
    }
    let code = 0n
    for (let i = 0; i < 5; i++) {
      code = (code << 8n) | BigInt(bin.charCodeAt(i)!)
    }
    return code & ((1n << 36n) - 1n)
  } catch {
    return undefined
  }
}

/** True for custom tags: negative ids encode canonical payload as `tagId = -1 - Number(code)`. */
export function isCustomTagId(tagId: number): boolean {
  return tagId < 0
}

export function customTagIdFromCanonicalCode(code: bigint): number {
  const n = Number(code)
  if (!Number.isSafeInteger(n) || n < 0 || n >= 2 ** 36) {
    throw new RangeError(`canonical code must be in [0, 2^36), got ${code}`)
  }
  return -1 - n
}

export function canonicalCodeFromCustomTagId(tagId: number): bigint | undefined {
  if (tagId >= 0) {
    return undefined
  }
  return BigInt(-tagId - 1)
}

/** 6×6 interior pattern: tag36h11 index `0..TAG36H11_COUNT-1`, or custom negative id. */
export function patternForResolvedTagId(tagId: number): TagPattern {
  if (tagId < 0) {
    const code = canonicalCodeFromCustomTagId(tagId)
    if (code === undefined) {
      throw new RangeError(`invalid custom tag id ${tagId}`)
    }
    return codeToPattern(code)
  }
  return codeToPattern(tag36h11Code(tagId))
}

export function displayLabelForTagId(tagId: number): string {
  if (tagId >= 0) {
    return String(tagId)
  }
  const code = canonicalCodeFromCustomTagId(tagId)
  return code !== undefined ? canonicalCodeToLabel(code) : String(tagId)
}

/**
 * Among four rotations, pick minimum `patternToCode` as unsigned 36-bit; tie-break lower rotation index.
 */
export function canonicalizeBinaryPatternMinCode(pattern: TagPattern): { code: bigint; rotation: number } | undefined {
  if (!patternIsFullyBinary(pattern)) {
    return undefined
  }
  let bestCode: bigint | undefined
  let bestRot = 0
  let current: TagPattern = [...pattern]
  for (let r = 0; r < 4; r++) {
    const code = patternToCode(current)
    if (bestCode === undefined || code < bestCode || (code === bestCode && r < bestRot)) {
      bestCode = code
      bestRot = r
    }
    current = rotatePattern(current)
  }
  return { code: bestCode!, rotation: bestRot }
}

/**
 * Decode observed pattern against session custom codewords (canonical bigints), four rotations.
 */
export function matchCustomCodewordsAnyRotation(
  pattern: TagPattern | undefined,
  codewords: bigint[],
  maxError: number,
): { tagId: number; rotation: number } | undefined {
  if (!pattern || codewords.length < 1) {
    return undefined
  }

  let best: { tagId: number; rotation: number; dist: number } | undefined
  let currentPattern = [...pattern]
  for (let r = 0; r < 4; r++) {
    for (const cw of codewords) {
      const { dist } = decodeBestAgainstCodeword(currentPattern, cw, maxError)
      if (dist <= maxError) {
        const tagId = customTagIdFromCanonicalCode(cw)
        if (
          !best ||
          dist < best.dist ||
          (dist === best.dist && tagId < best.tagId) ||
          (dist === best.dist && tagId === best.tagId && r < best.rotation)
        ) {
          best = { tagId, rotation: r, dist }
        }
      }
    }
    currentPattern = rotatePattern(currentPattern)
  }

  return best ? { tagId: best.tagId, rotation: best.rotation } : undefined
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

import { describe, it, expect } from 'vitest'

import {
  hammingDistance,
  patternToCode,
  codeToPattern,
  decodeTag36h11,
  rotatePattern,
  decodeTag36h11AnyRotation,
  tag36h11Code,
  TAG36H11_COUNT,
  canonicalCodeToLabel,
  labelToCanonicalCode,
  customTagIdFromCanonicalCode,
  canonicalCodeFromCustomTagId,
  canonicalizeBinaryPatternMinCode,
  matchCustomCodewordsAnyRotation,
  displayLabelForTagId,
  patternForResolvedTagId,
} from '@/lib/tag36h11'
import { TAG_MODULE_CELL, type TagPattern } from '@/lib/tagModuleCell'

describe('tag36h11', () => {
  describe('hammingDistance', () => {
    it('returns 0 for identical codes', () => {
      expect(hammingDistance(0x12345n, 0x12345n)).toBe(0)
    })

    it('returns 1 for single bit difference', () => {
      expect(hammingDistance(1n, 0n)).toBe(1)
    })

    it('counts multiple bit differences', () => {
      expect(hammingDistance(0b11111n, 0n)).toBe(5)
    })
  })

  describe('patternToCode', () => {
    it('handles single-bit patterns (MSB-first)', () => {
      // pattern[0] = row=0, col=0 → bit_x[0]=1, bit_y[0]=1 → bit 35 of code
      const pattern: (0 | 1)[] = Array.from({ length: 36 }, () => 0)
      pattern[0] = 1
      expect(patternToCode(pattern)).toBe(1n << 35n)
    })

    it('handles mixed patterns using LUT mapping', () => {
      // pattern[0] = row=0, col=0 → BIT_X[0]=1, BIT_Y[0]=1 → bit 0 of code → code bit 35
      // pattern[6] = row=1, col=0 → BIT_X[7]=4, BIT_Y[7]=2 → bit 7 of code → code bit 28
      const pattern: (0 | 1)[] = Array.from({ length: 36 }, () => 0)
      pattern[0] = 1 // bit 0 → code bit 35
      pattern[6] = 1 // bit 6 → code bit 29 (index 6 → BIT_X[6]=3, BIT_Y[6]=2 → row=1,col=2)
      const code = patternToCode(pattern)
      // Verify by decoding back
      const decoded = codeToPattern(code)
      expect(decoded[0]).toBe(1)
      expect(decoded[6]).toBe(1)
    })

    it('rejects wrong-length patterns', () => {
      expect(patternToCode([1, 0, 1])).toBe(-1n)
    })

    it('encodes valid tag patterns correctly', () => {
      // Build a valid pattern: border=0, some interior cells=1
      const pattern: (0 | 1)[] = Array.from({ length: 36 }, () => 0)
      pattern[7] = 1 // row=1, col=1 (interior)
      pattern[13] = 1 // row=2, col=1 (interior)
      const code = patternToCode(pattern)
      // Verify round-trip
      const decoded = codeToPattern(code)
      expect(decoded[7]).toBe(1)
      expect(decoded[13]).toBe(1)
    })
  })

  describe('codeToPattern', () => {
    it('produces valid patterns for reference codes', () => {
      // Every reference code should decode to a pattern that round-trips
      for (const code of [tag36h11Code(0), tag36h11Code(100), tag36h11Code(300), tag36h11Code(586)]) {
        const pattern = codeToPattern(code)
        expect(pattern).toHaveLength(36)
        // Decode should find a matching tag in the dictionary
        const decoded = decodeTag36h11(pattern, 5)
        expect(decoded).toBeGreaterThanOrEqual(0)
      }
    })

    it('round-trips reference codes (encode→decode→encode)', () => {
      for (const code of [tag36h11Code(0), tag36h11Code(100), tag36h11Code(586)]) {
        const pattern = codeToPattern(code)
        const reconstructed = patternToCode(pattern)
        expect(reconstructed).toBe(code)
      }
    })
  })

  describe('rotatePattern', () => {
    it('rotates 90 degrees clockwise', () => {
      const pattern: (0 | 1 | -1)[] = Array.from({ length: 36 }, () => 0)
      pattern[0] = 1 // row=0, col=0
      const rotated = rotatePattern(pattern)
      expect(rotated[5]).toBe(1) // (0,0) → (0,5)
    })

    it('four rotations return to original', () => {
      const pattern = [
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
      ] as (0 | 1 | -1 | -2)[]
      let current = pattern
      for (let i = 0; i < 4; i++) {
        current = rotatePattern(current)
      }
      expect(current).toEqual(pattern)
    })
  })

  describe('decodeTag36h11', () => {
    it('returns -1 when pattern is undefined', () => {
      expect(decodeTag36h11(undefined)).toBe(-1)
    })

    it('returns -1 for wrong-length pattern', () => {
      expect(decodeTag36h11([1, 0, 1] as unknown as (0 | 1 | -1)[])).toBe(-1)
    })

    it('has dictionary with 587 entries', () => {
      expect(TAG36H11_COUNT).toBe(587)
    })

    it('decodes all reference codes correctly', () => {
      for (let i = 0; i < TAG36H11_COUNT; i++) {
        const pattern = codeToPattern(tag36h11Code(i))
        const decoded = decodeTag36h11(pattern, 5)
        expect(decoded).toBe(i)
      }
    })
  })

  describe('decodeTag36h11AnyRotation', () => {
    it('returns undefined for undefined pattern', () => {
      expect(decodeTag36h11AnyRotation(undefined, 5)).toBeUndefined()
    })

    it('chooses rotation with lowest Hamming distance', () => {
      const tagId = 42
      const canonical = codeToPattern(tag36h11Code(tagId))
      let p = canonical
      for (let k = 0; k < 2; k++) {
        p = rotatePattern(p)
      }
      const m = decodeTag36h11AnyRotation(p, 7)
      expect(m).not.toBeNull()
      expect(m!.id).toBe(tagId)
      expect(m!.rotation).toBe(2)
    })
  })

  describe('custom tag ids and labels', () => {
    it('round-trips canonical code through base64url label', () => {
      const code = 0x123456789n
      const label = canonicalCodeToLabel(code)
      expect(labelToCanonicalCode(label)).toBe(code)
    })

    it('maps custom id to code and display label', () => {
      const code = 42n
      const id = customTagIdFromCanonicalCode(code)
      expect(canonicalCodeFromCustomTagId(id)).toBe(code)
      expect(displayLabelForTagId(id)).toBe(canonicalCodeToLabel(code))
      expect(displayLabelForTagId(0)).toBe('0')
    })

    it('patternForResolvedTagId supports custom ids for GPU packing', () => {
      const code = 0xdeadbeefn & ((1n << 36n) - 1n)
      const id = customTagIdFromCanonicalCode(code)
      const pat = patternForResolvedTagId(id)
      expect(patternToCode(pat)).toBe(code)
    })

    it('canonicalizeBinaryPatternMinCode picks min code rotation', () => {
      const p = Array(36).fill(TAG_MODULE_CELL.black) as TagPattern
      p[0] = TAG_MODULE_CELL.white
      const c = canonicalizeBinaryPatternMinCode(p)
      expect(c).toBeDefined()
    })

    it('matchCustomCodewordsAnyRotation matches session codeword with weak cell', () => {
      const code = 0xabcdef012n
      const full = codeToPattern(code)
      const noisy = [...full] as TagPattern
      noisy[10] = TAG_MODULE_CELL.weak
      const m = matchCustomCodewordsAnyRotation(noisy, [code], 3)
      expect(m).toBeDefined()
      expect(m!.tagId).toBe(customTagIdFromCanonicalCode(code))
    })
  })
})

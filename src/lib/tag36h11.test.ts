import { describe, it, expect } from 'vitest';
import {
  hammingDistance,
  patternToCode,
  decodeTag36h11,
  rotatePattern,
  decodeTag36h11AnyRotation,
  TAG36H11_COUNT,
} from './tag36h11';

describe('tag36h11', () => {
  describe('hammingDistance', () => {
    it('returns 0 for identical codes', () => {
      expect(hammingDistance(0x12345, 0x12345)).toBe(0);
    });

    it('returns 1 for single bit difference', () => {
      expect(hammingDistance(0x00001, 0x00000)).toBe(1);
    });

    it('counts multiple bit differences', () => {
      expect(hammingDistance(0x11111, 0x00000)).toBe(5);
    });
  });

  describe('patternToCode', () => {
    it('converts all-white pattern to 0', () => {
      const pattern = new Array(36).fill(0) as (0 | 1)[];
      expect(patternToCode(pattern)).toBe(0);
    });

    it('handles single-bit patterns', () => {
      const pattern = new Array(36).fill(0) as (0 | 1)[];
      pattern[0] = 1; // only bit 0 set
      expect(patternToCode(pattern)).toBe(1);
    });

    it('handles mixed patterns', () => {
      const pattern = new Array(36).fill(0) as (0 | 1)[];
      pattern[0] = 1; // bit 0
      pattern[6] = 1; // bit 6
      expect(patternToCode(pattern)).toBe(0x41); // bits 0 + 6
    });

    it('rejects wrong-length patterns', () => {
      expect(patternToCode([1, 0, 1] as unknown as (0 | 1 | -1)[])).toBe(-1);
    });
  });

  describe('rotatePattern', () => {
    it('rotates 90 degrees clockwise', () => {
      const pattern = new Array(36).fill(0) as (0 | 1 | -1)[];
      pattern[0] = 1; // bit at row 0, col 0
      const rotated = rotatePattern(pattern);
      // After 90° clockwise: (row, col) → (col, 5-row)
      // Original (0,0) → (0, 5) = index 5
      expect(rotated[5]).toBe(1);
    });

    it('four rotations return to original', () => {
      const pattern = [
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
      ] as (0 | 1 | -1)[];
      let current = pattern;
      for (let i = 0; i < 4; i++) {
        current = rotatePattern(current);
      }
      expect(current).toEqual(pattern);
    });
  });

  describe('decodeTag36h11', () => {
    it('returns -1 for null pattern', () => {
      expect(decodeTag36h11(null)).toBe(-1);
    });

    it('returns -1 for wrong-length pattern', () => {
      expect(decodeTag36h11([1, 0, 1] as unknown as (0 | 1 | -1)[])).toBe(-1);
    });

    it('has dictionary with entries', () => {
      expect(TAG36H11_COUNT).toBeGreaterThan(0);
    });
  });

  describe('decodeTag36h11AnyRotation', () => {
    it('returns null for null pattern', () => {
      expect(decodeTag36h11AnyRotation(null)).toBeNull();
    });
  });
});
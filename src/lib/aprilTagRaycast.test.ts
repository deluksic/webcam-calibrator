import { describe, it, expect } from 'vitest';
import { applyHomography, computeHomography, type Point } from './geometry';
import {
  finiteDifferenceSobelFromIntensity,
  imagePixelToUnitSquareUv,
  intensityAtTagUv,
  invertMat3RowMajor,
  renderAprilTagIntensity,
  renderAprilTagSobelFiniteDifference,
  sampleIntensityBilinear,
} from './aprilTagRaycast';
import { TAG36H11_CODES, codeToPattern, decodeTag36h11AnyRotation } from './tag36h11';
import { buildTagGrid, decodeTagPattern } from './grid';

/** TL, TR, BR, BL for `buildTagGrid` from homography strip TL, TR, BL, BR. */
function cornersGridOrder(strip: [Point, Point, Point, Point]): [Point, Point, Point, Point] {
  return [strip[0], strip[1], strip[3], strip[2]];
}

describe('aprilTagRaycast', () => {
  it('invertMat3RowMajor * M ≈ I', () => {
    const M = [2, 0, 1, 0, 3, 0, 0, 0, 1];
    const inv = invertMat3RowMajor(M)!;
    const mul = (A: number[], B: number[]) => {
      const o = new Array(9).fill(0);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          for (let k = 0; k < 3; k++) {
            o[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
          }
        }
      }
      return o;
    };
    const I = mul(M, [...inv]);
    expect(I[0]).toBeCloseTo(1, 6);
    expect(I[4]).toBeCloseTo(1, 6);
    expect(I[8]).toBeCloseTo(1, 6);
    expect(I[1]).toBeCloseTo(0, 5);
    expect(I[3]).toBeCloseTo(0, 5);
  });

  it('imagePixelToUnitSquareUv round-trips forward homography on a square', () => {
    const strip: [Point, Point, Point, Point] = [
      { x: 20, y: 20 },
      { x: 220, y: 30 },
      { x: 10, y: 200 },
      { x: 210, y: 210 },
    ];
    const h = computeHomography([...strip]);
    const u0 = 0.37;
    const v0 = 0.52;
    const p = applyHomography(h, u0, v0);
    const back = imagePixelToUnitSquareUv(h, p.x, p.y);
    expect(back.inside).toBe(true);
    expect(back.u).toBeCloseTo(u0, 5);
    expect(back.v).toBeCloseTo(v0, 5);
  });

  it('intensityAtTagUv: black 1/8 border + inner 6×6 from pattern', () => {
    const pattern = codeToPattern(TAG36H11_CODES[0]);
    expect(intensityAtTagUv(0.04, 0.04, pattern)).toBe(0);
    expect(intensityAtTagUv(0.96, 0.96, pattern)).toBe(0);
    const innerU = 3.5 / 8;
    const innerV = 3.5 / 8;
    const bit = pattern[2 * 6 + 2];
    expect(intensityAtTagUv(innerU, innerV, pattern)).toBe(bit === 1 ? 0 : 1);
    const lastU = 6.5 / 8;
    const lastV = 6.5 / 8;
    const bitBr = pattern[5 * 6 + 5];
    expect(intensityAtTagUv(lastU, lastV, pattern)).toBe(bitBr === 1 ? 0 : 1);
  });

  it('renderAprilTagIntensity matches UV law at each cell center (forward projection)', () => {
    const pattern = codeToPattern(TAG36H11_CODES[42]);
    const strip: [Point, Point, Point, Point] = [
      { x: 40, y: 40 },
      { x: 280, y: 45 },
      { x: 35, y: 260 },
      { x: 275, y: 265 },
    ];
    const w = 320;
    const h = 320;
    const intensity = renderAprilTagIntensity({ width: w, height: h, corners: strip, pattern });
    const h8 = computeHomography([...strip]);

    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 6; col++) {
        const u = (col + 1.5) / 8;
        const v = (row + 1.5) / 8;
        const want = intensityAtTagUv(u, v, pattern);
        const p = applyHomography(h8, u, v);
        const xi = Math.max(0, Math.min(w - 1, Math.round(p.x)));
        const yi = Math.max(0, Math.min(h - 1, Math.round(p.y)));
        const got = intensity[yi * w + xi];
        expect(got).toBe(want);
      }
    }
  });

  it('finiteDifferenceSobelFromIntensity has strong response on a vertical black/white edge', () => {
    const w = 64;
    const h = 64;
    const I = new Float32Array(w * h).fill(1);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < 32; x++) {
        I[y * w + x] = 0;
      }
    }
    const s = finiteDifferenceSobelFromIntensity(I, w, h);
    const cx = 32;
    const cy = 32;
    const o = (cy * w + cx) * 2;
    expect(Math.abs(s[o])).toBeGreaterThan(0.2);
  });

  it('buildTagGrid cell centers: UV indices match row/col; raster matches tag law (bilinear)', () => {
    const tagId = 7;
    const pattern = codeToPattern(TAG36H11_CODES[tagId]);
    const size = 360;
    const strip: [Point, Point, Point, Point] = [
      { x: 20, y: 20 },
      { x: 20 + size, y: 20 },
      { x: 20, y: 20 + size },
      { x: 20 + size, y: 20 + size },
    ];
    const w = 400;
    const h = 400;
    const intensity = renderAprilTagIntensity({ width: w, height: h, corners: strip, pattern });
    const grid = buildTagGrid(cornersGridOrder(strip), 6);
    const h8 = computeHomography([...strip]);

    for (const cell of grid.cells) {
      const { u, v, inside } = imagePixelToUnitSquareUv(h8, cell.center.x, cell.center.y);
      expect(inside).toBe(true);
      const col = Math.min(5, Math.max(0, Math.floor(u * 6 - 1e-9)));
      const row = Math.min(5, Math.max(0, Math.floor(v * 6 - 1e-9)));
      expect(row).toBe(cell.row);
      expect(col).toBe(cell.col);

      const analytical = intensityAtTagUv(u, v, pattern);
      const pix = sampleIntensityBilinear(intensity, w, h, cell.center.x, cell.center.y);
      expect(pix).toBeCloseTo(analytical, 4);
    }
  });

  it('decodeTagPattern recovers dictionary id from synthetic raycast + finite-difference Sobel', () => {
    const tagId = 0;
    const pattern = codeToPattern(TAG36H11_CODES[tagId]);
    const size = 360;
    const strip: [Point, Point, Point, Point] = [
      { x: 20, y: 20 },
      { x: 20 + size, y: 20 },
      { x: 20, y: 20 + size },
      { x: 20 + size, y: 20 + size },
    ];
    const w = 400;
    const h = 400;
    const { sobel } = renderAprilTagSobelFiniteDifference(
      { width: w, height: h, corners: strip, pattern, supersample: 4 },
      { gradientScale: 4 },
    );
    const grid = buildTagGrid(cornersGridOrder(strip), 6);
    const decodedPattern = decodeTagPattern(grid, sobel, w, undefined, h);
    const match = decodeTag36h11AnyRotation(decodedPattern, 8);
    expect(match).not.toBeNull();
    expect(match!.id).toBe(tagId);
  });
});

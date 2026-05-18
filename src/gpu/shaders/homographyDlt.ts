import { d, std, tgpu } from 'typegpu'
import { abs } from 'typegpu/std'

import { HOMOGRAPHY_PIVOT_EPS } from '@/gpu/lineFitThresholds'

const N = 8

const LinSys8 = d.struct({
  A: d.arrayOf(d.f32, N * N),
  b: d.arrayOf(d.f32, N),
})

const RowMajor9 = d.arrayOf(d.f32, 9)

export const HomographyResult = d.struct({
  ok: d.u32,
  /** Column-major `mat3x3f` for grid viz (`mul(H, vec3(uv,1))`). */
  homography: d.mat3x3f,
})

/** All UVs map to one image point — grid viz has no area (matches empty/padded quads in the GPU buffer). */
export function invalidGridHomography() {
  'use gpu'
  return d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1)
}

function mat3FromRowMajor(h: ReturnType<typeof RowMajor9>) {
  'use gpu'
  return d.mat3x3f(h[0]!, h[3]!, h[6]!, h[1]!, h[4]!, h[7]!, h[2]!, h[5]!, h[8]!)
}

function aIdx(row: number, col: number) {
  'use gpu'
  return d.u32(row) * d.u32(N) + d.u32(col)
}

/** 4-point DLT: unit square (TL,TR,BL,BR) → image corners; same system as CPU `tryComputeHomography`. */
export const tryHomographyFromCorners = tgpu.fn(
  [d.vec2f, d.vec2f, d.vec2f, d.vec2f],
  HomographyResult,
)((tl, tr, bl, br) => {
  'use gpu'
  const sys = LinSys8({
    A: d.arrayOf(d.f32, N * N)(),
    b: d.arrayOf(d.f32, N)(),
  })

  const pts = [d.vec2f(tl), d.vec2f(tr), d.vec2f(bl), d.vec2f(br)]
  const uv = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)]

  for (const i of tgpu.unroll(std.range(0, 4))) {
    const u = uv[i]!.x
    const v = uv[i]!.y
    const p = pts[i]!
    const r0 = d.u32(i * 2)
    const r1 = r0 + d.u32(1)
    sys.A[aIdx(r0, d.u32(0))] = u
    sys.A[aIdx(r0, d.u32(1))] = v
    sys.A[aIdx(r0, d.u32(2))] = d.f32(1)
    sys.A[aIdx(r0, d.u32(3))] = d.f32(0)
    sys.A[aIdx(r0, d.u32(4))] = d.f32(0)
    sys.A[aIdx(r0, d.u32(5))] = d.f32(0)
    sys.A[aIdx(r0, d.u32(6))] = -u * p.x
    sys.A[aIdx(r0, d.u32(7))] = -v * p.x
    sys.b[r0] = p.x

    sys.A[aIdx(r1, d.u32(0))] = d.f32(0)
    sys.A[aIdx(r1, d.u32(1))] = d.f32(0)
    sys.A[aIdx(r1, d.u32(2))] = d.f32(0)
    sys.A[aIdx(r1, d.u32(3))] = u
    sys.A[aIdx(r1, d.u32(4))] = v
    sys.A[aIdx(r1, d.u32(5))] = d.f32(1)
    sys.A[aIdx(r1, d.u32(6))] = -u * p.y
    sys.A[aIdx(r1, d.u32(7))] = -v * p.y
    sys.b[r1] = p.y
  }

  for (const col of tgpu.unroll(std.range(0, N))) {
    const colU = d.u32(col)
    let maxRow = colU
    for (const row of tgpu.unroll(std.range(0, N))) {
      const rowU = d.u32(row)
      if (rowU > colU) {
        if (abs(sys.A[aIdx(rowU, colU)]!) > abs(sys.A[aIdx(maxRow, colU)]!)) {
          maxRow = rowU
        }
      }
    }

    if (maxRow !== colU) {
      for (const j of tgpu.unroll(std.range(0, N))) {
        const jU = d.u32(j)
        const idx0 = aIdx(colU, jU)
        const idx1 = aIdx(maxRow, jU)
        const tmpA = sys.A[idx0]!
        sys.A[idx0] = sys.A[idx1]!
        sys.A[idx1] = tmpA
      }
      const tmpB = sys.b[colU]!
      sys.b[colU] = sys.b[maxRow]!
      sys.b[maxRow] = tmpB
    }

    const pivot = sys.A[aIdx(colU, colU)]!
    if (abs(pivot) < d.f32(HOMOGRAPHY_PIVOT_EPS)) {
      return HomographyResult({ ok: d.u32(0), homography: invalidGridHomography() })
    }

    for (const j of tgpu.unroll(std.range(0, N))) {
      const jU = d.u32(j)
      if (jU >= colU) {
        sys.A[aIdx(colU, jU)] = sys.A[aIdx(colU, jU)]! / pivot
      }
    }
    sys.b[colU] = sys.b[colU]! / pivot

    for (const row of tgpu.unroll(std.range(0, N))) {
      const rowU = d.u32(row)
      if (rowU !== colU) {
        const factor = sys.A[aIdx(rowU, colU)]!
        if (factor !== d.f32(0)) {
          for (const j of tgpu.unroll(std.range(0, N))) {
            const jU = d.u32(j)
            if (jU >= colU) {
              sys.A[aIdx(rowU, jU)] = sys.A[aIdx(rowU, jU)]! - factor * sys.A[aIdx(colU, jU)]!
            }
          }
          sys.b[rowU] = sys.b[rowU]! - factor * sys.b[colU]!
        }
      }
    }
  }

  const hOut = RowMajor9()
  for (const i of tgpu.unroll(std.range(0, N))) {
    hOut[i] = sys.b[d.u32(i)]!
  }
  hOut[d.u32(8)] = d.f32(1)

  return HomographyResult({ ok: d.u32(1), homography: mat3FromRowMajor(hOut) })
})

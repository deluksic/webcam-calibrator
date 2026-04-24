import { describe, expect, it } from 'vitest'

import { homographyFromKT, rotY } from '@/lib/calibrationTestUtils'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import { solveIntrinsicsFromHomographies } from '@/lib/zhangCalibration'
import type { Mat3 } from '@/lib/geometry'

const K0: CameraIntrinsics = { fx: 900, fy: 880, cx: 512, cy: 384 }
const Krow: [number, number, number, number, number, number, number, number, number] = [K0.fx, 0, K0.cx, 0, K0.fy, K0.cy, 0, 0, 1]

function collectHs(angles: number[], tzBase: number): Mat3[] {
  const out: Mat3[] = []
  for (let i = 0; i < angles.length; i++) {
    const R = rotY(angles[i]!)
    const tx = 0.02 * (i - 1)
    const ty = 0.01 * i
    const tz = tzBase + 0.2 * i
    out.push(homographyFromKT(Krow, R, tx, ty, tz))
  }
  return out
}

describe('solveIntrinsicsFromHomographies (Zhang)', () => {
  it('recovers similar intrinsics from noise-free synthetic homographies (3+ views)', () => {
    const hs = collectHs(
      [0.12, -0.08, 0.18, -0.11, 0.05, 0.22, -0.15],
      2.2,
    )
    const k = solveIntrinsicsFromHomographies(hs)
    expect(k).toBeDefined()
    if (!k) {
      return
    }
    expect(k.fx).toBeCloseTo(K0.fx, 0.5)
    expect(k.fy).toBeCloseTo(K0.fy, 0.5)
    expect(k.cx).toBeCloseTo(K0.cx, 0.5)
    expect(k.cy).toBeCloseTo(K0.cy, 0.5)
  })
})

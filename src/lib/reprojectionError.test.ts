import { describe, expect, it } from 'vitest'

import type { CameraIntrinsics } from '@/lib/cameraModel'
import { projectPlanePoint } from '@/lib/reprojectionError'
import type { Mat3R } from '@/lib/zhangCalibration'

const K: CameraIntrinsics = { fx: 500, fy: 510, cx: 300, cy: 200 }

const I: Mat3R = [1, 0, 0, 0, 1, 0, 0, 0, 1]

describe('projectPlanePoint', () => {
  it('maps plane origin to principal point for identity R and t = (0,0,2)', () => {
    const t = { x: 0, y: 0, z: 2 }
    const p = projectPlanePoint(K, I, t, 0, 0)
    expect(p.x).toBeCloseTo(K.cx, 5)
    expect(p.y).toBeCloseTo(K.cy, 5)
  })
})

import { describe, expect, it } from 'vitest'

import { rotY } from '@/lib/calibrationTestUtils'
import type { CalibrationFrameObservation, LabeledPoint, FramePoint } from '@/lib/calibrationTypes'
import { solveCalibration } from '@/lib/calibrationSolve'
import type { Corners } from '@/lib/geometry'
import { projectPlanePoint } from '@/lib/reprojectionError'
import type { TargetLayout } from '@/lib/targetLayout'
import type { Mat3R } from '@/lib/zhangCalibration'
import type { CameraIntrinsics } from '@/lib/cameraModel'

const Ktrue: CameraIntrinsics = { fx: 900, fy: 880, cx: 512, cy: 384 }
const layoutPlane = new Map<number, Corners>([
  [0, squarePlane(0, 0)],
  [1, squarePlane(1.3, 0.1)],
]) as unknown as TargetLayout

function squarePlane(ox: number, oy: number): Corners {
  return [
    { x: ox, y: oy },
    { x: ox + 1, y: oy },
    { x: ox, y: oy + 1 },
    { x: ox + 1, y: oy + 1 },
  ]
}

function frameCorners(R: Mat3R, t: { x: number; y: number; z: number }): { tag0: Corners; tag1: Corners } {
  const p0 = layoutPlane.get(0)!
  const p1 = layoutPlane.get(1)!
  return {
    tag0: p0.map((c) => projectPlanePoint(Ktrue, R, t, c.x, c.y)) as Corners,
    tag1: p1.map((c) => projectPlanePoint(Ktrue, R, t, c.x, c.y)) as Corners,
  }
}

function oneFrame(
  id: number,
  R: Mat3R,
  t: { x: number; y: number; z: number },
  score: number,
): CalibrationFrameObservation {
  const c = frameCorners(R, t)
  const framePoints: FramePoint[] = [
    { pointId: 0, imagePoint: c.tag0[0]! },
    { pointId: 1, imagePoint: c.tag0[1]! },
    { pointId: 2, imagePoint: c.tag0[2]! },
    { pointId: 3, imagePoint: c.tag0[3]! },
    { pointId: 10000, imagePoint: c.tag1[0]! },
    { pointId: 10001, imagePoint: c.tag1[1]! },
    { pointId: 10002, imagePoint: c.tag1[2]! },
    { pointId: 10003, imagePoint: c.tag1[3]! },
  ]
  return { frameId: id, framePoints }
}

function syntheticLabeledPoints(): LabeledPoint[] {
  return [
    { pointId: 0, plane: { x: 0, y: 0 } },
    { pointId: 1, plane: { x: 1, y: 0 } },
    { pointId: 2, plane: { x: 0, y: 1 } },
    { pointId: 3, plane: { x: 1, y: 1 } },
    { pointId: 10000, plane: { x: 1.3, y: 0.1 } },
    { pointId: 10001, plane: { x: 2.3, y: 0.1 } },
    { pointId: 10002, plane: { x: 1.3, y: 1.1 } },
    { pointId: 10003, plane: { x: 2.3, y: 1.1 } },
  ]
}

function syntheticFrames(): CalibrationFrameObservation[] {
  return [
    oneFrame(1, rotY(0.1), { x: 0.0, y: 0.0, z: 2.4 }, 0.9),
    oneFrame(2, rotY(-0.07), { x: 0.03, y: -0.01, z: 2.5 }, 0.9),
    oneFrame(3, rotY(0.15), { x: -0.02, y: 0.02, z: 2.35 }, 0.9),
    oneFrame(4, rotY(-0.12), { x: 0.01, y: 0.01, z: 2.45 }, 0.9),
  ]
}

describe('solveCalibration', () => {
  it('succeeds on synthetic multi-tag, multi-view data with low RMS', () => {
    const frames = syntheticFrames()
    const labeledPoints = syntheticLabeledPoints()
    const res = solveCalibration(layoutPlane, labeledPoints, frames)
    expect(res.kind).toBe('ok')
    if (res.kind !== 'ok') {
      return
    }
    expect(res.rmsPx).toBeLessThan(2)
    expect(res.K.fx).toBeCloseTo(Ktrue.fx, 0)
    expect(res.K.fy).toBeCloseTo(Ktrue.fy, 0)
  })

  it('returns too-few-views with <3 valid homographies', () => {
    const f = [oneFrame(1, rotY(0.1), { x: 0, y: 0, z: 2.5 }, 1)]
    const labeledPoints = syntheticLabeledPoints()
    const r = solveCalibration(layoutPlane, labeledPoints, f)
    expect(r.kind).toBe('error')
    if (r.kind === 'error') {
      expect(r.reason).toBe('too-few-views')
    }
  })
})

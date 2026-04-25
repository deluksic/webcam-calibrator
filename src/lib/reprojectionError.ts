// Reproject board plane (X,Y, z=0) with p_c = R * [x,y,0] + t; pinhole: u = fx*Xc/zc + cx, v = fy*Yc/zc + cy.

import type { Point } from '@/lib/geometry'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { Vec3, Mat3 } from '@/workers/calibration.worker'

function abs(x: number) {
  return x < 0 ? -x : x
}

export function projectPlanePoint(k: CameraIntrinsics, R: Mat3, t: Vec3, x: number, y: number): Point {
  const Xc = R[0]! * x + R[1]! * y + t.x
  const Yc = R[3]! * x + R[4]! * y + t.y
  const zc = R[6]! * x + R[7]! * y + t.z
  if (abs(zc) < 1e-15) {
    return { x: 0, y: 0 }
  }
  return { x: (k.fx * Xc) / zc + k.cx, y: (k.fy * Yc) / zc + k.cy }
}

export type Vec3 = { x: number; y: number; z: number }
export type Mat3 = [number, number, number, number, number, number, number, number, number]

const { acos, hypot } = Math

export function rvecToMatrix(rvec: [number, number, number]): Mat3 {
  const theta = hypot(rvec[0], rvec[1], rvec[2])
  if (theta < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  }
  const ux = rvec[0] / theta
  const uy = rvec[1] / theta
  const uz = rvec[2] / theta
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const t = 1 - c
  return [
    t * ux * ux + c,
    t * ux * uy - s * uz,
    t * ux * uz + s * uy,
    t * ux * uy + s * uz,
    t * uy * uy + c,
    t * uy * uz - s * ux,
    t * ux * uz - s * uy,
    t * uy * uz + s * ux,
    t * uz * uz + c,
  ]
}

export function cameraTiltDegFromR(R: Mat3): number {
  const nx = R[2]!
  const ny = R[5]!
  const nz = R[8]!
  const length = hypot(nx, hypot(ny, nz)) + 1e-12
  return (acos(Math.min(1, Math.max(-1, nz / length))) * 180) / Math.PI
}

export function cameraDistanceFromT(t: Vec3): number {
  return hypot(t.x, hypot(t.y, t.z))
}

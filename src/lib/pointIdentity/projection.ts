import type { PatternPoint, ViewPose, Vec2 } from '@/lib/pointIdentity/types'

const DEG_TO_RAD = Math.PI / 180
/** Positive slider values compress the outer field like a fisheye lens. */
const FISHEYE_GAIN = 0.6

export const DEFAULT_VIEW: ViewPose = {
  yawDeg: 0,
  pitchDeg: 0,
  cameraDistance: 2200,
  focalLength: 1000,
  sphericalDistortion: 0,
}

/**
 * Pinhole view of the target plane.
 *
 * The target starts in the XY plane. Yaw rotates around Y, pitch rotates around
 * X, and the camera sits at `(0, 0, cameraDistance)` looking at the origin.
 * Auto-fitting to the canvas happens later, so the projection stays a pure,
 * unit-preserving camera transform.
 */
export function projectPoint(point: Vec2, pose: ViewPose): Vec2 {
  const yaw = pose.yawDeg * DEG_TO_RAD
  const pitch = pose.pitchDeg * DEG_TO_RAD
  const cosYaw = Math.cos(yaw)
  const sinYaw = Math.sin(yaw)
  const cosPitch = Math.cos(pitch)
  const sinPitch = Math.sin(pitch)

  // Yaw around Y, then pitch around X.
  const x1 = point.x * cosYaw
  const y1 = point.y
  const z1 = -point.x * sinYaw

  const x2 = x1
  const y2 = y1 * cosPitch - z1 * sinPitch
  const z2 = y1 * sinPitch + z1 * cosPitch

  const depth = Math.max(pose.cameraDistance - z2, 1e-3)
  const perspective = pose.focalLength / depth
  return { x: x2 * perspective, y: y2 * perspective }
}

export function projectPattern(points: readonly PatternPoint[], pose: ViewPose): PatternPoint[] {
  const pinhole = points.map((point) => {
    const projected = projectPoint(point, pose)
    return { x: projected.x, y: projected.y, index: point.index }
  })

  const strength = -pose.sphericalDistortion * FISHEYE_GAIN
  if (Math.abs(strength) < 1e-6 || pinhole.length === 0) {
    return pinhole
  }

  let maxRadius = 0
  for (const point of pinhole) {
    maxRadius = Math.max(maxRadius, Math.hypot(point.x, point.y))
  }
  if (maxRadius < 1e-9) {
    return pinhole
  }

  return pinhole.map((point) => {
    const radius = Math.hypot(point.x, point.y)
    if (radius < 1e-9) {
      return point
    }
    const normalized = radius / maxRadius
    const warped = normalized * (1 + strength * normalized * normalized)
    const scale = warped / normalized
    return { x: point.x * scale, y: point.y * scale, index: point.index }
  })
}

/** Out-of-plane angle between the target normal and the optical axis. */
export function outOfPlaneTiltDeg(pose: ViewPose): number {
  const yaw = pose.yawDeg * DEG_TO_RAD
  const pitch = pose.pitchDeg * DEG_TO_RAD
  const cosine = Math.abs(Math.cos(yaw) * Math.cos(pitch))
  return Math.acos(Math.min(1, cosine)) / DEG_TO_RAD
}

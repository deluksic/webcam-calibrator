/**
 * Brown-Conrady Distortion Model
 *
 * Reference: opencv/modules/calib3d/src/calibration.cpp
 *
 * Model:
 *   r² = x² + y²
 *   x' = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)
 *        + (2*p1*x*y + p2*(r² + 2*x²))
 *   y' = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)
 *        + (p1*(r² + 2*y²) + 2*p2*x*y)
 */

/**
 * Distortion coefficients (8 parameters)
 * k1, k2: Radial distortion (2nd, 4th order)
 * p1, p2: Tangential distortion
 * k3, k4, k5, k6: Additional radial terms
 */
export interface DistortionCoeffs {
  k1: number
  k2: number
  p1: number
  p2: number
  k3: number
  k4: number
  k5: number
  k6: number
}

/** Normalized point before/after distortion */
export interface NormalizedPoint {
  x: number
  y: number
}

/**
 * Apply forward distortion to normalized coordinates.
 *
 * @param pt Input normalized point (x, y)
 * @param dist Distortion coefficients
 * @returns Distorted point (x', y')
 */
export function distortPoint(pt: NormalizedPoint, dist: DistortionCoeffs): NormalizedPoint {
  const { x, y } = pt
  const { k1, k2, p1, p2, k3, k4, k5, k6 } = dist

  const r2 = x * x + y * y
  const r4 = r2 * r2
  const r6 = r4 * r2

  // Radial distortion factor (rational form)
  const radialNum = 1 + k1 * r2 + k2 * r4 + k3 * r6
  const radialDen = 1 + k4 * r2 + k5 * r4 + k6 * r6

  let radialFactor: number
  if (radialDen < 1e-10) {
    radialFactor = 1 + k1 * r2 + k2 * r4 + k3 * r6
  } else {
    radialFactor = radialNum / radialDen
  }

  // Tangential distortion
  const tanX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
  const tanY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

  return {
    x: x * radialFactor + tanX,
    y: y * radialFactor + tanY,
  }
}

/**
 * Apply inverse distortion using Newton-Raphson iteration.
 *
 * @param distorted Distorted point (x_d, y_d)
 * @param dist Distortion coefficients
 * @param maxIter Maximum iterations (default 100)
 * @param tolerance Convergence tolerance (default 1e-10)
 * @returns Undistorted point (x, y)
 */
export function undistortPoint(
  distorted: NormalizedPoint,
  dist: DistortionCoeffs,
  maxIter = 100,
  tolerance = 1e-10
): NormalizedPoint {
  const { x: xd, y: yd } = distorted

  // Initial guess: undistorted ≈ distorted (works for small distortion)
  let x = xd
  let y = yd

  for (let i = 0; i < maxIter; i++) {
    // Apply distortion to guess
    const distortedGuess = distortPoint({ x, y }, dist)
    const dx = distortedGuess.x - xd
    const dy = distortedGuess.y - yd

    // Check convergence
    const err = dx * dx + dy * dy
    if (err < tolerance * tolerance) {
      break
    }

    // Newton-Raphson update: x_{n+1} = x_n - (f(x) - x_d) / f'(x)
    // Simplified: step toward the target by the error
    // For better convergence, use iterative refinement with Jacobian approximation
    const r2 = x * x + y * y
    const r4 = r2 * r2
    const r6 = r4 * r2
    const radialDeriv = 1 + 3 * dist.k1 * r2 + 5 * dist.k2 * r4 + 7 * dist.k3 * r6

    // Approximate correction (first-order approximation)
    const scale = 1 / Math.max(0.1, Math.abs(radialDeriv))
    x -= dx * scale
    y -= dy * scale
  }

  return { x, y }
}

/**
 * Create zero distortion coefficients (identity mapping).
 */
export function zeroDistortion(): DistortionCoeffs {
  return { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
}

/**
 * Create 5-parameter distortion (k1, k2, p1, p2, k3) - common in OpenCV.
 */
export function fiveParamDistortion(k1: number, k2: number, p1: number, p2: number, k3: number): DistortionCoeffs {
  return { k1, k2, p1, p2, k3, k4: 0, k5: 0, k6: 0 }
}

/**
 * Convert 8-param to 5-param (drops k4, k5, k6).
 */
export function toFiveParam(dist: DistortionCoeffs): number[] {
  return [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3]
}

/**
 * Convert 5-param to 8-param (k4=k5=k6=0).
 */
export function fromFiveParam(coeffs: readonly number[]): DistortionCoeffs {
  return {
    k1: coeffs[0] ?? 0,
    k2: coeffs[1] ?? 0,
    p1: coeffs[2] ?? 0,
    p2: coeffs[3] ?? 0,
    k3: coeffs[4] ?? 0,
    k4: 0,
    k5: 0,
    k6: 0,
  }
}

/**
 * Convert 8-param to 12-param (includes additional OpenCV parameters).
 */
export function toTwelveParam(dist: DistortionCoeffs): number[] {
  return [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3, 0, 0, 0, dist.k4, dist.k5, dist.k6, 0]
}

/**
 * Check if distortion is essentially zero (identity mapping).
 */
export function isZeroDistortion(dist: DistortionCoeffs, tol = 1e-10): boolean {
  return (
    Math.abs(dist.k1) < tol &&
    Math.abs(dist.k2) < tol &&
    Math.abs(dist.p1) < tol &&
    Math.abs(dist.p2) < tol &&
    Math.abs(dist.k3) < tol &&
    Math.abs(dist.k4) < tol &&
    Math.abs(dist.k5) < tol &&
    Math.abs(dist.k6) < tol
  )
}

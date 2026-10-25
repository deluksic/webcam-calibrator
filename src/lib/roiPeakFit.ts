/**
 * Fixed 5x5 cubic peak fit used by the ROI refined-peak detector.
 *
 * The same 25-point design is fitted for every candidate, so the least-squares
 * problem can be reduced to one precomputed 10x25 pseudo-inverse. The GPU
 * shader uses `CUBIC_FIT_COEFFICIENTS` directly; the CPU function below is a
 * line-for-line reference used by tests.
 *
 * Design (matching the Python model):
 *   u = dx / 2, v = dy / 2, dx,dy in [-2, 2]
 *   z = c0 + c1 u + c2 v + c3 u^2 + c4 uv + c5 v^2
 *         + c6 u^3 + c7 u^2 v + c8 uv^2 + c9 v^3
 */

export const CUBIC_5X5_RADIUS = 2
export const CUBIC_5X5_SAMPLE_COUNT = 25
export const CUBIC_5X5_TERM_COUNT = 10

/** Newton iteration limits. These are f32-safe variants of the Python f64 limits. */
export const NEWTON_MAX_ITERATIONS = 25
export const NEWTON_STEP_EPSILON = 1e-6
export const NEWTON_DET_EPSILON = 1e-9
export const NEWTON_VERTEX_EPSILON = 1e-5

/** Physical Hessian eigenvalue threshold for a definite peak. */
export const HESSIAN_EPSILON = 1e-7

/** OpenCV GaussianBlur(0, 0, sigma=1.0) uses a 9-tap kernel with these weights. */
export const GAUSSIAN_BLUR_RADIUS = 4
export const GAUSSIAN_BLUR_WEIGHTS: readonly number[] = [
  0.00013383062, 0.0044318614, 0.053991128, 0.24197145, 0.39894348, 0.24197145, 0.053991128, 0.0044318614,
  0.00013383062,
]

export type CubicPeakFit = {
  /** Stationary offset in normalized units, in [-1, 1]. */
  u: number
  v: number
  /** Cubic surface value at the stationary point. */
  peak: number
  /** Smaller physical Hessian eigenvalue. */
  eigenvalue1: number
  /** Larger physical Hessian eigenvalue. */
  eigenvalue2: number
  /** Physical Hessian off-diagonal term. */
  hessianXY: number
}

function cubicDesignRow(dx: number, dy: number): number[] {
  const u = dx / 2
  const v = dy / 2
  return [1, u, v, u * u, u * v, v * v, u * u * u, u * u * v, u * v * v, v * v * v]
}

export const CUBIC_DESIGN_MATRIX: readonly number[] = (() => {
  const rows: number[] = []
  for (let dy = -CUBIC_5X5_RADIUS; dy <= CUBIC_5X5_RADIUS; dy += 1) {
    for (let dx = -CUBIC_5X5_RADIUS; dx <= CUBIC_5X5_RADIUS; dx += 1) {
      rows.push(...cubicDesignRow(dx, dy))
    }
  }
  return rows
})()

function invertMatrix(matrix: readonly number[][]): number[][] {
  const size = matrix.length
  const augmented = matrix.map((row, rowIndex) => [
    ...row,
    ...Array.from({ length: size }, (_, columnIndex) => (rowIndex === columnIndex ? 1 : 0)),
  ])
  for (let column = 0; column < size; column += 1) {
    let pivotRow = column
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivotRow]![column]!)) {
        pivotRow = row
      }
    }
    if (Math.abs(augmented[pivotRow]![column]!) < 1e-12) {
      throw new Error('Cubic fit design matrix is singular.')
    }
    if (pivotRow !== column) {
      const temp = augmented[column]!
      augmented[column] = augmented[pivotRow]!
      augmented[pivotRow] = temp
    }
    const pivot = augmented[column]![column]!
    for (let entry = 0; entry < size * 2; entry += 1) {
      augmented[column]![entry] = augmented[column]![entry]! / pivot
    }
    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue
      }
      const factor = augmented[row]![column]!
      if (factor === 0) {
        continue
      }
      for (let entry = 0; entry < size * 2; entry += 1) {
        augmented[row]![entry] = augmented[row]![entry]! - factor * augmented[column]![entry]!
      }
    }
  }
  return augmented.map((row) => row.slice(size))
}

/**
 * P = (A^T A)^-1 A^T, stored term-major: P[term * 25 + sample].
 * Computed in f64 once at module load.
 */
export const CUBIC_FIT_COEFFICIENTS: readonly number[] = (() => {
  const normal = Array.from({ length: CUBIC_5X5_TERM_COUNT }, () =>
    Array.from({ length: CUBIC_5X5_TERM_COUNT }, () => 0),
  )
  for (let sample = 0; sample < CUBIC_5X5_SAMPLE_COUNT; sample += 1) {
    for (let first = 0; first < CUBIC_5X5_TERM_COUNT; first += 1) {
      const firstValue = CUBIC_DESIGN_MATRIX[sample * CUBIC_5X5_TERM_COUNT + first]!
      for (let second = 0; second < CUBIC_5X5_TERM_COUNT; second += 1) {
        normal[first]![second] =
          normal[first]![second]! + firstValue * CUBIC_DESIGN_MATRIX[sample * CUBIC_5X5_TERM_COUNT + second]!
      }
    }
  }
  const inverse = invertMatrix(normal)
  const coefficients = Array.from({ length: CUBIC_5X5_TERM_COUNT * CUBIC_5X5_SAMPLE_COUNT }, () => 0)
  for (let term = 0; term < CUBIC_5X5_TERM_COUNT; term += 1) {
    for (let sample = 0; sample < CUBIC_5X5_SAMPLE_COUNT; sample += 1) {
      let sum = 0
      for (let column = 0; column < CUBIC_5X5_TERM_COUNT; column += 1) {
        sum += inverse[term]![column]! * CUBIC_DESIGN_MATRIX[sample * CUBIC_5X5_TERM_COUNT + column]!
      }
      coefficients[term * CUBIC_5X5_SAMPLE_COUNT + sample] = sum
    }
  }
  return coefficients
})()

function sampleValues(values: ArrayLike<number>): Float64Array {
  const samples = new Float64Array(CUBIC_5X5_SAMPLE_COUNT)
  for (let index = 0; index < CUBIC_5X5_SAMPLE_COUNT; index += 1) {
    samples[index] = Number(values[index] ?? 0)
  }
  return samples
}

/**
 * CPU reference for the GPU fit. `values` is row-major 5x5, dy from -2 to 2 and
 * dx from -2 to 2. Returns null for a non-converged, non-finite, or out-of-window
 * stationary point. The caller applies the polarity-specific Hessian sign test.
 */
export function fitCubic5x5(values: ArrayLike<number>): CubicPeakFit | null {
  if (values.length < CUBIC_5X5_SAMPLE_COUNT) {
    return null
  }
  const samples = sampleValues(values)
  const coefficients = new Float64Array(CUBIC_5X5_TERM_COUNT)
  for (let term = 0; term < CUBIC_5X5_TERM_COUNT; term += 1) {
    let sum = 0
    for (let sample = 0; sample < CUBIC_5X5_SAMPLE_COUNT; sample += 1) {
      sum += CUBIC_FIT_COEFFICIENTS[term * CUBIC_5X5_SAMPLE_COUNT + sample]! * samples[sample]!
    }
    coefficients[term] = sum
  }
  if (!coefficients.every(Number.isFinite)) {
    return null
  }

  const c0 = coefficients[0]!
  const c1 = coefficients[1]!
  const c2 = coefficients[2]!
  const c3 = coefficients[3]!
  const c4 = coefficients[4]!
  const c5 = coefficients[5]!
  const c6 = coefficients[6]!
  const c7 = coefficients[7]!
  const c8 = coefficients[8]!
  const c9 = coefficients[9]!

  let u = 0
  let v = 0
  let converged = false
  for (let iteration = 0; iteration < NEWTON_MAX_ITERATIONS; iteration += 1) {
    const gx = c1 + 2 * c3 * u + c4 * v + 3 * c6 * u * u + 2 * c7 * u * v + c8 * v * v
    const gy = c2 + c4 * u + 2 * c5 * v + c7 * u * u + 2 * c8 * u * v + 3 * c9 * v * v
    const hxx = 2 * c3 + 6 * c6 * u + 2 * c7 * v
    const hxy = c4 + 2 * c7 * u + 2 * c8 * v
    const hyy = 2 * c5 + 2 * c8 * u + 6 * c9 * v
    const det = hxx * hyy - hxy * hxy
    if (!Number.isFinite(det) || Math.abs(det) < NEWTON_DET_EPSILON) {
      break
    }
    const stepU = (-hyy * gx + hxy * gy) / det
    const stepV = (hxy * gx - hxx * gy) / det
    if (!Number.isFinite(stepU) || !Number.isFinite(stepV)) {
      break
    }
    const stepLength = Math.hypot(stepU, stepV)
    const scale = stepLength > 0.5 ? 0.5 / stepLength : 1
    u += stepU * scale
    v += stepV * scale
    if (stepLength * scale < NEWTON_STEP_EPSILON) {
      converged = true
      break
    }
  }
  if (!converged) {
    return null
  }
  if (Math.abs(u) > 1 + NEWTON_VERTEX_EPSILON || Math.abs(v) > 1 + NEWTON_VERTEX_EPSILON) {
    return null
  }

  const hxx = 2 * c3 + 6 * c6 * u + 2 * c7 * v
  const hxy = c4 + 2 * c7 * u + 2 * c8 * v
  const hyy = 2 * c5 + 2 * c8 * u + 6 * c9 * v
  // d/dx = (1/2) d/du and d/dy = (1/2) d/dv.
  const physicalXX = hxx / 4
  const physicalXY = hxy / 4
  const physicalYY = hyy / 4
  const mean = (physicalXX + physicalYY) / 2
  const difference = (physicalXX - physicalYY) / 2
  const root = Math.sqrt(difference * difference + physicalXY * physicalXY)
  const peak =
    c0 +
    c1 * u +
    c2 * v +
    c3 * u * u +
    c4 * u * v +
    c5 * v * v +
    c6 * u ** 3 +
    c7 * u * u * v +
    c8 * u * v * v +
    c9 * v ** 3
  if (!Number.isFinite(peak)) {
    return null
  }
  return {
    u,
    v,
    peak,
    eigenvalue1: mean - root,
    eigenvalue2: mean + root,
    hessianXY: physicalXY,
  }
}

import { invertMat3RowMajor } from '@/lib/geometry'
import { applyHomography, type Mat3 } from '@/lib/geometry'
import { createRandom } from '@/lib/pointIdentity/random'
import type { Vec2 } from '@/lib/pointIdentity/types'

export type HomographyCorrespondence = {
  design: Vec2
  observed: Vec2
  weight?: number
}

export type HomographyFitResult = {
  matrix: Mat3
  /** Indices into the original correspondence array. */
  inlierIndices: number[]
  /** Design-space residual for every input correspondence. */
  residuals: number[]
  /** RMS design-space residual over the inlier set. */
  rms: number
  method: 'trimmed' | 'ransac'
}

export type RobustHomographyOptions = {
  /** Uniform design-space lattice spacing. */
  designSpacing: number
  trimFactor?: number
  trimFloor?: number
  maxRounds?: number
  minInliers?: number
  minInlierRatio?: number
  ransacIterations?: number
  seed?: number
  /** RANSAC inlier threshold as a fraction of design spacing. */
  inlierFactor?: number
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    return Number.POSITIVE_INFINITY
  }
  const sorted = [...values].sort((first, second) => first - second)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] | undefined {
  const size = rhs.length
  const augmented = matrix.map((row, index) => [...row, rhs[index]!])
  for (let column = 0; column < size; column += 1) {
    let pivotRow = column
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivotRow]![column]!)) {
        pivotRow = row
      }
    }
    const pivot = augmented[pivotRow]![column]!
    if (Math.abs(pivot) < 1e-12) {
      return undefined
    }
    ;[augmented[column], augmented[pivotRow]] = [augmented[pivotRow]!, augmented[column]!]
    for (let columnIndex = column; columnIndex <= size; columnIndex += 1) {
      augmented[column]![columnIndex]! /= pivot
    }
    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue
      }
      const factor = augmented[row]![column]!
      if (factor === 0) {
        continue
      }
      for (let columnIndex = column; columnIndex <= size; columnIndex += 1) {
        augmented[row]![columnIndex]! -= factor * augmented[column]![columnIndex]!
      }
    }
  }
  return augmented.map((row) => row[size]!)
}

function multiplyMat3(first: Mat3, second: Mat3): Mat3 {
  const result: number[] = []
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      let sum = 0
      for (let index = 0; index < 3; index += 1) {
        sum += first[row * 3 + index]! * second[index * 3 + column]!
      }
      result.push(sum)
    }
  }
  return result as Mat3
}

function finiteCorrespondences(
  correspondences: readonly HomographyCorrespondence[],
): Array<{ correspondence: HomographyCorrespondence; index: number }> {
  const result: Array<{ correspondence: HomographyCorrespondence; index: number }> = []
  for (let index = 0; index < correspondences.length; index += 1) {
    const correspondence = correspondences[index]!
    if (
      Number.isFinite(correspondence.design.x) &&
      Number.isFinite(correspondence.design.y) &&
      Number.isFinite(correspondence.observed.x) &&
      Number.isFinite(correspondence.observed.y) &&
      correspondence.design.x !== undefined &&
      correspondence.design.y !== undefined &&
      correspondence.observed.x !== undefined &&
      correspondence.observed.y !== undefined
    ) {
      result.push({ correspondence, index })
    }
  }
  return result
}

/**
 * Weighted normalized DLT with h33 = 1. Normalization keeps the normal
 * equations conditioned for design coordinates ~hundreds of units apart.
 */
export function fitHomography(correspondences: readonly HomographyCorrespondence[]): Mat3 | undefined {
  const finite = finiteCorrespondences(correspondences)
  if (finite.length < 4) {
    return undefined
  }

  let designMeanX = 0
  let designMeanY = 0
  let observedMeanX = 0
  let observedMeanY = 0
  for (const { correspondence } of finite) {
    designMeanX += correspondence.design.x
    designMeanY += correspondence.design.y
    observedMeanX += correspondence.observed.x
    observedMeanY += correspondence.observed.y
  }
  const count = finite.length
  designMeanX /= count
  designMeanY /= count
  observedMeanX /= count
  observedMeanY /= count

  let designScale = 0
  let observedScale = 0
  for (const { correspondence } of finite) {
    designScale += Math.hypot(correspondence.design.x - designMeanX, correspondence.design.y - designMeanY)
    observedScale += Math.hypot(
      correspondence.observed.x - observedMeanX,
      correspondence.observed.y - observedMeanY,
    )
  }
  designScale /= count
  observedScale /= count
  if (!(designScale > 1e-9) || !(observedScale > 1e-9)) {
    return undefined
  }

  const normal = Array.from({ length: 8 }, () => Array<number>(8).fill(0))
  const rhs = Array<number>(8).fill(0)
  for (const { correspondence } of finite) {
    const x = (correspondence.design.x - designMeanX) / designScale
    const y = (correspondence.design.y - designMeanY) / designScale
    const u = (correspondence.observed.x - observedMeanX) / observedScale
    const v = (correspondence.observed.y - observedMeanY) / observedScale
    const weight = Math.max(1e-6, correspondence.weight ?? 1)
    const rows: Array<[number[], number]> = [
      [[x, y, 1, 0, 0, 0, -x * u, -y * u], u],
      [[0, 0, 0, x, y, 1, -x * v, -y * v], v],
    ]
    for (const [row, value] of rows) {
      for (let first = 0; first < 8; first += 1) {
        for (let second = 0; second < 8; second += 1) {
          normal[first]![second]! += weight * row[first]! * row[second]!
        }
        rhs[first]! += weight * row[first]! * value
      }
    }
  }

  const solution = solveLinearSystem(normal, rhs)
  if (solution === undefined) {
    return undefined
  }

  const normalized: Mat3 = [
    solution[0]!,
    solution[1]!,
    solution[2]!,
    solution[3]!,
    solution[4]!,
    solution[5]!,
    solution[6]!,
    solution[7]!,
    1,
  ]
  const designNormalization: Mat3 = [
    1 / designScale,
    0,
    -designMeanX / designScale,
    0,
    1 / designScale,
    -designMeanY / designScale,
    0,
    0,
    1,
  ]
  const observedDenormalization: Mat3 = [
    observedScale,
    0,
    observedMeanX,
    0,
    observedScale,
    observedMeanY,
    0,
    0,
    1,
  ]
  const matrix = multiplyMat3(
    observedDenormalization,
    multiplyMat3(normalized, designNormalization),
  )
  if (matrix.some((value) => !Number.isFinite(value))) {
    return undefined
  }
  return matrix
}

export function designResiduals(
  matrix: Mat3,
  correspondences: readonly HomographyCorrespondence[],
): number[] {
  const inverse = invertMat3RowMajor(matrix)
  if (inverse === undefined) {
    return correspondences.map(() => Number.POSITIVE_INFINITY)
  }
  return correspondences.map((correspondence) => {
    const design = applyHomography(inverse, correspondence.observed.x, correspondence.observed.y)
    return Math.hypot(design.x - correspondence.design.x, design.y - correspondence.design.y)
  })
}

function buildFit(
  matrix: Mat3,
  correspondences: readonly HomographyCorrespondence[],
  inlierIndices: readonly number[],
  method: 'trimmed' | 'ransac',
): HomographyFitResult {
  const residuals = designResiduals(matrix, correspondences)
  let sumSquares = 0
  for (const index of inlierIndices) {
    const residual = residuals[index]!
    if (Number.isFinite(residual)) {
      sumSquares += residual * residual
    }
  }
  return {
    matrix,
    inlierIndices: [...inlierIndices].sort((first, second) => first - second),
    residuals,
    rms: inlierIndices.length === 0 ? Number.POSITIVE_INFINITY : Math.sqrt(sumSquares / inlierIndices.length),
    method,
  }
}

export function robustTrimmedHomography(
  correspondences: readonly HomographyCorrespondence[],
  options: RobustHomographyOptions,
): HomographyFitResult | undefined {
  const finite = finiteCorrespondences(correspondences)
  if (finite.length < 4) {
    return undefined
  }
  const trimFactor = options.trimFactor ?? 2.5
  const trimFloor = options.trimFloor ?? Math.max(0.1, options.designSpacing * 0.1)
  const maxRounds = options.maxRounds ?? 6
  let active = finite.map((entry) => entry.index)

  let matrix: Mat3 | undefined
  for (let round = 0; round < maxRounds; round += 1) {
    matrix = fitHomography(active.map((index) => correspondences[index]!))
    if (matrix === undefined) {
      return undefined
    }
    const residuals = designResiduals(matrix, correspondences)
    const activeResiduals = active
      .map((index) => residuals[index]!)
      .filter((residual) => Number.isFinite(residual))
    if (activeResiduals.length < 4) {
      return undefined
    }
    const threshold = Math.max(median(activeResiduals) * trimFactor, trimFloor)
    const next = active.filter((index) => {
      const residual = residuals[index]
      return residual !== undefined && Number.isFinite(residual) && residual <= threshold
    })
    if (next.length < 4) {
      return undefined
    }
    if (next.length === active.length) {
      break
    }
    active = next
  }

  if (matrix === undefined) {
    return undefined
  }
  const minInliers = options.minInliers ?? 6
  if (active.length < minInliers) {
    return undefined
  }
  return buildFit(matrix, correspondences, active, 'trimmed')
}

function sampleIndices(
  indices: readonly number[],
  count: number,
  random: () => number,
): number[] {
  const pool = [...indices]
  const result: number[] = []
  for (let index = 0; index < count && index < pool.length; index += 1) {
    const swapIndex = index + Math.floor(random() * (pool.length - index))
    ;[pool[index], pool[swapIndex]] = [pool[swapIndex]!, pool[index]!]
    result.push(pool[index]!)
  }
  return result
}

export function ransacHomography(
  correspondences: readonly HomographyCorrespondence[],
  options: RobustHomographyOptions,
): HomographyFitResult | undefined {
  const finite = finiteCorrespondences(correspondences)
  if (finite.length < 4) {
    return undefined
  }
  const threshold = Math.max(
    0.05,
    options.designSpacing * (options.inlierFactor ?? 0.3),
  )
  const iterations =
    finite.length <= 16
      ? 2048
      : Math.max(64, options.ransacIterations ?? 256)
  const random = createRandom(options.seed ?? 0x9e3779b9)
  const allIndices = finite.map((entry) => entry.index)
  let bestInliers: number[] = []
  let bestMedian = Number.POSITIVE_INFINITY

  const consider = (sample: readonly number[]): void => {
    const matrix = fitHomography(sample.map((index) => correspondences[index]!))
    if (matrix === undefined) {
      return
    }
    const residuals = designResiduals(matrix, correspondences)
    const inliers: number[] = []
    const inlierResiduals: number[] = []
    for (const index of allIndices) {
      const residual = residuals[index]!
      if (Number.isFinite(residual) && residual <= threshold) {
        inliers.push(index)
        inlierResiduals.push(residual)
      }
    }
    const inlierMedian = median(inlierResiduals)
    if (
      inliers.length > bestInliers.length ||
      (inliers.length === bestInliers.length && inlierMedian < bestMedian)
    ) {
      bestInliers = inliers
      bestMedian = inlierMedian
    }
  }

  if (finite.length <= 16) {
    for (let first = 0; first < finite.length - 3; first += 1) {
      for (let second = first + 1; second < finite.length - 2; second += 1) {
        for (let third = second + 1; third < finite.length - 1; third += 1) {
          for (let fourth = third + 1; fourth < finite.length; fourth += 1) {
            consider([allIndices[first]!, allIndices[second]!, allIndices[third]!, allIndices[fourth]!])
          }
        }
      }
    }
  } else {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      consider(sampleIndices(allIndices, 4, random))
    }
  }

  const minInliers = options.minInliers ?? 6
  if (bestInliers.length < minInliers) {
    return undefined
  }

  // Refit and reclassify on the consensus set.
  let active = bestInliers
  let matrix = fitHomography(active.map((index) => correspondences[index]!))
  if (matrix === undefined) {
    return undefined
  }
  const residuals = designResiduals(matrix, correspondences)
  const reclassified = active.filter((index) => {
    const residual = residuals[index]
    return residual !== undefined && Number.isFinite(residual) && residual <= threshold
  })
  if (reclassified.length >= minInliers) {
    active = reclassified
    matrix = fitHomography(active.map((index) => correspondences[index]!))
    if (matrix === undefined) {
      return undefined
    }
  }
  return buildFit(matrix, correspondences, active, 'ransac')
}

export function robustHomography(
  correspondences: readonly HomographyCorrespondence[],
  options: RobustHomographyOptions,
): HomographyFitResult | undefined {
  const trimmed = robustTrimmedHomography(correspondences, options)
  const minInlierRatio = options.minInlierRatio ?? 0.5
  if (trimmed !== undefined) {
    const ratio =
      correspondences.length === 0 ? 0 : trimmed.inlierIndices.length / correspondences.length
    if (ratio >= minInlierRatio) {
      return trimmed
    }
  }
  const ransac = ransacHomography(correspondences, options)
  if (ransac !== undefined) {
    return ransac
  }
  return trimmed
}

/** Positive determinant of the local Jacobian at the design centroid. */
export function isHomographyOrientationValid(matrix: Mat3, designPoint: Vec2): boolean {
  const [h0, h1, h2, h3, h4, h5, h6, h7, h8] = matrix
  const x = designPoint.x
  const y = designPoint.y
  const w = h6 * x + h7 * y + h8
  if (!(Math.abs(w) > 1e-12)) {
    return false
  }
  const inverseWSquared = 1 / (w * w)
  const numeratorX = h0 * x + h1 * y + h2
  const numeratorY = h3 * x + h4 * y + h5
  const duDx = (h0 * w - numeratorX * h6) * inverseWSquared
  const duDy = (h1 * w - numeratorX * h7) * inverseWSquared
  const dvDx = (h3 * w - numeratorY * h6) * inverseWSquared
  const dvDy = (h4 * w - numeratorY * h7) * inverseWSquared
  const determinant = duDx * dvDy - duDy * dvDx
  return Number.isFinite(determinant) && determinant > 0
}

/** Reject anchor sets that span too little or are nearly collinear. */
export function hasAnchorSpread(points: readonly Vec2[], designSpacing: number): boolean {
  if (points.length < 3 || !(designSpacing > 1e-9)) {
    return false
  }
  let meanX = 0
  let meanY = 0
  for (const point of points) {
    meanX += point.x
    meanY += point.y
  }
  meanX /= points.length
  meanY /= points.length

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let covarianceXX = 0
  let covarianceYY = 0
  let covarianceXY = 0
  for (const point of points) {
    const dx = point.x - meanX
    const dy = point.y - meanY
    covarianceXX += dx * dx
    covarianceYY += dy * dy
    covarianceXY += dx * dy
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }
  covarianceXX /= points.length
  covarianceYY /= points.length
  covarianceXY /= points.length
  const trace = covarianceXX + covarianceYY
  const determinant = covarianceXX * covarianceYY - covarianceXY * covarianceXY
  const discriminant = Math.max(0, trace * trace - 4 * determinant)
  const smallerEigenvalue = (trace - Math.sqrt(discriminant)) / 2
  const diagonal = Math.hypot(maxX - minX, maxY - minY)
  return (
    diagonal >= designSpacing * 3 &&
    smallerEigenvalue >= (designSpacing * 0.2) ** 2
  )
}

export function homographyCentroid(points: readonly Vec2[]): Vec2 {
  if (points.length === 0) {
    return { x: 0, y: 0 }
  }
  let x = 0
  let y = 0
  for (const point of points) {
    x += point.x
    y += point.y
  }
  return { x: x / points.length, y: y / points.length }
}

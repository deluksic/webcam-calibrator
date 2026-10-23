import type { CalibrationFrameObservation, TargetObjectPoint } from '@/lib/calibrationTypes'

export const CALIBRATION_MIN_FRAMES = 3
export const CALIBRATION_MIN_POINTS_PER_FRAME = 8
export const CALIBRATION_MIN_SHARED_POINTS = 8

export type CalibrationInput = {
  objectPoints: [number, number, number][][]
  imagePoints: [number, number][][]
  frameIds: number[]
  sharedDesignIds: number[]
}

export type CalibrationInputResult =
  | { kind: 'ok'; input: CalibrationInput }
  | { kind: 'error'; details: string }

type FramePoints = {
  frameId: number
  points: Map<number, [number, number]>
}

function normalizeFrames(
  targetById: ReadonlyMap<number, TargetObjectPoint>,
  frames: readonly CalibrationFrameObservation[],
): FramePoints[] {
  const valid: FramePoints[] = []
  for (const frame of frames) {
    const points = new Map<number, [number, number]>()
    let duplicate = false
    for (const point of frame.points) {
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !targetById.has(point.designId)) {
        continue
      }
      if (points.has(point.designId)) {
        duplicate = true
        break
      }
      points.set(point.designId, [point.x, point.y])
    }
    if (duplicate || points.size < CALIBRATION_MIN_POINTS_PER_FRAME) {
      continue
    }
    valid.push({ frameId: frame.frameId, points })
  }
  return valid
}

/**
 * Build a single shared object-point template for `calibrateCameraRO`.
 *
 * The solver runs in object-release mode, so every view must share the same
 * row order. Only design ids visible in every usable view are kept.
 */
export function buildCalibrationInput(
  targetPoints: readonly TargetObjectPoint[],
  frames: readonly CalibrationFrameObservation[],
): CalibrationInputResult {
  const targetById = new Map<number, TargetObjectPoint>()
  for (const point of targetPoints) {
    targetById.set(point.designId, point)
  }
  const validFrames = normalizeFrames(targetById, frames)
  if (validFrames.length < CALIBRATION_MIN_FRAMES) {
    return {
      kind: 'error',
      details: `need >=${CALIBRATION_MIN_FRAMES} usable views (have ${validFrames.length})`,
    }
  }

  let shared = [...validFrames[0]!.points.keys()]
  for (let index = 1; index < validFrames.length; index += 1) {
    const frame = validFrames[index]!
    shared = shared.filter((designId) => frame.points.has(designId))
  }
  shared.sort((a, b) => a - b)
  if (shared.length < CALIBRATION_MIN_SHARED_POINTS) {
    return {
      kind: 'error',
      details: `need >=${CALIBRATION_MIN_SHARED_POINTS} shared target points across all views (have ${shared.length})`,
    }
  }

  const objectTemplate: [number, number, number][] = shared.map((designId) => {
    const point = targetById.get(designId)!
    return [point.x, point.y, point.z]
  })
  const objectPoints = validFrames.map(() => objectTemplate)
  const imagePoints = validFrames.map((frame) =>
    shared.map((designId) => {
      const point = frame.points.get(designId)!
      return [point[0], point[1]] as [number, number]
    }),
  )

  return {
    kind: 'ok',
    input: {
      objectPoints,
      imagePoints,
      frameIds: validFrames.map((frame) => frame.frameId),
      sharedDesignIds: shared,
    },
  }
}

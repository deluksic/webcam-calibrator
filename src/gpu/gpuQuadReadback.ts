import type { TgpuRoot } from 'typegpu'

import type { CameraPipeline } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/detectedQuad'
import {
  DECODED_TAG_ID_DICT_MISS,
  DECODED_TAG_ID_UNKNOWN,
  MAX_DETECTED_TAGS,
  type QuadData,
} from '@/gpu/pipelines/gridVizPipeline'
import { quadAreaPx } from '@/lib/calibrationQuality'
import type { Corners } from '@/lib/geometry'
import { rotateRing } from '@/lib/geometry'

const DICT_MISS_U32 = DECODED_TAG_ID_DICT_MISS >>> 0
const UNKNOWN_U32 = DECODED_TAG_ID_UNKNOWN >>> 0

function screenCornersToCorners(quad: QuadData): Corners {
  const c = quad.screenCorners
  return [
    { x: c[0]!.x, y: c[0]!.y },
    { x: c[1]!.x, y: c[1]!.y },
    { x: c[2]!.x, y: c[2]!.y },
    { x: c[3]!.x, y: c[3]!.y },
  ]
}

function quadDataToDetected(quad: QuadData, label: number): DetectedQuad {
  const failureCode = quad.debug.failureCode
  const hasCorners = failureCode === 0
  let corners = screenCornersToCorners(quad)
  const edgePixelCount = quad.debug.edgePixelCount * 100
  const area = quadAreaPx(corners)
  const minX = Math.min(corners[0].x, corners[1].x, corners[2].x, corners[3].x)
  const maxX = Math.max(corners[0].x, corners[1].x, corners[2].x, corners[3].x)
  const minY = Math.min(corners[0].y, corners[1].y, corners[2].y, corners[3].y)
  const maxY = Math.max(corners[0].y, corners[1].y, corners[2].y, corners[3].y)
  const w = maxX - minX
  const h = maxY - minY
  const aspectRatio = h > 0 ? w / h : 1

  const tagU32 = quad.decodedTagId >>> 0
  let decodedTagId: number | undefined
  let decodedRotation: number | undefined
  let vizTagId: number | undefined

  if (tagU32 !== UNKNOWN_U32 && tagU32 !== DICT_MISS_U32) {
    decodedTagId = tagU32
    decodedRotation = quad.decodedRotation
    if (decodedRotation > 0) {
      corners = rotateRing(corners, decodedRotation)
    }
  } else if (tagU32 === DICT_MISS_U32) {
    vizTagId = DICT_MISS_U32
  }

  return {
    corners,
    label,
    count: Math.round(edgePixelCount),
    aspectRatio,
    area,
    pattern: undefined,
    hasCorners,
    cornerDebug: {
      failureCode,
      edgePixelCount,
      minR2: quad.debug.minR2,
      intersectionCount: quad.debug.intersectionCount,
    },
    vizTagId,
    decodedTagId,
    decodedRotation,
    decodedTagKind: decodedTagId !== undefined ? 'tag36h11' : undefined,
  }
}

/** Map GPU quad buffer entries to {@link DetectedQuad} for calibration callbacks. */
export function gpuQuadBufferToDetected(
  quadData: QuadData[],
  quadCount: number,
  sourceLabelIds: number[],
): DetectedQuad[] {
  const n = Math.min(quadCount, MAX_DETECTED_TAGS, quadData.length)
  const quads: DetectedQuad[] = []
  for (let i = 0; i < n; i++) {
    quads.push(quadDataToDetected(quadData[i]!, sourceLabelIds[i] ?? i))
  }
  return quads
}

/** Read GPU tag path results after compute+decode submit. */
export async function readGpuDetection(root: TgpuRoot, pipeline: CameraPipeline): Promise<DetectedQuad[]> {
  await root.device.queue.onSubmittedWorkDone()

  const [quadData, quadCountRaw, sourceLabelIds] = await Promise.all([
    pipeline.grid.quadCornersBuffer.read(),
    pipeline.edgeHistogram.quadCount.read(),
    pipeline.edgeHistogram.quadSourceLabelId.read(),
  ])

  const quadCount = Array.isArray(quadCountRaw) ? (quadCountRaw[0] ?? 0) : Number(quadCountRaw)
  return gpuQuadBufferToDetected(quadData, quadCount, sourceLabelIds)
}

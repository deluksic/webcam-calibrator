import type { TgpuRoot } from 'typegpu'

import type { CameraPipeline } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/detectedQuad'
import {
  DECODED_TAG_ID_DICT_MISS,
  DECODED_TAG_ID_UNKNOWN,
  MAX_DETECTED_TAGS,
} from '@/gpu/pipelines/gridVizPipeline'
import type { HostQuadReadback } from '@/gpu/pipelines/hostQuadReadbackPipeline'
import { quadAreaPx } from '@/lib/calibrationQuality'
import type { Corners } from '@/lib/geometry'
import { TAG_MODULE_CELL, type TagPattern } from '@/lib/tagModuleCell'

const DICT_MISS_U32 = DECODED_TAG_ID_DICT_MISS >>> 0
const UNKNOWN_U32 = DECODED_TAG_ID_UNKNOWN >>> 0
const MODULES_PER_QUAD = 36

/** GPU classify buffer uses 0=black, 1=white, 2=weak, 3=tie. */
function gpuPatternCellToHost(cell: number): TagPattern[number] {
  if (cell === 1) return TAG_MODULE_CELL.white
  if (cell === 2) return TAG_MODULE_CELL.weak
  if (cell === 3) return TAG_MODULE_CELL.tie
  return TAG_MODULE_CELL.black
}

function patternFromGpuBuffer(flat: number[], quadIndex: number): TagPattern {
  const base = quadIndex * MODULES_PER_QUAD
  const pattern = [] as unknown as TagPattern
  for (let i = 0; i < MODULES_PER_QUAD; i++) {
    pattern[i] = gpuPatternCellToHost(flat[base + i] ?? 0)
  }
  return pattern
}

async function readU32Prefix(device: GPUDevice, gpuBuffer: GPUBuffer, count: number): Promise<number[]> {
  if (count <= 0) {
    return []
  }
  const byteSize = count * 4
  const staging = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(gpuBuffer, 0, staging, 0, byteSize)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ, 0, byteSize)
  const mapped = staging.getMappedRange(0, byteSize)
  const ids = Array.from(new Uint32Array(mapped.slice(0)))
  staging.unmap()
  staging.destroy()
  return ids
}

function screenCornersToCorners(quad: Pick<HostQuadReadback, 'screenCorners'>): Corners {
  const c = quad.screenCorners
  return [
    { x: c[0]!.x, y: c[0]!.y },
    { x: c[1]!.x, y: c[1]!.y },
    { x: c[2]!.x, y: c[2]!.y },
    { x: c[3]!.x, y: c[3]!.y },
  ]
}

function hostQuadToDetected(quad: HostQuadReadback, label: number, pattern?: TagPattern): DetectedQuad {
  const failureCode = quad.debug.failureCode
  const hasCorners = failureCode === 0
  const corners = screenCornersToCorners(quad)
  const edgePixelCount = quad.debug.edgePixelCount
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
    decodedRotation = 0
  } else if (tagU32 === DICT_MISS_U32) {
    vizTagId = DICT_MISS_U32
  }

  return {
    corners,
    label,
    count: Math.round(edgePixelCount),
    aspectRatio,
    area,
    pattern,
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

/** Map host readback entries to {@link DetectedQuad} for calibration callbacks. */
export function hostQuadsToDetected(
  hostQuads: HostQuadReadback[],
  quadCount: number,
  sourceLabelIds: number[],
  patternFlat?: number[],
): DetectedQuad[] {
  const n = Math.min(quadCount, MAX_DETECTED_TAGS, hostQuads.length)
  const quads: DetectedQuad[] = []
  for (let i = 0; i < n; i++) {
    const pattern = patternFlat !== undefined ? patternFromGpuBuffer(patternFlat, i) : undefined
    quads.push(hostQuadToDetected(hostQuads[i]!, sourceLabelIds[i] ?? i, pattern))
  }
  return quads
}

/**
 * Read GPU tag results after compute+decode+pack.
 * Reads {@link HostQuadReadback} (no homography mat3x3f) — not `quadCornersBuffer`.
 */
export async function readGpuDetection(
  root: TgpuRoot,
  pipeline: CameraPipeline,
): Promise<{ quads: DetectedQuad[]; quadCount: number }> {
  const quadCountRaw = await pipeline.edgeHistogram.quadCount.read()
  const quadCount = Array.isArray(quadCountRaw) ? (quadCountRaw[0] ?? 0) : Number(quadCountRaw)
  const n = Math.min(quadCount, MAX_DETECTED_TAGS)

  if (n <= 0) {
    return { quads: [], quadCount: 0 }
  }

  const patternU32Count = n * MODULES_PER_QUAD

  const [allHostQuads, patternRaw, sourceLabelIds] = await Promise.all([
    pipeline.hostQuadReadback.hostQuadReadbackBuffer.read(),
    patternU32Count > 0
      ? readU32Prefix(root.device, pipeline.tagDecode.patternBuf.buffer, patternU32Count)
      : Promise.resolve([]),
    readU32Prefix(root.device, pipeline.edgeHistogram.quadSourceLabelId.buffer, n),
  ])

  const hostQuads = (Array.isArray(allHostQuads) ? allHostQuads : []).slice(0, n)
  const patternFlat = patternRaw.map((v) => Number(v))

  return {
    quads: hostQuadsToDetected(hostQuads, n, sourceLabelIds, patternFlat),
    quadCount,
  }
}

import type { TgpuRoot } from 'typegpu'

import type { CameraPipeline } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/detectedQuad'
import { MAX_DETECTED_TAGS } from '@/gpu/pipelines/gridVizPipeline'
import type { HostQuadReadback } from '@/gpu/pipelines/hostQuadReadbackPipeline'
import { quadAreaPx } from '@/lib/calibrationQuality'
import type { Corners } from '@/lib/geometry'
import { TAG_MODULE_CELL, type TagPattern } from '@/lib/tagModuleCell'
const MODULES_PER_QUAD = 36

/** GPU classify buffer uses i32: 0=black, 1=white, -1=weak, -2=tie.
 *  Read as u32 then convert to signed via |0. */
function gpuPatternCellToHost(cell: number): TagPattern[number] {
  const s = (cell | 0)
  if (s === 1) return TAG_MODULE_CELL.white
  if (s === -1) return TAG_MODULE_CELL.weak
  if (s === -2) return TAG_MODULE_CELL.tie
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

  const kind = quad.tagKind
  let decodedTagId: number | undefined
  let decodedRotation: number | undefined
  let decodedTagKind: 'tag36h11' | 'custom' | 'clean' | undefined

  if (kind === 1) {
    decodedTagId = quad.decodedTagId // i32: >=0 standard, <0 custom
    decodedRotation = 0
    decodedTagKind = decodedTagId < 0 ? 'custom' : 'tag36h11'
  } else if (kind === 2) {
    decodedTagKind = 'clean'
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
    vizTagId: undefined,
    decodedTagId,
    decodedRotation,
    decodedTagKind,
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
 * When `readGray` is true, also reads the gray buffer from the same GPU submission
 * to guarantee no frame drift between corner data and pixel values.
 */
export async function readGpuDetection(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  readGray?: boolean,
): Promise<{ quads: DetectedQuad[]; quadCount: number; grayData: Float32Array }> {
  // All reads fire in one Promise.all — every copy command is submitted to the GPU
  // queue before any of them resolve. This avoids a two-phase race where quadCount
  // and quad data came from different frames.
  const maxPattern = MAX_DETECTED_TAGS * MODULES_PER_QUAD

  const grayPromise = readGray ? pipeline.gray.buffer.read() : Promise.resolve([] as number[])

  const [quadCountRaw, allHostQuads, patternRaw, sourceLabelIds, grayRaw] = await Promise.all([
    pipeline.edgeHistogram.quadCount.read(),
    pipeline.hostQuadReadback.hostQuadReadbackBuffer.read(),
    readU32Prefix(root.device, pipeline.tagDecode.patternBuf.buffer, maxPattern),
    readU32Prefix(root.device, pipeline.edgeHistogram.quadSourceLabelId.buffer, MAX_DETECTED_TAGS),
    grayPromise,
  ])

  const quadCount = Array.isArray(quadCountRaw) ? (quadCountRaw[0] ?? 0) : Number(quadCountRaw)
  const n = Math.min(quadCount, MAX_DETECTED_TAGS)

  if (n <= 0) {
    return { quads: [], quadCount: 0, grayData: new Float32Array(0) }
  }

  const hostQuads = (Array.isArray(allHostQuads) ? allHostQuads : []).slice(0, n)
  const patternFlat = patternRaw.slice(0, n * MODULES_PER_QUAD).map((v) => Number(v))

  const grayData = Array.isArray(grayRaw) ? new Float32Array(grayRaw) : new Float32Array(0)

  return {
    quads: hostQuadsToDetected(hostQuads, n, sourceLabelIds.slice(0, n), patternFlat),
    quadCount,
    grayData,
  }
}

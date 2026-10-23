import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import { DEFAULT_TARGET_CONFIG } from '@/lib/hexTarget/config'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { CalibrationOk } from '@/workers/calibration.worker'

function floatGrayToPngBlob(gray: Float32Array, width: number, height: number): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  const imageData = ctx.createImageData(width, height)
  const data = imageData.data
  for (let i = 0; i < gray.length; i++) {
    const v = Math.round(Math.max(0, Math.min(1, gray[i] ?? 0)) * 255)
    const j = i * 4
    data[j] = v
    data[j + 1] = v
    data[j + 2] = v
    data[j + 3] = 255
  }
  ctx.putImageData(imageData, 0, 0)
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b!), 'image/png'))
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  queueMicrotask(() => URL.revokeObjectURL(url))
}

function downloadJson(obj: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(obj, undefined, 2)], { type: 'application/json' })
  downloadBlob(blob, filename)
}

function hexExport(
  calibration: CalibrationOk,
  observations: CalibrationOk['observations'] = [],
  frameImages: Record<string, string> = {},
) {
  const config = calibration.targetConfig ?? DEFAULT_TARGET_CONFIG
  const pattern = generateHexPattern({
    columns: config.columns,
    rows: config.rows,
    pointsPerEdge: config.pointsPerEdge,
  })
  const colors = randomBinaryColors(pattern.points.length, config.colorSeed)
  const { updatedTargetPoints, targetConfig: _targetConfig, ...rest } = calibration
  void _targetConfig
  return {
    schemaVersion: 2,
    ...rest,
    target: {
      kind: 'hex',
      columns: config.columns,
      rows: config.rows,
      pointsPerEdge: config.pointsPerEdge,
      colorSeed: config.colorSeed,
      markerScale: config.markerScale,
      identityDepth: config.identityDepth,
      widthPx: config.widthPx ?? calibration.imageSize.width,
      heightPx: config.heightPx ?? calibration.imageSize.height,
      objectUnits: 'median-design-spacing',
      colors,
      markers: updatedTargetPoints.map((point) => ({
        designId: point.designId,
        x: point.x,
        y: point.y,
        z: point.z,
      })),
    },
    frameImages,
    observations,
  }
}

export function downloadCalibrationOkJson(c: CalibrationOk) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  downloadJson(hexExport(c), `calibration-ok-${stamp}.json`)
}

/**
 * Enriches the calibration result with target metadata, frame image references,
 * and per-frame point observations; downloads JSON plus one PNG per frame.
 */
export async function downloadCalibrationOkWithImages(
  c: CalibrationOk,
  framePool: CalibrationFrameObservation[],
) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const imageFilePrefix = (frameId: number) => `frame_${frameId}.png`

  const frameImages: Record<string, string> = {}
  const observations: NonNullable<CalibrationOk['observations']> = []
  const pngDownloads: Promise<void>[] = []

  for (const frame of framePool) {
    const points = frame.points.map((point) => ({
      designId: point.designId,
      x: point.x,
      y: point.y,
    }))
    if (!frame.grayData || !frame.imageWidth || !frame.imageHeight) {
      observations.push({ frameId: frame.frameId, imageFile: '', points })
      continue
    }
    const filename = imageFilePrefix(frame.frameId)
    frameImages[String(frame.frameId)] = filename
    observations.push({ frameId: frame.frameId, imageFile: filename, points })
    pngDownloads.push(
      floatGrayToPngBlob(frame.grayData, frame.imageWidth, frame.imageHeight).then((blob) =>
        downloadBlob(blob, filename),
      ),
    )
  }

  downloadJson(hexExport(c, observations, frameImages), `calibration-ok-${stamp}.json`)

  for (let i = 0; i < pngDownloads.length; i++) {
    await new Promise((resolve) => setTimeout(resolve, 100))
    await pngDownloads[i]
  }
}

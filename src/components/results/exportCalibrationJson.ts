import { strToU8, zipSync } from 'fflate'

import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import type { CalibrationOk } from '@/workers/calibration.worker'

function floatGrayToPngBytes(
  gray: Float32Array,
  width: number,
  height: number,
): Promise<Uint8Array> {
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
  return new Promise((resolve) =>
    canvas.toBlob((b) => {
      if (!b) { resolve(new Uint8Array(0)); return }
      b.arrayBuffer().then((buf) => resolve(new Uint8Array(buf)))
    }, 'image/png'),
  )
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  queueMicrotask(() => URL.revokeObjectURL(url))
}

export function downloadCalibrationOkJson(c: CalibrationOk) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const blob = new Blob([JSON.stringify(c, undefined, 2)], { type: 'application/json' })
  downloadBlob(blob, `calibration-ok-${stamp}.json`)
}

/**
 * Enriches the calibration result with frame image references and downloads
 * a single ZIP file containing the JSON plus a PNG for each frame with gray data.
 */
export async function downloadCalibrationOkWithImages(
  c: CalibrationOk,
  framePool: CalibrationFrameObservation[],
) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const imageFilePrefix = (frameId: number) => `frame_${frameId}.png`

  const frameImages: Record<string, string> = {}
  const observations: CalibrationOk['observations'] = []

  // Build zip entries: { filename: Uint8Array }
  const zipEntries: Record<string, Uint8Array> = {}

  for (const frame of framePool) {
    if (!frame.grayData || !frame.imageWidth || !frame.imageHeight) {
      observations.push({
        frameId: frame.frameId,
        imageFile: '',
        tags: frame.tags.map((t) => ({
          tagId: t.tagId,
          corners: [t.corners[0], t.corners[1], t.corners[2], t.corners[3]],
        })),
      })
      continue
    }

    const filename = imageFilePrefix(frame.frameId)
    frameImages[String(frame.frameId)] = filename

    observations.push({
      frameId: frame.frameId,
      imageFile: filename,
      tags: frame.tags.map((t) => ({
        tagId: t.tagId,
        corners: [t.corners[0], t.corners[1], t.corners[2], t.corners[3]],
      })),
    })

    const pngBytes = await floatGrayToPngBytes(
      frame.grayData, frame.imageWidth, frame.imageHeight,
    )
    zipEntries[filename] = pngBytes
  }

  const enriched: CalibrationOk = { ...c, frameImages, observations }
  zipEntries[`calibration-ok-${stamp}.json`] = strToU8(JSON.stringify(enriched, undefined, 2))

  const zipBytes = zipSync(zipEntries)
  downloadBlob(new Blob([zipBytes], { type: 'application/zip' }), `calibration-ok-${stamp}.zip`)
}

import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import type { CalibrationOk } from '@/workers/calibration.worker'

function floatGrayToPngBlob(
  gray: Float32Array,
  width: number,
  height: number,
): Promise<Blob> {
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

export function downloadCalibrationOkJson(c: CalibrationOk) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  downloadJson(c, `calibration-ok-${stamp}.json`)
}

/**
 * Enriches the calibration result with frame image references and downloads
 * the JSON plus PNG files for each frame that has gray data.
 */
export async function downloadCalibrationOkWithImages(
  c: CalibrationOk,
  framePool: CalibrationFrameObservation[],
) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const imageFilePrefix = (frameId: number) => `frame_${frameId}.png`

  const frameImages: Record<string, string> = {}
  const observations: CalibrationOk['observations'] = []

  const pngDownloads: Promise<void>[] = []

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

    pngDownloads.push(
      floatGrayToPngBlob(frame.grayData, frame.imageWidth, frame.imageHeight).then(
        (blob) => downloadBlob(blob, filename),
      ),
    )
  }

  const enriched: CalibrationOk = { ...c, frameImages, observations }
  downloadJson(enriched, `calibration-ok-${stamp}.json`)

  // Download PNGs after a short delay so the browser doesn't block them
  for (let i = 0; i < pngDownloads.length; i++) {
    await new Promise((r) => setTimeout(r, 100))
    await pngDownloads[i]
  }
}

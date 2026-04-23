/**
 * Writes a synthetic raycast AprilTag (tag36h11) to output/april-tag-raycast.png
 * for visual inspection. Run: pnpm run render:april-tag
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { PNG } from 'pngjs'

import type { Corners } from '@/lib/geometry'
import { codeToPattern, tag36h11Code, TAG36H11_CODES } from '@/lib/tag36h11'
import { renderAprilTagIntensity } from '@/tests/utils/syntheticAprilTag'

const { min, max, round } = Math
const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..')

function main() {
  const raw = process.argv[2]
  let tagId = 0
  if (raw !== undefined) {
    tagId = Number(raw)
    if (!Number.isInteger(tagId) || tagId < 0 || tagId >= TAG36H11_CODES.length) {
      console.error(`Usage: pnpm run render:april-tag [tagId 0-${TAG36H11_CODES.length - 1}]`)
      process.exit(1)
    }
  }

  const pattern = codeToPattern(tag36h11Code(tagId))
  const size = 360
  const margin = 20
  const strip: Corners = [
    { x: margin, y: margin },
    { x: margin + size, y: margin },
    { x: margin, y: margin + size },
    { x: margin + size, y: margin + size },
  ]

  // Mild perspective (same spirit as grid tests)
  const perspective: Corners = [
    { x: 40, y: 30 },
    { x: 380, y: 20 },
    { x: 30, y: 360 },
    { x: 370, y: 370 },
  ]

  const w = 420
  const h = 400

  const supersample = 4

  const intensityAxis = renderAprilTagIntensity({
    width: w,
    height: h,
    corners: strip,
    pattern,
    supersample,
  })
  const intensityPersp = renderAprilTagIntensity({
    width: w,
    height: h,
    corners: perspective,
    pattern,
    supersample,
  })

  const outDir = join(repoRoot, 'output')
  mkdirSync(outDir, { recursive: true })

  writeGreyPng(join(outDir, `april-tag-raycast-id${tagId}-axis.png`), w, h, intensityAxis)
  writeGreyPng(join(outDir, `april-tag-raycast-id${tagId}-perspective.png`), w, h, intensityPersp)

  console.log(
    `Wrote ${join(outDir, `april-tag-raycast-id${tagId}-axis.png`)} and ...-perspective.png (tag id ${tagId}, ${supersample}× supersample)`,
  )
}

function writeGreyPng(filePath: string, width: number, height: number, intensity: Float32Array): void {
  const data = Buffer.alloc(width * height)
  for (let i = 0; i < intensity.length; i++) {
    const v = round(min(255, max(0, intensity[i]! * 255)))
    data[i] = v
  }

  const pngBuffer = PNG.sync.write(
    { width, height, data },
    {
      colorType: 0,
      inputColorType: 0,
      inputHasAlpha: false,
      bitDepth: 8,
    },
  )
  writeFileSync(filePath, pngBuffer)
}

main()

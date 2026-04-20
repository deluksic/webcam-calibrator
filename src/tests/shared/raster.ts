import { codeToPattern, TAG36H11_CODES } from '@/lib/tag36h11'
import type { TagPattern } from '@/lib/tag36h11'
import type { StripCorners } from '@/tests/shared/types'
import { finiteDifferenceSobelFromIntensity, renderAprilTagIntensity } from '@/tests/utils/syntheticAprilTag'

export interface RasterFromStripOptions {
  width: number
  height: number
  strip: StripCorners
  pattern: TagPattern
  supersample: number
}

export function rasterAprilTagGrayscaleSobel(opts: RasterFromStripOptions): {
  grayscale: Float32Array
  sobel: Float32Array
} {
  const grayscale = renderAprilTagIntensity({
    width: opts.width,
    height: opts.height,
    corners: opts.strip,
    pattern: opts.pattern,
    supersample: opts.supersample,
  })
  const sobel = finiteDifferenceSobelFromIntensity(grayscale, opts.width, opts.height, {
    gradientScale: 4,
  })
  return { grayscale, sobel }
}

export function patternForTagId(tagId: number): TagPattern {
  return codeToPattern(TAG36H11_CODES[tagId]!)
}

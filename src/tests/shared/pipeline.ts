import { applyRadialDistortion01 } from '@/tests/shared/distortion/radial'
import { applyAdditiveGaussian01, applySaltPepper01, applySpeckle01 } from '@/tests/shared/noise/index'
import { patternForTagId, rasterAprilTagGrayscaleSobel } from '@/tests/shared/raster'
import { buildGroundTruthStrip } from '@/tests/shared/scenes'
import type { BuildImageOptions, RasterPack } from '@/tests/shared/types'
import { finiteDifferenceSobelFromIntensity } from '@/tests/utils/syntheticAprilTag'

/**
 * Full image formation: scene → raster → optional radial distortion → noise chain → Sobel from final intensity.
 */
export function buildRasterPack(opts: BuildImageOptions): RasterPack {
  const { scene, noise = [], radialDistortion } = opts
  const strip = buildGroundTruthStrip(scene)
  const pattern = patternForTagId(scene.tagId)
  let grayscale = rasterAprilTagGrayscaleSobel({
    width: scene.width,
    height: scene.height,
    strip,
    pattern,
    supersample: scene.supersample,
  }).grayscale

  if (radialDistortion) {
    grayscale = applyRadialDistortion01(grayscale, scene.width, scene.height, radialDistortion)
  }

  for (const op of noise) {
    const g = grayscale
    switch (op.type) {
      case 'speckle':
        applySpeckle01(g, op.amplitude, op.seed)
        break
      case 'gaussian':
        applyAdditiveGaussian01(g, op.sigma, op.seed)
        break
      case 'saltPepper':
        applySaltPepper01(g, op.rate, op.seed)
        break
      default:
        break
    }
  }

  const sobel = finiteDifferenceSobelFromIntensity(grayscale, scene.width, scene.height, {
    gradientScale: 4,
  })
  return {
    width: scene.width,
    height: scene.height,
    groundTruthStrip: strip,
    grayscale,
    sobel,
  }
}

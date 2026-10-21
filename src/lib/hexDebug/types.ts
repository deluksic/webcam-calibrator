export type HexDebugRejection = 'accepted' | 'fill' | 'ring' | 'contrast' | 'noRing'

export type HexDebugTuning = {
  baseTolerance: number
  z: number
  maxTolerance: number
  minArea: number
  maxArea: number
  minFillPercent: number
  minContrast: number
  maxRingVariance: number
  ringMargin: number
  iterations: number
}

export type HexDebugComponent = {
  label: number
  area: number
  meanGray: number
  variance: number
  bbox: { minX: number; maxX: number; minY: number; maxY: number }
  centroid: { x: number; y: number }
  color: 0 | 1
  fillRatio: number
  ringMean: number
  ringVariance: number
  rejection: HexDebugRejection
}

export type HexDebugFrame = {
  width: number
  height: number
  blockSize: number
  components: HexDebugComponent[]
  iterations: number
  elapsedMs: number
}

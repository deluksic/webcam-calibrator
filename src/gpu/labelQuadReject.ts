/** Per compact `labelId`: why `compactQuads` did not register a quad (0 = registered). */

export const LabelQuadRejectCode = {
  registered: 0,
  /** Fewer or more than four orientation histogram peaks. */
  peaks: 1,
  /** Not all four peaks have a valid TLS line with enough inliers. */
  lineFit: 2,
  /** Four peaks but only two distinct orientations (parallel pairs). */
  parallel: 3,
  /** `MAX_QUADS` registration cap hit. */
  cap: 4,
} as const

export type LabelQuadRejectCodeValue = (typeof LabelQuadRejectCode)[keyof typeof LabelQuadRejectCode]

export const LABEL_QUAD_REJECT_LEGEND: ReadonlyArray<{
  code: LabelQuadRejectCodeValue
  label: string
  color: string
}> = [
  { code: LabelQuadRejectCode.peaks, label: 'Peak count ≠ 4', color: '#aa55ff' },
  { code: LabelQuadRejectCode.lineFit, label: 'Line fit / inliers', color: '#4488ff' },
  { code: LabelQuadRejectCode.parallel, label: 'Parallel edge pairs', color: '#ff9933' },
  { code: LabelQuadRejectCode.cap, label: 'Quad cap', color: '#ff4444' },
]

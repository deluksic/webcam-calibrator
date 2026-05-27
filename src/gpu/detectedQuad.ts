import type { Corners } from '@/lib/geometry'
import type { TagPattern } from '@/lib/tagModuleCell'

/** Compact label sentinel — must match GPU shaders and pointer-jump output. */
export const COMPONENT_LABEL_INVALID = 0xffff_ffff

export type DecodedTagKind = 'tag36h11' | 'custom' | 'clean'

export interface CornerDebugInfo {
  failureCode: number
  edgePixelCount: number
  minR2: number
  intersectionCount: number
}

/** Host-side quad from GPU detection readback (grid / Calibrate). */
export interface DetectedQuad {
  corners: Corners
  label: number
  count: number
  aspectRatio: number
  area: number
  pattern: TagPattern | undefined
  /** True when GPU corner pipeline succeeded (`failureCode === 0`). */
  hasCorners: boolean
  cornerDebug: CornerDebugInfo | undefined
  /** Deprecated; no longer used (replaced by `tagKind` discriminant). */
  vizTagId?: number
  decodedTagId?: number
  decodedRotation?: number
  decodedTagKind?: DecodedTagKind
}

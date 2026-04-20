import { decodeStressStripWithHomographyMismatchOffsetsPx } from '@/lib/decodeStressHarness'
import type { Point } from '@/lib/geometry'
import { xorshift32State, xorshift32U01 } from '@/tests/shared/rng'
import type { StripCorners } from '@/tests/shared/types'

export type InitialSpec =
  | { kind: 'mismatchTemplate'; scale: number }
  | { kind: 'cornerJitter'; sigmaPx: number; seed: number }

export function applyInitialRoughStrip(gtStrip: StripCorners, spec: InitialSpec): StripCorners {
  if (spec.kind === 'mismatchTemplate') {
    return decodeStressStripWithHomographyMismatchOffsetsPx(gtStrip, spec.scale)
  }
  const st = xorshift32State(spec.seed)
  const out: Point[] = []
  for (const p of gtStrip) {
    const jx = Math.round((xorshift32U01(st) * 2 - 1) * spec.sigmaPx)
    const jy = Math.round((xorshift32U01(st) * 2 - 1) * spec.sigmaPx)
    out.push({ x: p.x + jx, y: p.y + jy })
  }
  return out as StripCorners
}

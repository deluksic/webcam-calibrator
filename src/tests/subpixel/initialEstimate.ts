import { decodeStressStripWithHomographyMismatchOffsetsPx } from '@/lib/decodeStressHarness'
import type { Corners, Point } from '@/lib/geometry'
import { xorshift32State, xorshift32U01 } from '@/tests/shared/rng'

const { round } = Math

export type InitialSpec =
  | { kind: 'mismatchTemplate'; scale: number }
  | { kind: 'cornerJitter'; sigmaPx: number; seed: number }

export function applyInitialRoughStrip(gtStrip: Corners, spec: InitialSpec): Corners {
  if (spec.kind === 'mismatchTemplate') {
    return decodeStressStripWithHomographyMismatchOffsetsPx(gtStrip, spec.scale)
  }
  const st = xorshift32State(spec.seed)
  const jitter = (p: Point): Point => ({
    x: p.x + round((xorshift32U01(st) * 2 - 1) * spec.sigmaPx),
    y: p.y + round((xorshift32U01(st) * 2 - 1) * spec.sigmaPx),
  })
  return [jitter(gtStrip[0]), jitter(gtStrip[1]), jitter(gtStrip[2]), jitter(gtStrip[3])]
}

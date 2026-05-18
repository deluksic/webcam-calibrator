import type { Corners } from '@/lib/geometry'

const { abs, max, min } = Math

/** Strong-perspective quad in ~320px space (same family as aprilTagRaycast tests). */
const REF_STRIP: Corners = [
  { x: 20, y: 20 },
  { x: 280, y: 45 },
  { x: 35, y: 260 },
  { x: 275, y: 265 },
]

const REF_CX = (20 + 280 + 35 + 275) / 4
const REF_CY = (20 + 45 + 260 + 265) / 4

/** Normalized image-space Δpx per corner (TL, TR, BL, BR) for subpixel mismatch cases. */
export const HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX: Corners = [
  { x: 0.35, y: -0.2 },
  { x: -0.25, y: 0.25 },
  { x: 0.2, y: 0.3 },
  { x: -0.3, y: -0.25 },
]

/** Axis-aligned square strip inside the image. */
export function axisAlignedStrip(w: number, h: number, margin: number, side: number): Corners {
  const x0 = margin
  const y0 = margin
  return [
    { x: x0, y: y0 },
    { x: x0 + side, y: y0 },
    { x: x0, y: y0 + side },
    { x: x0 + side, y: y0 + side },
  ]
}

/** Scale + center REF_STRIP into `[margin, w-margin] × [margin, h-margin]`. */
export function fitPerspectiveStrip(
  w: number,
  h: number,
  opts?: { margin?: number; perspectiveBoost?: number },
): Corners {
  const margin = opts?.margin ?? 6
  const boost = opts?.perspectiveBoost ?? 1
  const cw = w - 2 * margin
  const ch = h - 2 * margin
  const spanX =
    max(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x) -
    min(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x)
  const spanY =
    max(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y) -
    min(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y)
  const s = min(cw / spanX, ch / spanY) * boost
  const cx = margin + cw / 2
  const cy = margin + ch / 2
  return REF_STRIP.map((p) => ({
    x: cx + (p.x - REF_CX) * s,
    y: cy + (p.y - REF_CY) * s,
  })) as Corners
}

/** Rough strip = ground truth + `scale *` template offsets (recompute H via `buildTagGrid`). */
export function stripWithHomographyMismatchOffsetsPx(rasterStrip: Corners, scale: number): Corners {
  return rasterStrip.map((p, i) => ({
    x: p.x + HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.x * scale,
    y: p.y + HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.y * scale,
  })) as Corners
}

/** `max |Δx|, |Δy|` over the mismatch template (pixels before scale). */
export function homographyMismatchTemplateMaxAxisPx(): number {
  let m = 0
  for (const p of HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX) {
    m = max(m, abs(p.x), abs(p.y))
  }
  return m
}

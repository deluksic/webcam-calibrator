import { describe, expect, it } from 'vitest'

import { pointerToFrame } from '@/lib/roiLabPointer'

const rect = { left: 10, top: 20, width: 640, height: 480 }

describe('pointerToFrame', () => {
  it('maps the center of an aspect-matched canvas', () => {
    const point = pointerToFrame(330, 260, rect, 640, 480, 1, 0, 0)
    expect(point).toBeDefined()
    expect(point!.canvasX).toBeCloseTo(320, 6)
    expect(point!.canvasY).toBeCloseTo(240, 6)
    expect(point!.frameX).toBeCloseTo(320, 6)
    expect(point!.frameY).toBeCloseTo(240, 6)
  })

  it('removes letterbox bars before normalizing', () => {
    // 1280x720 content inside a 900x600 box: 900x506.25 content, 46.875 px bars.
    const box = { left: 0, top: 0, width: 900, height: 600 }
    const center = pointerToFrame(450, 300, box, 1280, 720, 1, 0, 0)
    expect(center).toBeDefined()
    expect(center!.frameX).toBeCloseTo(640, 6)
    expect(center!.frameY).toBeCloseTo(360, 6)

    const contentTopLeft = pointerToFrame(0, 46.875, box, 1280, 720, 1, 0, 0)
    expect(contentTopLeft).toBeDefined()
    expect(contentTopLeft!.frameX).toBeCloseTo(0, 6)
    expect(contentTopLeft!.frameY).toBeCloseTo(0, 6)

    expect(pointerToFrame(450, 10, box, 1280, 720, 1, 0, 0)).toBeUndefined()
  })

  it('removes pillarbox bars before normalizing', () => {
    const box = { left: 0, top: 0, width: 600, height: 600 }
    const center = pointerToFrame(300, 300, box, 640, 480, 1, 0, 0)
    expect(center).toBeDefined()
    expect(center!.frameX).toBeCloseTo(320, 6)
    expect(center!.frameY).toBeCloseTo(240, 6)
  })

  it('inverts zoom and pan after normalizing', () => {
    const point = pointerToFrame(330, 260, rect, 640, 480, 2, 100, 50)
    expect(point).toBeDefined()
    expect(point!.frameX).toBeCloseTo(110, 6)
    expect(point!.frameY).toBeCloseTo(95, 6)
  })

  it('rejects clicks outside the canvas content', () => {
    expect(pointerToFrame(0, 0, rect, 640, 480, 1, 0, 0)).toBeUndefined()
    expect(pointerToFrame(700, 300, rect, 640, 480, 1, 0, 0)).toBeUndefined()
  })
})

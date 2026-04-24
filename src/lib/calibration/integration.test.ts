/**
 * Integration test: Zhang calibration on real calibration data.
 *
 * This test loads calibration data from JSON export and:
 * 1. Parses detected corners into object/image point correspondences
 * 2. Computes homographies for each frame
 * 3. Runs Zhang's calibration to extract K
 * 4. Compares against the existing tsCalibResult
 */

import { describe, expect, test } from 'vitest'
import { computeHomography, applyHomography, Vec2 } from './dltHomography'
import { solveIntrinsicsFromHomographies } from '../zhangCalibration'
import type { Mat3 } from './mat3'
import { readFileSync } from 'fs'

interface LayoutTag {
  tagId: number
  corners: [number, number][]
}

interface CalFrame {
  frameId: number
  framePoints: Array<{
    pointId: number
    imagePoint: [number, number]
  }>
}

// Parse pointId: format is TTCCCCC where TT=tagId, CCCCC=cornerIndex (0-3)
function parsePointId(pointId: number): { tagId: number; corner: number } {
  const tagId = Math.floor(pointId / 10000)
  const corner = pointId % 10000
  return { tagId, corner }
}

// Group points by tag, using actual layout coordinates
function groupByTag(frame: CalFrame, layout: LayoutTag[]): Map<number, { obj: Vec2[], img: Vec2[] }> {
  const layoutMap = new Map<number, [number, number][]>()
  for (const tag of layout) {
    layoutMap.set(tag.tagId, tag.corners as [number, number][])
  }

  const tagImagePts = new Map<number, Map<number, Vec2>>()
  for (const pt of frame.framePoints) {
    const { tagId, corner } = parsePointId(pt.pointId)
    if (corner >= 0 && corner <= 3) {
      if (!tagImagePts.has(tagId)) tagImagePts.set(tagId, new Map())
      tagImagePts.get(tagId)!.set(corner, { x: pt.imagePoint[0], y: pt.imagePoint[1] })
    }
  }

  const completeTags = new Map<number, { obj: Vec2[], img: Vec2[] }>()
  for (const [tagId, imagePts] of tagImagePts) {
    const layoutCorners = layoutMap.get(tagId)
    if (!layoutCorners) continue

    const obj: Vec2[] = []
    const img: Vec2[] = []
    for (let i = 0; i < 4; i++) {
      const imgPt = imagePts.get(i)
      if (!imgPt) break
      obj.push({ x: layoutCorners[i]![0], y: layoutCorners[i]![1] })
      img.push(imgPt)
    }

    if (obj.length === 4) completeTags.set(tagId, { obj, img })
  }

  return completeTags
}

function computeReprojectionError(H: Mat3, src: Vec2[], dst: Vec2[]): number {
  let totalError = 0
  for (let i = 0; i < src.length; i++) {
    const projected = applyHomography(H, src[i]!)
    const dx = projected.x - dst[i]!.x
    const dy = projected.y - dst[i]!.y
    totalError += dx * dx + dy * dy
  }
  return Math.sqrt(totalError / src.length)
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Zhang calibration on real data', () => {
  test('computes K matching existing calibration result', () => {
    const data = JSON.parse(readFileSync('/calibration_export_1777037266230.json', 'utf8'))

    expect(data.resolution).toBeDefined()
    expect(data.calibrationFrames.length).toBeGreaterThanOrEqual(3)

    const homographies: Mat3[] = []

    for (const frame of data.calibrationFrames) {
      const tagPoints = groupByTag(frame as CalFrame, data.layout as LayoutTag[])
      if (tagPoints.size === 0) continue

      const allSrc: Vec2[] = []
      const allDst: Vec2[] = []
      for (const { obj, img } of tagPoints.values()) {
        allSrc.push(...obj)
        allDst.push(...img)
      }

      if (allSrc.length < 4) continue

      const H = computeHomography(allSrc, allDst)
      homographies.push(H)
    }

    expect(homographies.length).toBeGreaterThanOrEqual(3)

    const intrinsics = solveIntrinsicsFromHomographies(homographies)
    expect(intrinsics).toBeDefined()

    if (data.tsCalibResult) {
      const existing = data.tsCalibResult.K

      // K should match within 0.01% relative error
      const fxErr = Math.abs(intrinsics!.fx - existing.fx) / existing.fx
      const fyErr = Math.abs(intrinsics!.fy - existing.fy) / existing.fy
      const cxErr = Math.abs(intrinsics!.cx - existing.cx) / existing.cx
      const cyErr = Math.abs(intrinsics!.cy - existing.cy) / existing.cy

      expect(fxErr).toBeLessThan(0.0001)
      expect(fyErr).toBeLessThan(0.0001)
      expect(cxErr).toBeLessThan(0.0001)
      expect(cyErr).toBeLessThan(0.0001)
    }
  })

  test('homographies have reasonable reprojection error', () => {
    const data = JSON.parse(readFileSync('/calibration_export_1777037266230.json', 'utf8'))

    const frameErrors: number[] = []

    for (const frame of data.calibrationFrames) {
      const tagPoints = groupByTag(frame as CalFrame, data.layout as LayoutTag[])
      if (tagPoints.size === 0) continue

      const allSrc: Vec2[] = []
      const allDst: Vec2[] = []
      for (const { obj, img } of tagPoints.values()) {
        allSrc.push(...obj)
        allDst.push(...img)
      }

      if (allSrc.length < 4) continue

      const H = computeHomography(allSrc, allDst)
      const error = computeReprojectionError(H, allSrc, allDst)
      frameErrors.push(error)
    }

    expect(frameErrors.length).toBeGreaterThan(0)

    const avgRms = frameErrors.reduce((s, e) => s + e, 0) / frameErrors.length
    const maxRms = Math.max(...frameErrors)

    // Average RMS should be under 1 pixel
    expect(avgRms).toBeLessThan(1)
    // Max RMS should be under 2 pixels
    expect(maxRms).toBeLessThan(2)
  })
})
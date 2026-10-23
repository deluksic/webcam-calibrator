import { describe, expect, it } from 'vitest'

import {
  buildCalibrationUrl,
  DEFAULT_TARGET_CONFIG,
  hexTargetConfigFromSearchParams,
  hexTargetConfigToSearchParams,
  normalizeHexTargetConfig,
} from '@/lib/hexTarget/config'

describe('hex target config', () => {
  it('uses defaults when params are missing', () => {
    expect(hexTargetConfigFromSearchParams(new URLSearchParams())).toEqual(DEFAULT_TARGET_CONFIG)
  })

  it('parses all editable parameters and resolution', () => {
    const config = hexTargetConfigFromSearchParams(
      new URLSearchParams(
        'columns=10&rows=6&points=3&seed=42&scale=0.55&depth=9&w=1600&h=1200',
      ),
    )
    expect(config).toEqual({
      columns: 10,
      rows: 6,
      pointsPerEdge: 3,
      markerScale: 0.55,
      colorSeed: 42,
      identityDepth: 9,
      widthPx: 1600,
      heightPx: 1200,
    })
  })

  it('clamps values to supported ranges', () => {
    const config = normalizeHexTargetConfig({
      columns: 1,
      rows: 99,
      pointsPerEdge: 0,
      markerScale: 2,
      colorSeed: -4,
      identityDepth: 40,
    })
    expect(config.columns).toBe(2)
    expect(config.rows).toBe(24)
    expect(config.pointsPerEdge).toBe(1)
    expect(config.markerScale).toBe(0.7)
    expect(config.colorSeed).toBe(0)
    expect(config.identityDepth).toBe(12)
  })

  it('round-trips through search params', () => {
    const config = normalizeHexTargetConfig({
      ...DEFAULT_TARGET_CONFIG,
      columns: 10,
      rows: 5,
      widthPx: 1600,
      heightPx: 1200,
    })
    const params = hexTargetConfigToSearchParams(config)
    expect(hexTargetConfigFromSearchParams(params)).toEqual(config)
  })

  it('builds a hash-router calibrate URL', () => {
    const url = buildCalibrationUrl(DEFAULT_TARGET_CONFIG, { widthPx: 1280, heightPx: 960 })
    expect(url).toContain('#/calibrate?')
    expect(url).toContain('columns=8')
    expect(url).toContain('rows=4')
    expect(url).toContain('w=1280')
    expect(url).toContain('h=960')
  })
})

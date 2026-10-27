import { describe, expect, it } from 'vitest'

import {
  RESOLUTION_PRESETS,
  filterResolutionPresets,
  resolutionFromKey,
  resolutionKey,
  resolutionLabel,
  supportedResolutions,
} from '@/components/camera/cameraStreamAcquire'

function fakeStream(capabilities: { width?: { min?: number; max?: number }; height?: { min?: number; max?: number } }) {
  return {
    getVideoTracks: () => [
      {
        getCapabilities: () => capabilities,
      },
    ],
  } as unknown as MediaStream
}

describe('camera resolutions', () => {
  it('round-trips canonical keys', () => {
    const resolution = { width: 1920, height: 1080 }
    expect(resolutionKey(resolution)).toBe('1920x1080')
    expect(resolutionFromKey('1920x1080')).toEqual(resolution)
    expect(resolutionLabel(resolution)).toBe('1920×1080')
  })

  it('rejects legacy names and malformed keys', () => {
    expect(resolutionFromKey(undefined)).toBeUndefined()
    expect(resolutionFromKey('')).toBeUndefined()
    expect(resolutionFromKey('low')).toBeUndefined()
    expect(resolutionFromKey('medium')).toBeUndefined()
    expect(resolutionFromKey('high')).toBeUndefined()
    expect(resolutionFromKey('1280')).toBeUndefined()
    expect(resolutionFromKey('1280x')).toBeUndefined()
    expect(resolutionFromKey('1280x720x2')).toBeUndefined()
    expect(resolutionFromKey('0x720')).toBeUndefined()
    expect(resolutionFromKey('-1280x720')).toBeUndefined()
  })

  it('offers modes above 1280 by default', () => {
    const options = filterResolutionPresets(undefined)
    expect(options).toEqual(RESOLUTION_PRESETS)
    expect(options).toContainEqual({ width: 1920, height: 1080 })
    expect(options).toContainEqual({ width: 2560, height: 1440 })
    expect(options).toContainEqual({ width: 3840, height: 2160 })
  })

  it('drops modes outside the camera-reported width/height range', () => {
    const options = filterResolutionPresets({
      width: { max: 1920 },
      height: { max: 1080 },
    })
    expect(options).toContainEqual({ width: 1920, height: 1080 })
    expect(options).not.toContainEqual({ width: 1920, height: 1440 })
    expect(options).not.toContainEqual({ width: 2560, height: 1440 })
    expect(options).not.toContainEqual({ width: 3840, height: 2160 })
    expect(options.at(-1)).toEqual({ width: 1920, height: 1080 })
  })

  it('respects minimum sizes too', () => {
    const options = filterResolutionPresets({
      width: { min: 1280 },
      height: { min: 720 },
    })
    expect(options).not.toContainEqual({ width: 1024, height: 768 })
    expect(options).toContainEqual({ width: 1280, height: 720 })
  })

  it('offers the camera-reported maximum when it is not a preset', () => {
    const options = supportedResolutions(fakeStream({ width: { max: 2592 }, height: { max: 1944 } }))
    expect(options.at(-1)).toEqual({ width: 2592, height: 1944 })
    expect(options).toContainEqual({ width: 2560, height: 1920 })
    expect(options).not.toContainEqual({ width: 3840, height: 2160 })
  })

  it('falls back to the full ladder when capabilities are unavailable', () => {
    const options = supportedResolutions(undefined)
    expect(options).toEqual(RESOLUTION_PRESETS)
  })
})

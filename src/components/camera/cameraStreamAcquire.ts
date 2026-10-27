const { navigator } = globalThis

export type CameraResolution = {
  width: number
  height: number
}

export type ResolutionLimits = {
  width?: { min?: number; max?: number }
  height?: { min?: number; max?: number }
}

/** Canonical `widthxheight` form used in URLs and `<option>` values. */
export type ResolutionKey = string

/**
 * Common capture modes, ascending. Browsers expose only capability ranges
 * (`width`/`height` min–max), not the camera's discrete mode list, so the
 * ladder is our candidate set; `getUserMedia` picks the closest real mode for
 * the `ideal` constraints.
 */
export const RESOLUTION_PRESETS: readonly CameraResolution[] = [
  { width: 320, height: 240 },
  { width: 640, height: 360 },
  { width: 640, height: 480 },
  { width: 800, height: 600 },
  { width: 960, height: 540 },
  { width: 1024, height: 768 },
  { width: 1280, height: 720 },
  { width: 1280, height: 960 },
  { width: 1600, height: 900 },
  { width: 1600, height: 1200 },
  { width: 1920, height: 1080 },
  { width: 1920, height: 1440 },
  { width: 2560, height: 1440 },
  { width: 2560, height: 1920 },
  { width: 3840, height: 2160 },
  { width: 4096, height: 2160 },
]

export const DEFAULT_RESOLUTION: CameraResolution = { width: 640, height: 480 }

export function resolutionKey(resolution: CameraResolution): ResolutionKey {
  return `${resolution.width}x${resolution.height}`
}

/** Human-readable size label for a resolution (e.g. for `<select>` options). */
export function resolutionLabel(resolution: CameraResolution): string {
  return `${resolution.width}×${resolution.height}`
}

/** Parse a canonical `WxH` key. */
export function resolutionFromKey(key: string | null | undefined): CameraResolution | undefined {
  if (!key) {
    return undefined
  }
  const match = /^(\d{1,5})x(\d{1,5})$/i.exec(key.trim())
  if (!match) {
    return undefined
  }
  const width = Number(match[1])
  const height = Number(match[2])
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
    return undefined
  }
  return { width, height }
}

/**
 * Keep the candidate modes the camera can reach. A mode outside the reported
 * width/height range cannot be supported by the device.
 */
export function filterResolutionPresets(
  limits: ResolutionLimits | undefined,
  presets: readonly CameraResolution[] = RESOLUTION_PRESETS,
): CameraResolution[] {
  if (!limits?.width && !limits?.height) {
    return [...presets]
  }
  const { width, height } = limits
  return presets.filter((resolution) => {
    if (width?.min !== undefined && resolution.width < width.min) {
      return false
    }
    if (width?.max !== undefined && resolution.width > width.max) {
      return false
    }
    if (height?.min !== undefined && resolution.height < height.min) {
      return false
    }
    if (height?.max !== undefined && resolution.height > height.max) {
      return false
    }
    return true
  })
}

/**
 * Candidate capture modes for the stream's camera. Falls back to the full
 * ladder when the browser does not expose capabilities.
 */
export function supportedResolutions(stream: MediaStream | undefined): CameraResolution[] {
  const track = stream?.getVideoTracks()[0]
  if (!track || typeof track.getCapabilities !== 'function') {
    return [...RESOLUTION_PRESETS]
  }
  try {
    const capabilities = track.getCapabilities()
    const options = filterResolutionPresets(capabilities)
    const maxWidth = capabilities.width?.max
    const maxHeight = capabilities.height?.max
    // Presets stop at 4K; some cameras report a larger native mode (e.g. a
    // 2592×1944 sensor). Offer the reported maximum as well.
    if (
      maxWidth !== undefined &&
      maxHeight !== undefined &&
      maxWidth >= 320 &&
      maxHeight >= 240 &&
      maxWidth <= 8192 &&
      maxHeight <= 8192 &&
      !options.some((option) => option.width === maxWidth && option.height === maxHeight)
    ) {
      options.push({ width: maxWidth, height: maxHeight })
      options.sort((a, b) => a.width * a.height - b.width * b.height)
    }
    return options
  } catch {
    return [...RESOLUTION_PRESETS]
  }
}

/**
 * Open a video-only stream for `deviceId` (exact), trying the requested size as
 * an ideal constraint and letting the camera choose its closest supported mode.
 */
export async function acquireVideoStream(deviceId: string, resolution: CameraResolution): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
      },
      audio: false,
    })
  } catch (e) {
    console.error('acquireVideoStream', e)
    throw e
  }
}

/** Prime permission so `enumerateDevices()` returns non-empty labels where supported. */
export async function primeCameraPermission(): Promise<void> {
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    })
    for (const track of s.getTracks()) {
      track.stop()
    }
  } catch {
    // user denied or no device — enumeration may still work
  }
}

export async function listVideoInputDevices(): Promise<MediaDeviceInfo[]> {
  const all = await navigator.mediaDevices.enumerateDevices()
  return all.filter((d) => d.kind === 'videoinput')
}

export function stopCameraStream(stream: MediaStream): void {
  stream.getTracks().forEach((t) => t.stop())
}

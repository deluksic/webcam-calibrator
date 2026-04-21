import { sleep } from '@/utils/sleep'

const { navigator } = globalThis
const { min } = Math
const ACCEPT_THRESHOLD = 0.9

export type Resolution = 'high' | 'medium' | 'low'
export const RESOLUTION_LADDER = {
  high: { width: { ideal: 1280 }, height: { ideal: 720 } },
  medium: { width: { ideal: 640 }, height: { ideal: 480 } },
  low: {},
} satisfies Record<string, MediaTrackConstraints>

export async function tryUpgradeVideoTrack(track: MediaStreamTrack): Promise<void> {
  const caps = track.getCapabilities()
  const { width, height } = caps
  if (!width?.max || !height?.max) {
    return
  }

  const settings = track.getSettings()
  const sw = settings.width ?? 0
  const sh = settings.height ?? 0
  if (sw >= width.max * ACCEPT_THRESHOLD && sh >= height.max * ACCEPT_THRESHOLD) {
    return
  }

  const targetW = min(width.max, 1280)
  const targetH = min(height.max, 720)
  for (const attempt of [0, 1, 2]) {
    try {
      await track.applyConstraints({
        width: { ideal: targetW },
        height: { ideal: targetH },
      })
      await sleep(120 + attempt * 100)
      const after = track.getSettings()
      const aw = after.width ?? 0
      const ah = after.height ?? 0
      if (aw >= targetW * ACCEPT_THRESHOLD && ah >= targetH * ACCEPT_THRESHOLD) {
        break
      }
    } catch {
      // keep default
    }
  }
}

/**
 * Open a video-only stream for `deviceId` (exact), trying size ideals then falling back.
 */
export async function acquireVideoStream(deviceId: string, resolution: Resolution | undefined): Promise<MediaStream> {
  let lastErr: unknown
  for (const size of resolution ? [RESOLUTION_LADDER[resolution]] : Object.values(RESOLUTION_LADDER)) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: deviceId },
          ...size,
        },
        audio: false,
      })
      const track = stream.getVideoTracks()[0]
      if (track) {
        await tryUpgradeVideoTrack(track)
      }
      return stream
    } catch (e) {
      lastErr = e
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
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

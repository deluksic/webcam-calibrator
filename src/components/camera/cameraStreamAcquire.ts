const { navigator } = globalThis

export type Resolution = keyof typeof RESOLUTION_LADDER
export const RESOLUTION_LADDER = {
  medium: { width: { ideal: 1280 }, height: { ideal: 720 } },
  low: { width: { ideal: 640 }, height: { ideal: 480 } },
} satisfies Record<string, MediaTrackConstraints>

/**
 * Open a video-only stream for `deviceId` (exact), trying size ideals then falling back.
 */
export async function acquireVideoStream(deviceId: string, resolution: Resolution): Promise<MediaStream> {
  const size = RESOLUTION_LADDER[resolution]
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        ...size,
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

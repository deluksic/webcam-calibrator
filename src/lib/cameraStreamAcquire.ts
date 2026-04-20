/**
 * Production-oriented `getUserMedia` + `applyConstraints` upgrade.
 * Browsers often start at 640×480; climb via constraint ladder + capabilities.
 */

const SIZE_LADDER: readonly MediaTrackConstraints[] = [
  { width: { ideal: 1920 }, height: { ideal: 1080 } },
  { width: { ideal: 1280 }, height: { ideal: 720 } },
  { width: { ideal: 640 }, height: { ideal: 480 } },
  {},
];

export async function tryUpgradeVideoTrack(track: MediaStreamTrack): Promise<void> {
  const vt = track as MediaStreamTrack & {
    getCapabilities?: () => MediaTrackCapabilities;
  };
  if (typeof vt.getCapabilities !== 'function') return;

  const caps = vt.getCapabilities();
  const w = caps.width;
  const h = caps.height;
  if (!w?.max || !h?.max) return;

  const settings = track.getSettings();
  const sw = settings.width ?? 0;
  const sh = settings.height ?? 0;
  if (sw >= w.max * 0.92 && sh >= h.max * 0.92) return;

  const targetW = Math.min(w.max, 1920);
  const targetH = Math.min(h.max, 1080);
  try {
    await track.applyConstraints({
      width: { ideal: targetW },
      height: { ideal: targetH },
    });
    await new Promise((r) => setTimeout(r, 120));
  } catch {
    /* keep default */
  }
}

/**
 * Open a video-only stream for `deviceId` (exact), trying size ideals then falling back.
 */
export async function acquireVideoStream(deviceId: string): Promise<MediaStream> {
  let lastErr: unknown;
  for (const size of SIZE_LADDER) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: deviceId },
          ...size,
        },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      if (track) await tryUpgradeVideoTrack(track);
      return stream;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

/** Prime permission so `enumerateDevices()` returns non-empty labels where supported. */
export async function primeCameraPermission(): Promise<void> {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    s.getTracks().forEach((t) => t.stop());
  } catch {
    /* user denied or no device — enumeration may still work */
  }
}

export async function listVideoInputDevices(): Promise<MediaDeviceInfo[]> {
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === 'videoinput');
}

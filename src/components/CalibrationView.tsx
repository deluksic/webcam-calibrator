import { createSignal, createMemo, onCleanup } from 'solid-js';
import { initGPU } from '../gpu/init';
import { createCameraPipeline, computeThreshold, processFrame, type CameraPipeline, type DisplayMode } from '../gpu/camera';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [threshold, setThreshold] = createSignal<number>(0.0);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('debug');

  // ── User-facing state ──────────────────────────────────────────────────
  const [selectedCameraId, setSelectedCameraId] = createSignal<string>('');
  const [availableCameraDevices, setAvailableCameraDevices] = createSignal<MediaDeviceInfo[]>([]);

  // ── One-time pipeline reference (set once, read forever) ───────────────
  const [pipeline, setPipeline] = createSignal<CameraPipeline | undefined>(undefined);

  // ── Computed ──────────────────────────────────────────────────────────

  const gpuRoot = createMemo(async () => {
    try {
      return await initGPU();
    } catch (e) {
      console.error('GPU init failed:', e);
      return null;
    }
  });

  const selectedCameraDevice = createMemo(() => {
    const id = selectedCameraId();
    if (!id) return availableCameraDevices()[0] ?? null;
    return availableCameraDevices().find((d) => d.deviceId === id) ?? null;
  });

  // ── Camera stream (reopens when selectedCameraDevice changes) ────────

  const stream = createMemo(async () => {
    const device = selectedCameraDevice();
    if (!device) return null;

    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: device.deviceId },
        facingMode: 'environment',
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    return mediaStream;
  });

  const frameSize = createMemo(() => {
    const s = stream();
    if (!s) return { w: 1920, h: 1080 };
    const settings = s.getVideoTracks()[0]?.getSettings();
    return {
      w: (settings?.width ?? 1920) as number,
      h: (settings?.height ?? 1080) as number,
    };
  });

  // ── Pipeline setup (runs once when stream starts) ────────────────────

  createMemo(async () => {
    const gpu = gpuRoot();
    const video = videoEl();
    const canvas = canvasEl();
    const histCanvas = histCanvasEl();
    const s = stream();
    if (!gpu || !video || !canvas || !histCanvas || !s) return;

    const size = frameSize();
    canvas.width = size.w;
    canvas.height = size.h;
    histCanvas.width = 512;
    histCanvas.height = 120;

    const pip = createCameraPipeline(
      gpu,
      canvas,
      histCanvas,
      size.w,
      size.h,
      navigator.gpu.getPreferredCanvasFormat(),
    );
    setPipeline(pip);

    video.srcObject = s;
    await video.play().catch(() => {});

    const loop = async () => {
      const vid = videoEl();
      const g = gpuRoot();
      const pip = pipeline();
      if (!vid || !g || !pip) return;
      processFrame(g, pip, vid, threshold(), displayMode());
      const bins = await pip.histogramBuffer.read();
      setThreshold(computeThreshold(bins, 0.9));
      vid.requestVideoFrameCallback(loop);
    };
    video.requestVideoFrameCallback(loop);
  });

  // ── Initial device enumeration ────────────────────────────────────────

  async function initCameraDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const sorted = devices
      .filter((d) => d.kind === 'videoinput')
      .sort((a, b) => deviceScore(b) - deviceScore(a));
    setAvailableCameraDevices(sorted);
    if (!selectedCameraId() && sorted[0]) {
      setSelectedCameraId(sorted[0].deviceId);
    }
  }
  initCameraDevices();

  // ── Cleanup ──────────────────────────────────────────────────────────

  onCleanup(() => {
    const vid = videoEl();
    if (vid) {
      vid.pause();
      vid.srcObject = null;
    }
    const s = stream();
    s?.getVideoTracks().forEach((t) => t.stop());
  });

  // ── Helpers ───────────────────────────────────────────────────────────

  function deviceScore(d: MediaDeviceInfo): number {
    const name = (d.label || '').toLowerCase();
    let s = 0;
    if (name.includes('back') || name.includes('rear')) s += 10;
    if (name.includes('front')) s -= 5;
    if (name.includes('ultra') || (name.includes('wide') && name.includes('0'))) s -= 8;
    if (name.includes('telephoto') || name.includes('macro') || name.includes('aux')) s -= 3;
    if (name.includes('0.5') || name.includes('0x')) s -= 8;
    return s;
  }

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div class={styles.root}>
      <div class={styles.feedRow}>
        <div class={`${styles.feedPanel} ${styles.feedPanelMain}`}>
          <div class={styles.feedHeader}>
            <span class={styles.feedLabel}>
              Camera Feed — {frameSize().w}×{frameSize().h}
            </span>
            <div class={styles.modeButtons}>
              <select
                class={styles.cameraSelect}
                value={selectedCameraId()}
                onChange={(e) => setSelectedCameraId(e.currentTarget.value)}
              >
                {availableCameraDevices().map((cam) => (
                  <option value={cam.deviceId}>
                    {cam.label || `Camera ${cam.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
              <button
                class={displayMode() === 'grayscale' ? styles.modeButtonActive : styles.modeButton}
                onClick={() => setDisplayMode('grayscale')}
              >
                Gray
              </button>
              <button
                class={displayMode() === 'edges' ? styles.modeButtonActive : styles.modeButton}
                onClick={() => setDisplayMode('edges')}
              >
                Edges
              </button>
              <button
                class={displayMode() === 'edgesDilated' ? styles.modeButtonActive : styles.modeButton}
                onClick={() => setDisplayMode('edgesDilated')}
              >
                Dilated
              </button>
              <button
                class={displayMode() === 'labels' ? styles.modeButtonActive : styles.modeButton}
                onClick={() => setDisplayMode('labels')}
              >
                Labels
              </button>
              <button
                class={displayMode() === 'debug' ? styles.modeButtonActive : styles.modeButton}
                onClick={() => setDisplayMode('debug')}
              >
                Debug
              </button>
            </div>
          </div>
          <canvas ref={setCanvasEl} class={styles.feedCanvas} />
        </div>
        <div class={`${styles.feedPanel} ${styles.feedPanelSide}`}>
          <span class={styles.feedLabel}>Edge Detection</span>
          <canvas
            ref={setHistCanvasEl}
            class={styles.histogramCanvas}
            style={{ width: '512px', height: '120px' }}
          />
          <div class={styles.histogramInfo}>
            <span class={styles.thresholdLabel}>85th Percentile Threshold</span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255{' '}
              <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
        </div>
      </div>
      <video ref={setVideoEl} style={{ display: 'none' }} />
    </div>
  );
}

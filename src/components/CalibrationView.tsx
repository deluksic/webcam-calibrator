import { createMemo, createSignal, onCleanup } from 'solid-js';
import { initGPU } from '../gpu/init';
import {
  createCameraPipeline,
  processFrame,
  type CameraPipeline,
  type DisplayMode,
} from '../gpu/camera';
import { computeThreshold } from '../gpu/pipelines/constants';
import styles from './CalibrationView.module.css';

function deviceScore(d: MediaDeviceInfo): number {
  const label = d.label.toLowerCase();
  let score = 0;
  if (label.includes('back') || label.includes('rear')) score += 100;
  if (label.includes('wide')) score += 50;
  if (label.includes('ultra')) score += 30;
  if (label.includes('tele')) score -= 20;
  if (label.includes('front') || label.includes('user')) score -= 100;
  return score;
}

async function enumerateVideoInputs(): Promise<MediaDeviceInfo[]> {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  } catch {
    // Labels may be empty without permission; enumeration still works.
  }
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === 'videoinput');
}

type GpuRoot = Awaited<ReturnType<typeof initGPU>>;

function CalibrationView() {
  const cameraVideo = document.createElement('video');
  cameraVideo.muted = true;
  cameraVideo.playsInline = true;

  onCleanup(() => {
    cameraVideo.pause();
    cameraVideo.srcObject = null;
  });

  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>();
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>();

  const [threshold, setThreshold] = createSignal(0);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('debug');

  const availableCameraDevices = createMemo(async () => {
    const inputs = await enumerateVideoInputs();
    return [...inputs].sort((a, b) => deviceScore(b) - deviceScore(a));
  }, [] as MediaDeviceInfo[]);

  /** Writable derived value: syncs with `availableCameraDevices`, preserves the id while it still exists. */
  const [selectedCameraId, setSelectedCameraId] = createSignal(
    (prev: string | undefined) => {
      const cams = availableCameraDevices();
      if (prev !== undefined && cams.some((d) => d.deviceId === prev)) return prev;
      return cams[0]?.deviceId;
    },
    undefined,
  );

  const selectedCameraDevice = createMemo(() => {
    const list = availableCameraDevices();
    const id = selectedCameraId();
    if (id === undefined) return list[0] ?? null;
    return list.find((d) => d.deviceId === id) ?? null;
  });

  const gpu = createMemo<GpuRoot | undefined>(async (_prev) => {
    try {
      return await initGPU();
    } catch (e) {
      console.error('GPU init failed:', e);
      return undefined;
    }
  });

  const mediaStream = createMemo<MediaStream | undefined>(async (prev) => {
    prev?.getTracks().forEach((t) => t.stop());

    let active: MediaStream | undefined = undefined;
    let disposed = false;
    onCleanup(() => {
      disposed = true;
      active?.getTracks().forEach((t) => t.stop());
      active = undefined;
    });

    const device = selectedCameraDevice();
    if (disposed || !device) return undefined;

    const base = { deviceId: { exact: device.deviceId } as const };
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          ...base,
          width: { min: 1280, ideal: 1920 },
          height: { min: 720, ideal: 1080 },
        },
        audio: false,
      });
      if (disposed) {
        stream.getTracks().forEach((t) => t.stop());
        return undefined;
      }
      active = stream;
      return stream;
    } catch (e) {
      if (disposed) return undefined;
      console.warn('getUserMedia: HD min constraints failed, retrying relaxed', e);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...base,
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (disposed) {
          stream.getTracks().forEach((t) => t.stop());
          return undefined;
        }
        active = stream;
        return stream;
      } catch (e2) {
        console.error('getUserMedia failed:', e2);
        return undefined;
      }
    }
  });

  const frameSize = createMemo(() => {
    const stream = mediaStream();
    if (!stream) return { w: 1280, h: 720 };
    const track = stream.getVideoTracks()[0];
    if (!track) return { w: 1280, h: 720 };
    const settings = track.getSettings();
    return {
      w: settings.width ?? 1280,
      h: settings.height ?? 720,
    };
  });

  /** WebGPU pipeline + rVFC loop; `onCleanup` runs before `await video.play()`. */
  const cameraPipeline = createMemo(
    async (_prev: CameraPipeline | undefined) => {
      const g = gpu();
      const canvas = canvasEl();
      const histCanvas = histCanvasEl();
      const stream = mediaStream();
      if (!g || !canvas || !histCanvas || !stream) {
        return undefined;
      }
      const video = cameraVideo;

      const size = frameSize();
      let primeHandle = 0;
      let rafHandle = 0;
      let disposed = false;

      onCleanup(() => {
        disposed = true;
        if (primeHandle) video.cancelVideoFrameCallback(primeHandle);
        if (rafHandle) video.cancelVideoFrameCallback(rafHandle);
        video.pause();
        video.srcObject = null;
      });

      canvas.width = size.w;
      canvas.height = size.h;
      histCanvas.width = 512;
      histCanvas.height = 120;

      const pip = createCameraPipeline(
        g,
        canvas,
        histCanvas,
        size.w,
        size.h,
        navigator.gpu.getPreferredCanvasFormat(),
      );
      video.srcObject = stream;
      await video.play().catch(() => {});
      if (disposed) return undefined;

      // `importExternalTexture` needs a presented frame; `play()` can resolve earlier.
      await new Promise<void>((resolve) => {
        primeHandle = video.requestVideoFrameCallback(() => {
          primeHandle = 0;
          resolve();
        });
      });
      if (disposed) return undefined;

      const loop = () => {
        if (disposed) return;
        const gpuNow = gpu();
        if (!gpuNow) return;
        processFrame(gpuNow, pip, cameraVideo, threshold(), displayMode());
        void pip.histogramBuffer.read().then((bins) => {
          if (disposed) return;
          const data = bins instanceof Uint32Array ? bins : new Uint32Array(bins);
          setThreshold(computeThreshold([...data], 0.85));
        });
        rafHandle = cameraVideo.requestVideoFrameCallback(loop);
      };
      rafHandle = cameraVideo.requestVideoFrameCallback(loop);

      return pip;
    },
    undefined as CameraPipeline | undefined,
    { lazy: false },
  );

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
                type="button"
                class={
                  displayMode() === 'grayscale'
                    ? styles.modeButtonActive
                    : styles.modeButton
                }
                onClick={() => setDisplayMode('grayscale')}
              >
                Gray
              </button>
              <button
                type="button"
                class={
                  displayMode() === 'edges' ? styles.modeButtonActive : styles.modeButton
                }
                onClick={() => setDisplayMode('edges')}
              >
                Edges
              </button>
              <button
                type="button"
                class={
                  displayMode() === 'edgesDilated'
                    ? styles.modeButtonActive
                    : styles.modeButton
                }
                onClick={() => setDisplayMode('edgesDilated')}
              >
                Dilated
              </button>
              <button
                type="button"
                class={
                  displayMode() === 'labels' ? styles.modeButtonActive : styles.modeButton
                }
                onClick={() => setDisplayMode('labels')}
              >
                Labels
              </button>
              <button
                type="button"
                class={
                  displayMode() === 'debug' ? styles.modeButtonActive : styles.modeButton
                }
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
    </div>
  );
}

export default CalibrationView;
export { CalibrationView };

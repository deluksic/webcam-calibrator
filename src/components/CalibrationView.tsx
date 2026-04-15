import { createSignal, createMemo, onCleanup } from 'solid-js';
import { initGPU } from '../gpu/init';
import { createCameraPipeline, computeThreshold, processFrame, type CameraPipeline, type DisplayMode } from '../gpu/camera';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);
  const [threshold, setThreshold] = createSignal<number>(0.0);
  const [histogramData, setHistogramData] = createSignal<number[]>(new Array(256).fill(0));
  const [gpuReady, setGpuReady] = createSignal(false);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('debug');

  let pipeline: CameraPipeline | null = null;
  let videoCallbackId: number | null = null;

  // Initialize GPU
  const root = createMemo(async () => {
    try {
      const gpu = await initGPU();
      setGpuReady(true);
      return gpu;
    } catch (e) {
      console.error('GPU init failed:', e);
      return null;
    }
  });

  // Request camera stream
  createMemo(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      });
      setStream(mediaStream);

      const track = mediaStream.getVideoTracks()[0];
      const settings = track.getSettings();
      setFrameSize({
        w: (settings.width ?? 1280) as number,
        h: (settings.height ?? 720) as number,
      });
    } catch (e) {
      console.error('Camera access failed:', e);
    }
  });

  // Set up pipeline and render loop when all elements are ready
  createMemo(async () => {
    const video = videoEl();
    const canvas = canvasEl();
    const histCanvas = histCanvasEl();
    const gpuRoot = root();

    if (!gpuRoot || !video || !canvas || !histCanvas) return;

    const size = frameSize();
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // Create pipeline with separate histogram canvas
    pipeline = createCameraPipeline(gpuRoot, canvas, histCanvas, size.w, size.h, presentationFormat);

    // Set canvas dimensions
    canvas.width = pipeline.width;
    canvas.height = pipeline.height;
    histCanvas.width = pipeline.histWidth;
    histCanvas.height = pipeline.histHeight;

    // Attach video stream
    video.srcObject = stream();
    await video.play().catch(() => {});

    // Render loop using requestVideoFrameCallback
    const loop = async () => {
      if (video.readyState >= 2 && gpuRoot && pipeline) {
        // Process frame (all GPU work in single submit)
        processFrame(gpuRoot, pipeline, video, threshold(), displayMode());

        // Wait for completion and read back histogram data
        const bins = await pipeline.histogramBuffer.read();
        const thresh = computeThreshold(bins, 0.85);
        setHistogramData(bins);
        setThreshold(thresh);
      }
      videoCallbackId = video.requestVideoFrameCallback(loop);
    };
    videoCallbackId = video.requestVideoFrameCallback(loop);
  });

  onCleanup(() => {
    if (videoCallbackId !== null) {
      // Note: requestVideoFrameCallback can't be cancelled directly
      // The cleanup happens automatically when the video element is removed
    }
  });

  return (
    <div class={styles.root}>
      <div class={styles.feedRow}>
        <div class={styles.feedPanel}>
          <div class={styles.feedHeader}>
            <span class={styles.feedLabel}>Camera Feed — {frameSize().w}×{frameSize().h}</span>
            <div class={styles.modeButtons}>
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
          <canvas
            ref={setCanvasEl}
            class={styles.feedCanvas}
            style={{ width: '640px', height: '360px' }}
          />
        </div>
        <div class={styles.feedPanel}>
          <span class={styles.feedLabel}>Edge Detection</span>
          <canvas
            ref={setHistCanvasEl}
            class={styles.histogramCanvas}
            style={{ width: '512px', height: '120px' }}
          />
          <div class={styles.histogramInfo}>
            <span class={styles.thresholdLabel}>85th Percentile Threshold</span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255 <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
        </div>
      </div>
      <video ref={setVideoEl} style={{ display: 'none' }} />
    </div>
  );
}

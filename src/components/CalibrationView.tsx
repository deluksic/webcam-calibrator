import { createSignal, createMemo, For } from 'solid-js';
import { initGPU } from '../gpu/init';
import { processFrameAsync, type CameraPipeline, DisplayMode } from '../gpu/camera';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('edges');
  const [threshold, setThreshold] = createSignal<number>(0.0);
  const [histogramData, setHistogramData] = createSignal<number[]>(new Array(256).fill(0));

  // Initialize GPU
  const gpuRoot = createMemo(() => initGPU());

  // Request camera stream
  createMemo(async () => {
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
  });

  // Set canvas size and start render loop when ready
  createMemo(() => {
    const root = gpuRoot();
    const video = videoEl();
    const canvas = canvasEl();
    if (!root || !video || !canvas) return;

    const size = frameSize();
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // Dynamically import pipeline factory
    import('../gpu/camera').then(async ({ createCameraPipeline, computeThreshold }) => {
      const p = createCameraPipeline(root, canvas, size.w, size.h, presentationFormat);

      // Set canvas dimensions
      canvas.width = p.width;
      canvas.height = p.height;

      // Attach video stream
      video.srcObject = stream();
      await video.play().catch(() => {});

      // Render loop using requestVideoFrameCallback
      const loop = async () => {
        if (video.readyState >= 2) {
          await processFrameAsync(root, p, video, displayMode());

          // Update UI with histogram data
          const bins = await p.histogramBuffer.read();
          console.log('Histogram bins (first 10):', bins.slice(0, 10));
          console.log('Histogram total:', bins.reduce((a, b) => a + b, 0));
          setHistogramData(bins);

          const thresh = computeThreshold(bins, 0.85);
          console.log('Computed threshold:', thresh);
          setThreshold(thresh);
        }
        video.requestVideoFrameCallback(loop);
      };
      video.requestVideoFrameCallback(loop);
    });
  });

  // Create array of 256 indices for rendering histogram bars
  const histogramBars = createMemo(() => {
    const bins = histogramData();
    const maxCount = Math.max(...bins, 1);
    return bins.map((count, i) => ({
      index: i,
      height: Math.round((count / maxCount) * 100),
      isThreshold: Math.abs(i / 256 - threshold()) < 0.005,
    }));
  });

  return (
    <div class={styles.root}>
      <div class={styles.feedRow}>
        <div class={styles.feedPanel}>
          <span class={styles.feedLabel}>Camera Feed — {frameSize().w}×{frameSize().h}</span>
          <canvas
            ref={setCanvasEl}
            class={styles.feedCanvas}
            style={{ width: '640px', height: '360px' }}
          />
          <div class={styles.histogramContainer}>
            <div class={styles.thresholdLabel}>Threshold (85th percentile)</div>
            <div class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255 <span>({(threshold() * 100).toFixed(1)}%)</span>
            </div>
          </div>
          <div class={styles.feedCanvas} style={{
            'width': '512px',
            'height': '120px',
            'align-items': 'flex-end',
            background: '#1a1a2e',
            'padding-top': '8px',
            gap: '0',
          }}>
            <For each={histogramBars()}>
              {(bar) => (
                <div style={{
                  width: '2px',
                  'height': `${bar().height}%`,
                  'background-color': bar().isThreshold ? '#ff4a4a' : '#4a9eff',
                  'min-height': bar().height > 0 ? '1px' : '0',
                }} />
              )}
            </For>
          </div>
          <video ref={setVideoEl} style={{ display: 'none' }} />
        </div>
      </div>
      <div class={styles.controls}>
        <button onClick={() => setDisplayMode('edges')} class={displayMode() === 'edges' ? styles.active : ''}>Edges</button>
        <button onClick={() => setDisplayMode('sobel')} class={displayMode() === 'sobel' ? styles.active : ''}>Sobel</button>
        <button onClick={() => setDisplayMode('grayscale')} class={displayMode() === 'grayscale' ? styles.active : ''}>Grayscale</button>
        <button onClick={() => setDisplayMode('original')} class={displayMode() === 'original' ? styles.active : ''}>Original</button>
      </div>
    </div>
  );
}

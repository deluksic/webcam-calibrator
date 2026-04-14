import { createSignal, createMemo } from 'solid-js';
import { initGPU } from '../gpu/init';
import { processFrameAsync, type CameraPipeline, DisplayMode } from '../gpu/camera';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [histogramCanvasEl, setHistogramCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('edges');
  const [threshold, setThreshold] = createSignal<number>(0.5);
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

  // Create camera pipeline when GPU, canvas, and frame size are ready
  const pipeline = createMemo<CameraPipeline | null>(() => {
    const root = gpuRoot();
    const canvas = canvasEl();
    const size = frameSize();
    if (!root || !canvas) return null;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    // Import dynamically to avoid circular deps - but we need the pipeline
    return null; // Pipeline created in the memo below
  });

  // Set canvas size and start render loop when ready
  createMemo(() => {
    const root = gpuRoot();
    const video = videoEl();
    const canvas = canvasEl();
    const histCanvas = histogramCanvasEl();
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

      // Configure histogram canvas
      if (histCanvas) {
        histCanvas.width = 512;
        histCanvas.height = 150;
      }

      // Render loop using requestVideoFrameCallback
      const loop = async () => {
        if (video.readyState >= 2) {
          await processFrameAsync(root, p, video, displayMode());

          // Update UI with histogram data
          const histResult = await p.histogramBuffer.read();
          const bins = histResult.map((b: { count: number }) => b.count);
          setHistogramData(bins);

          const thresh = computeThreshold(bins.map((c, i) => ({ count: c })), 0.85);
          setThreshold(thresh);

          // Draw histogram on canvas
          if (histCanvas) {
            const ctx = histCanvas.getContext('2d');
            if (ctx) {
              ctx.fillStyle = '#1a1a2e';
              ctx.fillRect(0, 0, histCanvas.width, histCanvas.height);

              const maxCount = Math.max(...bins);
              const binWidth = histCanvas.width / 256;
              const histHeight = histCanvas.height - 20;

              ctx.fillStyle = '#4a9eff';
              for (let i = 0; i < 256; i++) {
                const barHeight = (bins[i] / maxCount) * histHeight;
                const x = i * binWidth;
                ctx.fillRect(x, histHeight - barHeight, binWidth - 1, barHeight);
              }

              // Draw threshold line
              const thresholdX = thresh * histCanvas.width;
              ctx.strokeStyle = '#ff4a4a';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(thresholdX, 0);
              ctx.lineTo(thresholdX, histHeight);
              ctx.stroke();

              // Label
              ctx.fillStyle = '#888';
              ctx.font = '12px monospace';
              ctx.fillText(`Threshold: ${thresh.toFixed(3)}`, 10, histCanvas.height - 5);
            }
          }
        }
        video.requestVideoFrameCallback(loop);
      };
      video.requestVideoFrameCallback(loop);
    });
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
          <canvas
            ref={setHistogramCanvasEl}
            class={styles.feedCanvas}
            style={{ width: '512px', height: '150px' }}
          />
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

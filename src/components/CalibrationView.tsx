import { createSignal, createMemo } from 'solid-js';
import { initGPU } from '../gpu/init';
import { processFrameAsync, type CameraPipeline, DisplayMode } from '../gpu/camera';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('edges');

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
    if (!root || !video || !canvas) return;

    const size = frameSize();
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // Dynamically import pipeline factory
    import('../gpu/camera').then(async ({ createCameraPipeline }) => {
      const p = createCameraPipeline(root, canvas, size.w, size.h, presentationFormat);

      // Set canvas dimensions
      canvas.width = p.width;
      canvas.height = p.height;

      // Attach video stream
      video.srcObject = stream();
      await video.play().catch(() => {});

      // Render loop using requestVideoFrameCallback
      const loop = (_timestamp: number, metadata: VideoFrameCallbackMetadata) => {
        if (video.readyState >= 2) {
          processFrameAsync(root, p, video, displayMode());
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

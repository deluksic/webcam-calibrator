import { createSignal, createMemo, isPending } from 'solid-js';
import { initGPU } from '../gpu/init';
import { createGrayscalePipeline, processFrame, type GrayscalePipeline } from '../gpu/grayscale';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  const [videoEl, setVideoEl] = createSignal<HTMLVideoElement | undefined>(undefined);
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement | undefined>(undefined);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);

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

  // Create grayscale pipeline when GPU and frame size are ready
  const pipeline = createMemo<GrayscalePipeline | null>(() => {
    const root = gpuRoot();
    const size = frameSize();
    if (!root) return null;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    return createGrayscalePipeline(root, size.w, size.h, presentationFormat);
  });

  // Set canvas size and start render loop when pipeline is ready
  createMemo(() => {
    const p = pipeline();
    const video = videoEl();
    const canvas = canvasEl();
    if (!p || !video || !canvas) return;

    // Set canvas dimensions
    canvas.width = p.width;
    canvas.height = p.height;

    // Attach video stream
    video.srcObject = stream();
    video.play().catch(() => {});

    // Get WebGPU context
    const context = canvas.getContext('webgpu');
    if (!context) return;

    // Animation loop
    const loop = () => {
      if (video.readyState >= 2) {
        processFrame(gpuRoot(), p, video, context);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
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
    </div>
  );
}

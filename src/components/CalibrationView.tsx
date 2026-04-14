import { createSignal, createMemo, isPending } from 'solid-js';
import { initGPU } from '../gpu/init';
import { createCameraPipeline, processFrame, type CameraPipeline } from '../gpu/camera';
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

  // Create camera pipeline when GPU, canvas, and frame size are ready
  const pipeline = createMemo<CameraPipeline | null>(() => {
    const root = gpuRoot();
    const canvas = canvasEl();
    const size = frameSize();
    if (!root || !canvas) return null;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    return createCameraPipeline(root, canvas, size.w, size.h, presentationFormat);
  });

  // Set canvas size and start render loop when pipeline is ready
  createMemo(() => {
    const p = pipeline();
    const video = videoEl();
    if (!p || !video) return;

    // Set canvas dimensions
    const canvas = canvasEl();
    if (canvas) {
      canvas.width = p.width;
      canvas.height = p.height;
    }

    // Attach video stream
    video.srcObject = stream();
    video.play().catch(() => {});

    // Animation loop
    const loop = () => {
      if (video.readyState >= 2) {
        processFrame(gpuRoot(), p, video, 'sobel');
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

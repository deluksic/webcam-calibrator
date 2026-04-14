import { createSignal, createMemo, isPending, Show, onCleanup } from 'solid-js';
import { initGPU } from '../gpu/init';
import { createGrayscalePipeline, processFrame, type GrayscalePipeline } from '../gpu/grayscale';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  let videoEl: HTMLVideoElement | undefined;
  let canvasEl: HTMLCanvasElement | undefined;

  const [error, setError] = createSignal<string | null>(null);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });
  const [stream, setStream] = createSignal<MediaStream | null>(null);

  // Initialize GPU
  const gpuRoot = createMemo(() => {
    const gpu = initGPU();
    if (gpu instanceof Promise) {
      gpu.catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }
    return gpu;
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
      setError(e instanceof Error ? e.message : String(e));
    }
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
    if (!p || !videoEl || !canvasEl) return;

    // Set canvas dimensions
    canvasEl.width = p.width;
    canvasEl.height = p.height;

    // Attach video stream
    videoEl.srcObject = stream();
    videoEl.play().catch(() => {});

    // Get WebGPU context
    const context = canvasEl.getContext('webgpu');
    if (!context) {
      setError('Failed to get WebGPU context');
      return;
    }

    // Animation loop
    let animId: number;
    const loop = () => {
      if (videoEl && videoEl.readyState >= 2) {
        processFrame(gpuRoot(), p, videoEl, context);
      }
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);

    onCleanup(() => cancelAnimationFrame(animId));
  });

  // Cleanup stream on unmount
  onCleanup(() => {
    const s = stream();
    if (s) {
      s.getTracks().forEach((t) => t.stop());
    }
  });

  return (
    <div class={styles.root}>
      <Show when={error()}>
        <p class={styles.error}>{error()}</p>
      </Show>
      <Show when={!error()}>
        <div class={styles.feedRow}>
          <div class={styles.feedPanel}>
            <span class={styles.feedLabel}>Camera Feed — {frameSize().w}×{frameSize().h}</span>
            <canvas
              ref={canvasEl}
              class={styles.feedCanvas}
              style={{ width: '640px', height: '360px' }}
            />
          </div>
        </div>
        <Show when={isPending(() => gpuRoot())}>
          <p class={styles.loading}>Initializing GPU...</p>
        </Show>
      </Show>
    </div>
  );
}

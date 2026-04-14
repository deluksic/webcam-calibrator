import { createSignal, onCleanup, createRoot } from 'solid-js';
import { initGPU } from '../gpu/init';
import * as grayscale from '../gpu/grayscale';
import styles from './CalibrationView.module.css';

export default function CalibrationView() {
  let videoEl: HTMLVideoElement | undefined;
  let canvasEl: HTMLCanvasElement | undefined;
  let animFrameId = 0;

  const [error, setError] = createSignal('');
  const [ready, setReady] = createSignal(false);
  const [frameSize, setFrameSize] = createSignal({ w: 1280, h: 720 });

  createRoot(async (dispose) => {
    try {
      // Initialize GPU first
      await initGPU();

      // Request camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      });

      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      const w = (settings.width ?? 1280) as number;
      const h = (settings.height ?? 720) as number;
      setFrameSize({ w, h });

      if (videoEl) {
        videoEl.srcObject = stream;
        await videoEl.play();
      }

      // Set canvas size to match camera resolution (must precede init)
      if (canvasEl) {
        canvasEl.width = w;
        canvasEl.height = h;
      }

      await grayscale.init(w, h, canvasEl!);
      setReady(true);

      const loop = () => {
        if (videoEl && videoEl.readyState >= 2) {
          grayscale.processFrame(videoEl);
        }
        animFrameId = requestAnimationFrame(loop);
      };
      animFrameId = requestAnimationFrame(loop);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }

    onCleanup(() => {
      cancelAnimationFrame(animFrameId);
      if (videoEl && videoEl.srcObject) {
        const tracks = (videoEl.srcObject as MediaStream).getTracks();
        for (let i = 0; i < tracks.length; i++) tracks[i].stop();
      }
      dispose();
    });
  });

  return (
    <div class={styles.root}>
      {error() !== '' ? (
        <p class={styles.error}>{error()}</p>
      ) : (
        <>
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
          {!ready() && <p class={styles.loading}>Loading...</p>}
        </>
      )}
    </div>
  );
}

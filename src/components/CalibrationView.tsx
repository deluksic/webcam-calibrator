import { For, Show, createMemo, createSignal, onCleanup } from 'solid-js';
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

interface Bbox { minX: number; minY: number; maxX: number; maxY: number; area: number }

function BboxOverlay(props: {
  bboxes: () => Bbox[];
  sx: () => number;
  sy: () => number;
  fw: () => number;
  fh: () => number;
}) {
  const visible = createMemo(() => {
    const half = (props.fw() * props.fh()) / 2;
    return props.bboxes().filter((b) => b.area > 100 && b.area < half);
  });
  return (
    <For each={visible()} keyed={false}>
      {(box) => (
        <div
          class={styles.bbox}
          style={{
            '--bbox-x': `${box().minX * props.sx()}px`,
            '--bbox-y': `${box().minY * props.sy()}px`,
            '--bbox-w': `${(box().maxX - box().minX) * props.sx()}px`,
            '--bbox-h': `${(box().maxY - box().minY) * props.sy()}px`,
          }}
        />
      )}
    </For>
  );
}

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
  const [renderedW, setRenderedW] = createSignal(1280);
  const [renderedH, setRenderedH] = createSignal(720);

  const canvasRefCallback = (el: HTMLCanvasElement | undefined) => {
    setCanvasEl(el);
    if (el) {
      setRenderedW(el.clientWidth || 1280);
      setRenderedH(el.clientHeight || 720);
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          setRenderedW(entry.contentRect.width);
          setRenderedH(entry.contentRect.height);
        }
      });
      ro.observe(el);
      onCleanup(() => ro.disconnect());
    }
  };

  const [threshold, setThreshold] = createSignal(0);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('debug');
  const [bboxes, setBboxes] = createSignal<Bbox[]>([]);
  const [logs, setLogs] = createSignal<string[]>([]);

  const log = (msg: string) => {
    setLogs(prev => [...prev.slice(-8), `${new Date().toISOString().slice(11, 19)} ${msg}`]);
    console.log(msg);
  };

  const availableCameraDevices = createMemo(async () => {
    const inputs = await enumerateVideoInputs();
    return [...inputs].sort((a, b) => deviceScore(b) - deviceScore(a));
  }, [] as MediaDeviceInfo[]);

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
      log('GPU init...');
      const g = await initGPU();
      log('GPU ready');
      return g;
    } catch (e) {
      log(`GPU init failed: ${e}`);
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
          width: { min: 640, ideal: 1280 },
          height: { min: 480, ideal: 720 },
        },
        audio: false,
      });
      if (disposed) {
        stream.getTracks().forEach((t) => t.stop());
        return undefined;
      }
      active = stream;
      log('Stream opened');
      return stream;
    } catch (e) {
      if (disposed) return undefined;
      log('getUserMedia HD failed, retrying...');
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
        log('Stream opened (fallback)');
        return stream;
      } catch (e2) {
        log(`getUserMedia failed: ${e2}`);
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

  const cameraPipeline = createMemo(
    async (_prev: CameraPipeline | undefined) => {
      const g = gpu();
      const canvas = canvasEl();
      const histCanvas = histCanvasEl();
      const stream = mediaStream();
      if (!g || !canvas || !histCanvas || !stream) {
        log('Pipeline: missing deps');
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
        log('Pipeline cleanup');
      });

      canvas.width = size.w;
      canvas.height = size.h;
      histCanvas.width = 512;
      histCanvas.height = 120;

      log('Creating pipeline...');
      const pip = createCameraPipeline(
        g,
        canvas,
        histCanvas,
        size.w,
        size.h,
        navigator.gpu.getPreferredCanvasFormat(),
      );
      log(`Pipeline created ${size.w}x${size.h}`);
      video.srcObject = stream;
      await video.play().catch(() => {});
      if (disposed) return undefined;

      await new Promise<void>((resolve) => {
        primeHandle = video.requestVideoFrameCallback(() => {
          primeHandle = 0;
          resolve();
        });
      });
      if (disposed) return undefined;
      log('First frame presented');

      const EXTENT_FIELDS = 4;
      const MAX_COMPONENTS = 65536;

      let frameCount = 0;
      const loop = () => {
        if (disposed) return;
        const gpuNow = gpu();
        if (!gpuNow) return;
        frameCount++;
        const dm = displayMode();
        processFrame(gpuNow, pip, cameraVideo, threshold(), dm);
        if (frameCount % 30 === 0) log(`frame ${frameCount} mode=${dm}`);
        void pip.histogramBuffer.read().then((bins) => {
          if (disposed) return;
          const data = bins instanceof Uint32Array ? bins : new Uint32Array(bins);
          setThreshold(computeThreshold([...data], 0.85));
        });
        // Trigger extent readback every 10 frames — reads latest available data
        if (frameCount % 10 === 0) {
          const thisFrame = frameCount;
          pip.extentBuffer.read(new Uint32Array(MAX_COMPONENTS * EXTENT_FIELDS)).then((raw) => {
            if (disposed) return;
            if (frameCount - thisFrame > 5) return; // stale result
            const safeRaw = raw instanceof Uint32Array ? raw : new Uint32Array(raw);
            const boxes: Bbox[] = [];
            let skippedInvalid = 0, skippedSentinel = 0, skippedSmall = 0, skippedNeg = 0;
            for (let i = 0; i < MAX_COMPONENTS; i++) {
              const minX = safeRaw[i * EXTENT_FIELDS + 0];
              const minY = safeRaw[i * EXTENT_FIELDS + 1];
              const maxX = safeRaw[i * EXTENT_FIELDS + 2];
              const maxY = safeRaw[i * EXTENT_FIELDS + 3];
              if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) { skippedInvalid++; continue; }
              if (minX === 0xFFFFFFFF) { skippedSentinel++; continue; }
              const w = maxX - minX;
              const h = maxY - minY;
              if (minX > maxX) { skippedNeg++; continue; }
              if (w <= 0 || h <= 0) { skippedSmall++; continue; }
              boxes.push({ minX, minY, maxX, maxY, area: w * h });
            }
            boxes.sort((a, b) => b.area - a.area);
            const top5 = boxes.slice(0, 5);
            console.log('EXTENT raw[0-4]:', Array.from({ length: 5 }, (_, i) => {
              const b = top5[i];
              return b ? `[${b.minX},${b.minY},${b.maxX},${b.maxY}] area=${b.area}` : 'invalid';
            }));
            console.log('EXTENT setBboxes:', top5.map(b => `${b.area}@(${b.minX},${b.minY})-(${b.maxX},${b.maxY})`));
            setBboxes(boxes.slice(0, 50));
            const top3 = boxes.slice(0, 3).map(b => `${b.area}`).join(', ') || 'none';
            log(`Extents: found=${boxes.length} top3=${top3} skip{inv=${skippedInvalid}, sen=${skippedSentinel}, neg=${skippedNeg}, small=${skippedSmall}}`);
          });
        }
        rafHandle = cameraVideo.requestVideoFrameCallback(loop);
      };
      rafHandle = cameraVideo.requestVideoFrameCallback(loop);
      log('rVFC loop started');

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
          <div style={{ position: 'relative' }}>
            <canvas ref={canvasRefCallback} class={styles.feedCanvas} />
            <Show when={displayMode() === 'debug'}>
              <BboxOverlay
                bboxes={bboxes}
                fw={() => frameSize().w}
                fh={() => frameSize().h}
                sx={() => (canvasEl() && renderedW() > 0 && frameSize().w > 0) ? renderedW() / frameSize().w : 1}
                sy={() => (canvasEl() && renderedH() > 0 && frameSize().h > 0) ? renderedH() / frameSize().h : 1}
              />
            </Show>
          </div>
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
          <div style={{ font: '9px monospace', color: '#4f8', marginTop: 4 }}>
            {logs().map(l => <div>{l}</div>)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CalibrationView;
export { CalibrationView };
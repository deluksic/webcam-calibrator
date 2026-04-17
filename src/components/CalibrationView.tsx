import { For, Show, createMemo, createSignal, onCleanup } from 'solid-js';
import { initGPU } from '../gpu/init';
import {
  createCameraPipeline,
  processFrame,
  readExtentBuffer,
  readExtentDataForQuads,
  updateQuadCornersBuffer,
  detectContours,
  type CameraPipeline,
  type DisplayMode,
  type ExtentRow,
  MAX_U32,
  MAX_COMPONENTS,
  MAX_DETECTED_TAGS,
} from '../gpu/camera';
import { type DetectedQuad } from '../gpu/contour';
import { computeThreshold, THRESHOLD_PERCENTILE } from '../gpu/pipelines/constants';
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

function QuadCandidateOverlay(props: {
  bboxes: Bbox[];
  sx: number;
  sy: number;
  fw: number;
  fh: number;
}) {
  const candidates = createMemo(() => {
    const MIN_AREA = 400;
    const MAX_AREA = 200000;
    const MIN_AR = 0.6;
    const MAX_AR = 1.7;
    let okAreaAR = 0, okContained = 0, dropped = 0;
    const passing = props.bboxes.filter((b) => {
      const w = b.maxX - b.minX;
      const h = b.maxY - b.minY;
      if (w <= 0 || h <= 0) { dropped++; return false; }
      const area = w * h;
      if (area < MIN_AREA || area > MAX_AREA) { dropped++; return false; }
      const ar = w / h;
      if (ar < MIN_AR || ar > MAX_AR) { dropped++; return false; }
      okAreaAR++;
      return true;
    });
    const result = passing.filter((candidate) => {
      for (const other of passing) {
        if (other === candidate) continue;
        // Discard candidate if it is fully contained inside another box
        if (other.minX <= candidate.minX && other.maxX >= candidate.maxX &&
            other.minY <= candidate.minY && other.maxY >= candidate.maxY) {
          okContained++;
          return false;
        }
      }
      return true;
    });
    return result;
  });

  return (
    <For each={candidates()} keyed={false}>
      {(box) => (
        <div
          class={styles.bbox}
          style={{
            '--bbox-x': `${box().minX * props.sx}px`,
            '--bbox-y': `${box().minY * props.sy}px`,
            '--bbox-w': `${(box().maxX - box().minX) * props.sx}px`,
            '--bbox-h': `${(box().maxY - box().minY) * props.sy}px`,
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
    console.log('[CalibrationView] onCleanup fired - pausing cameraVideo');
    cameraVideo.pause();
    cameraVideo.srcObject = null;
  });

  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>();
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>();
  const [renderedW, setRenderedW] = createSignal(1280);
  const [renderedH, setRenderedH] = createSignal(720);

  const scaleX = createMemo(() => canvasEl() ? renderedW() / frameSize().w : 1);
  const scaleY = createMemo(() => canvasEl() ? renderedH() / frameSize().h : 1);

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
  const [quadCandidateCount, setQuadCandidateCount] = createSignal(0);

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
      console.log('[mediaStream] onCleanup fired');
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
        console.log('[cameraPipeline] onCleanup fired');
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

      let frameCount = 0;
      let extentReadPending = false;
      let quadDetectionPending = false;

      const scheduleExtentRead = () => {
        if (extentReadPending || disposed) return;
        extentReadPending = true;
        readExtentBuffer(pip).then((extentData: ExtentRow[]) => {
          if (disposed) return;
          extentReadPending = false;
          const boxes: Bbox[] = [];
          for (const entry of extentData) {
            if (entry.minX === MAX_U32) continue; // uninitialized
            const w = entry.maxX - entry.minX;
            const h = entry.maxY - entry.minY;
            if (w <= 0 || h <= 0) continue;
            boxes.push({ minX: entry.minX, minY: entry.minY, maxX: entry.maxX, maxY: entry.maxY, area: w * h });
          }
          boxes.sort((a, b) => b.area - a.area);
          setBboxes(boxes.slice(0, 128));
        }).catch((e) => {
          if (disposed) return;
          extentReadPending = false;
          console.log(`[extentRead] error: ${e}`);
        });
      };

      const scheduleQuadDetection = () => {
        if (quadDetectionPending || disposed) return;
        quadDetectionPending = true;
        const g = gpu();
        if (!g) return;
        detectContours(g, pip).then((result) => {
          if (disposed) return;
          quadDetectionPending = false;
          const { quads } = result;
          console.log('[quadDetection] quads length:', quads?.length);
          for (let i = 0; i < quads.length; i++) {
            if (!quads[i]) console.log('[quadDetection] null at index', i);
          }
          const validQuads = quads.filter((q) => q != null && typeof q.count === 'number');
          console.log('[quadDetection] valid quads:', validQuads.length);
          validQuads.sort((a, b) => b.count - a.count);
          const top = validQuads.slice(0, MAX_DETECTED_TAGS);
          console.log('[quadDetection] writing', top.length, 'quads to buffer');
          updateQuadCornersBuffer(pip, top);
        }).catch((e) => {
          if (disposed) return;
          quadDetectionPending = false;
          log(`detectContours error: ${e}`);
        });
      };

      const loop = () => {
        if (disposed) return;
        const gpuNow = gpu();
        if (!gpuNow) return;
        frameCount++;
        const dm = displayMode();
        processFrame(gpuNow, pip, cameraVideo, threshold(), dm, (err) => log(err));
        if (dm === 'debug') scheduleExtentRead();
        if (dm === 'grid' && frameCount % 30 === 0) {
          log(`frame ${frameCount} grid: sched quad`);
          scheduleQuadDetection();
        }
        void pip.histogramBuffer.read().then((bins: Uint32Array | number[]) => {
          if (disposed) return;
          const data = bins instanceof Uint32Array ? bins : new Uint32Array(bins);
          setThreshold(computeThreshold([...data], THRESHOLD_PERCENTILE));
        });
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
                  displayMode() === 'nms' ? styles.modeButtonActive : styles.modeButton
                }
                onClick={() => setDisplayMode('nms')}
              >
                NMS
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
                  displayMode() === 'grid' ? styles.modeButtonActive : styles.modeButton
                }
                onClick={() => setDisplayMode('grid')}
              >
                Grid
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
              <QuadCandidateOverlay
                bboxes={bboxes()}
                fw={frameSize().w}
                fh={frameSize().h}
                sx={scaleX()}
                sy={scaleY()}
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
            <span class={styles.thresholdLabel}>{(THRESHOLD_PERCENTILE * 100).toFixed(0)}th Percentile Threshold</span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255{' '}
              <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
          <div style={{ font: '9px monospace', color: '#4f8', 'margin-top': '4px' }}>
            {logs().map(l => <div>{l}</div>)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CalibrationView;
export { CalibrationView };
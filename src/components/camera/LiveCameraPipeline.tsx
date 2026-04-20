import {
  For,
  Show,
  type Accessor,
  type JSX,
  createEffect,
  createMemo,
  createSignal,
  createTrackedEffect,
  onCleanup,
} from "solid-js";
import { initGPU } from "../../gpu/init";
import {
  createCameraPipeline,
  processFrame,
  readExtentBuffer,
  updateQuadCornersBuffer,
  detectContours,
  type CameraPipeline,
  type DisplayMode,
  type ExtentRow,
  MAX_U32,
  MAX_DETECTED_TAGS,
} from "../../gpu/camera";
import {
  computeThreshold,
  THRESHOLD_PERCENTILE,
} from "../../gpu/pipelines/constants";
import type { DetectedQuad } from "../../gpu/contour";
import styles from "./LiveCameraPipeline.module.css";

export type { DisplayMode };

interface Bbox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  area: number;
}

function QuadCandidateOverlay(props: {
  bboxes: Bbox[];
  sx: number;
  sy: number;
}) {
  const candidates = createMemo(() => {
    const MIN_AREA = 400;
    const MAX_AREA = 200000;
    const MIN_AR = 0.3;
    const MAX_AR = 3.5;
    return props.bboxes.filter((b) => {
      const w = b.maxX - b.minX;
      const h = b.maxY - b.minY;
      if (w <= 0 || h <= 0) return false;
      const area = w * h;
      if (area < MIN_AREA || area > MAX_AREA) return false;
      const ar = w / h;
      if (ar < MIN_AR || ar > MAX_AR) return false;
      return true;
    });
  });

  return (
    <For each={candidates()} keyed={false}>
      {(box) => (
        <div
          class={styles.bbox}
          style={{
            "--bbox-x": `${box().minX * props.sx}px`,
            "--bbox-y": `${box().minY * props.sy}px`,
            "--bbox-w": `${(box().maxX - box().minX) * props.sx}px`,
            "--bbox-h": `${(box().maxY - box().minY) * props.sy}px`,
          }}
        />
      )}
    </For>
  );
}

function TagIdGridOverlay(props: {
  quads: DetectedQuad[];
  sx: number;
  sy: number;
}) {
  return (
    <For each={props.quads}>
      {(quad) => {
        const c = () => quad().corners;
        const cx = () => (c()[0]!.x + c()[1]!.x + c()[2]!.x + c()[3]!.x) / 4;
        const cy = () => (c()[0]!.y + c()[1]!.y + c()[2]!.y + c()[3]!.y) / 4;
        const label = () => {
          const q = quad();
          if (typeof q.decodedTagId === "number") return String(q.decodedTagId);
          return "?";
        };
        return (
          <div
            class={styles.tagIdOverlay}
            style={{
              "--tag-x": `${cx() * props.sx}px`,
              "--tag-y": `${cy() * props.sy}px`,
            }}
          >
            {label()}
          </div>
        );
      }}
    </For>
  );
}

type GpuRoot = Awaited<ReturnType<typeof initGPU>>;

export type LiveCameraPipelineProps = {
  displayMode: Accessor<DisplayMode>;
  showGrid: Accessor<boolean>;
  showFallbacks: Accessor<boolean>;
  showHistogramCanvas: Accessor<boolean>;
  stream: Accessor<Promise<MediaStream | undefined>>;
  trackSize: Accessor<{ w: number; h: number }>;
  onLog?: (msg: string) => void;
  onQuadDetection?: (quads: DetectedQuad[], meta: { frameId: number }) => void;
  /** Extra controls (camera select, mode buttons, …). */
  toolbar?: () => JSX.Element;
};

export function LiveCameraPipeline(props: LiveCameraPipelineProps) {
  const cameraVideo = (() => {
    const v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    return v;
  })();

  onCleanup(() => {
    cameraVideo.pause();
    cameraVideo.srcObject = null;
  });

  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>();
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>();
  const [renderedW, setRenderedW] = createSignal(1280, { ownedWrite: true });
  const [renderedH, setRenderedH] = createSignal(720, { ownedWrite: true });

  /** Set when pipeline attaches real `video.videoWidth` / `video.videoHeight`. */
  const [bufferSize, setBufferSize] = createSignal<
    { w: number; h: number } | undefined
  >(undefined, { ownedWrite: true });

  const intrinsic = createMemo(
    () => bufferSize() ?? props.trackSize(),
  );

  const scaleX = createMemo(() =>
    canvasEl() ? renderedW() / intrinsic().w : 1,
  );
  const scaleY = createMemo(() =>
    canvasEl() ? renderedH() / intrinsic().h : 1,
  );

  createTrackedEffect(() => {
    const canvas = canvasEl();
    if (!canvas) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setRenderedW(entry.contentRect.width);
        setRenderedH(entry.contentRect.height);
      }
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  });

  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true });
  const [bboxes, setBboxes] = createSignal<Bbox[]>([], { ownedWrite: true });
  const [gridOverlayQuads, setGridOverlayQuads] = createSignal<
    DetectedQuad[]
  >([], { ownedWrite: true });

  const log = (msg: string) => {
    props.onLog?.(msg);
  };

  const gpu = createMemo<GpuRoot | undefined>(async () => {
    try {
      log("GPU init...");
      const g = await initGPU();
      log("GPU ready");
      return g;
    } catch (e) {
      log(`GPU init failed: ${e}`);
      return undefined;
    }
  });

  const cameraPipeline = createMemo(async (_prev: CameraPipeline | undefined) => {
    const size = props.trackSize();
    const g = gpu();
    const canvas = canvasEl();
    const histCanvas = histCanvasEl();
    const video = cameraVideo;
    let primeHandle = 0;
    let rafHandle = 0;
    let disposed = false;
    let frameSeq = 0;

    onCleanup(() => {
      disposed = true;
      if (primeHandle) video.cancelVideoFrameCallback(primeHandle);
      if (rafHandle) video.cancelVideoFrameCallback(rafHandle);
      video.pause();
      video.srcObject = null;
      setBufferSize(undefined);
      log("Pipeline cleanup");
    });

    const stream = await props.stream();
    if (!g || !canvas || !histCanvas || !stream) {
      log("Pipeline: missing deps");
      return undefined;
    }

    histCanvas.width = 512;
    histCanvas.height = 120;

    video.srcObject = stream;
    await video.play().catch(() => {});
    if (disposed) return undefined;

    const vw =
      video.videoWidth > 0 ? video.videoWidth : Math.max(1, size.w);
    const vh =
      video.videoHeight > 0 ? video.videoHeight : Math.max(1, size.h);
    canvas.width = vw;
    canvas.height = vh;
    setBufferSize({ w: vw, h: vh });

    log("Creating pipeline...");
    const pip = createCameraPipeline(
      g,
      canvas,
      histCanvas,
      vw,
      vh,
      navigator.gpu.getPreferredCanvasFormat(),
    );
    log(`Pipeline created ${vw}x${vh}`);

    await new Promise<void>((resolve) => {
      primeHandle = video.requestVideoFrameCallback(() => {
        primeHandle = 0;
        resolve();
      });
    });
    if (disposed) return undefined;
    log("First frame presented");

    let extentReadPending = false;
    let quadDetectionPending = false;

    const scheduleExtentRead = () => {
      if (extentReadPending || disposed) return;
      extentReadPending = true;
      readExtentBuffer(pip)
        .then((extentData: ExtentRow[]) => {
          if (disposed) return;
          extentReadPending = false;
          const boxes: Bbox[] = [];
          for (const entry of extentData) {
            if (entry.minX === MAX_U32) continue;
            const w = entry.maxX - entry.minX;
            const h = entry.maxY - entry.minY;
            if (w <= 0 || h <= 0) continue;
            boxes.push({
              minX: entry.minX,
              minY: entry.minY,
              maxX: entry.maxX,
              maxY: entry.maxY,
              area: w * h,
            });
          }
          boxes.sort((a, b) => b.area - a.area);
          setBboxes(boxes.slice(0, 128));
        })
        .catch(() => {
          if (disposed) return;
          extentReadPending = false;
        });
    };

    const scheduleQuadDetection = (sf: boolean) => {
      if (quadDetectionPending || disposed) return;
      quadDetectionPending = true;
      const gNow = gpu();
      if (!gNow) return;
      detectContours(gNow, pip)
        .then((result) => {
          if (disposed) return;
          quadDetectionPending = false;
          const { quads } = result;
          const validQuads = quads.filter(
            (q) => q != null && typeof q.count === "number",
          );
          validQuads.sort((a, b) => b.count - a.count);
          const top = validQuads.slice(0, MAX_DETECTED_TAGS);
          const tagged = top.map((q) => {
            const ok =
              q.hasCorners &&
              q.cornerDebug !== null &&
              q.cornerDebug.failureCode === 0;
            return {
              ...q,
              vizTagId:
                ok && typeof q.decodedTagId === "number"
                  ? q.decodedTagId
                  : undefined,
            };
          });
          setGridOverlayQuads(
            tagged.filter((q) => {
              if (
                !q.hasCorners ||
                q.cornerDebug === null ||
                q.cornerDebug.failureCode !== 0
              ) {
                return false;
              }
              if (sf) return true;
              return typeof q.decodedTagId === "number";
            }),
          );
          updateQuadCornersBuffer(pip, tagged, sf);
          frameSeq++;
          props.onQuadDetection?.(tagged, { frameId: frameSeq });
        })
        .catch((e) => {
          if (disposed) return;
          quadDetectionPending = false;
          log(`detectContours error: ${e}`);
        });
    };

    const loop = () => {
      if (disposed) return;
      const gpuNow = gpu();
      if (!gpuNow) return;
      const dm = props.displayMode();
      processFrame(gpuNow, pip, cameraVideo, threshold(), dm, (_err) => {});
      if (dm === "debug") scheduleExtentRead();
      if (dm === "grid" && props.showGrid()) {
        scheduleQuadDetection(props.showFallbacks());
      }
      void pip.histogramBuffer.read().then((bins: Uint32Array | number[]) => {
        if (disposed) return;
        const data =
          bins instanceof Uint32Array ? bins : new Uint32Array(bins);
        setThreshold(computeThreshold([...data], THRESHOLD_PERCENTILE));
      });
      rafHandle = cameraVideo.requestVideoFrameCallback(loop);
    };
    rafHandle = cameraVideo.requestVideoFrameCallback(loop);
    log("rVFC loop started");

    return pip;
  });

  /** Subscribe so the async pipeline memo runs (lazy memos are not pulled otherwise). */
  createEffect(
    () => cameraPipeline(),
    () => {},
  );

  return (
    <div class={styles.feedRow}>
      <div class={[styles.feedPanel, styles.feedPanelMain]}>
        <div class={styles.feedHeader}>
          <span class={styles.feedLabel}>
            Camera Feed — {intrinsic().w}×{intrinsic().h}
          </span>
          {props.toolbar?.()}
        </div>
        <div style={{ position: "relative" }}>
          <canvas ref={setCanvasEl} class={styles.feedCanvas} />
          <Show when={props.displayMode() === "debug"}>
            <QuadCandidateOverlay
              bboxes={bboxes()}
              sx={scaleX()}
              sy={scaleY()}
            />
          </Show>
          <Show when={props.displayMode() === "grid" && props.showGrid()}>
            <TagIdGridOverlay
              quads={gridOverlayQuads()}
              sx={scaleX()}
              sy={scaleY()}
            />
          </Show>
        </div>
      </div>
      <div class={[styles.feedPanel, styles.feedPanelSide]}>
        <Show when={props.showHistogramCanvas()}>
          <span class={styles.feedLabel}>Edge Detection</span>
          <canvas
            ref={setHistCanvasEl}
            class={styles.histogramCanvas}
            style={{ width: "512px", height: "120px" }}
          />
          <div class={styles.histogramInfo}>
            <span class={styles.thresholdLabel}>
              {(THRESHOLD_PERCENTILE * 100).toFixed(0)}th Percentile Threshold
            </span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255{" "}
              <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
        </Show>
        <Show when={!props.showHistogramCanvas()}>
          <canvas
            ref={setHistCanvasEl}
            class={styles.histogramHidden}
            width={512}
            height={120}
            aria-hidden="true"
          />
        </Show>
      </div>
    </div>
  );
}

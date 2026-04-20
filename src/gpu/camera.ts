// Camera pipeline — fully GPU-based, single submit
import { tgpu, d, std } from "typegpu";
import { sqrt, atomicAdd, atomicStore, atomicLoad } from "typegpu/std";

import {
  EDGES_VIEW_BINARY_MASK,
  HISTOGRAM_BINS,
  POINTER_JUMP_ITERATIONS,
  COMPUTE_WORKGROUP_SIZE,
  computeDispatch2d,
  computeThreshold,
} from "./pipelines/constants";
import { createLayouts } from "./pipelines/layouts";
import { createCopyPipeline } from "./pipelines/copyPipeline";
import { createGrayPipeline } from "./pipelines/grayPipeline";
import { createSobelPipeline } from "./pipelines/sobelPipeline";
import {
  createHistogramResetPipeline,
  createHistogramAccumulatePipeline,
} from "./pipelines/histogramPipelines";
import { createEdgesPipeline } from "./pipelines/edgesPipeline";
import { createEdgeFilterPipeline } from "./pipelines/edgeFilterPipeline";
import { createHistogramRenderPipeline } from "./pipelines/histogramRenderPipeline";
import { createGrayRenderPipeline } from "./pipelines/grayRenderPipeline";
import { createSobelRenderPipeline } from "./pipelines/sobelRenderPipeline";
import { createFilteredRenderPipeline } from "./pipelines/filteredRenderPipeline";
import { createLabelVizPipeline } from "./pipelines/labelVizPipeline";
import { createEdgeDilatePipeline } from "./pipelines/edgeDilatePipeline";
import {
  createPointerJumpLayouts,
  createPointerJumpInitPipeline,
  createPointerJumpStepPipeline,
  createPointerJumpLabelsToAtomicPipeline,
  createPointerJumpParentTightenPipeline,
  createPointerJumpAtomicToLabelsPipeline,
} from "./pipelines/pointerJumpPipeline";
import {
  createExtentTrackingLayouts,
  createExtentResetPipeline,
  createExtentTrackPipeline,
  ExtentEntry,
} from "./pipelines/extentTrackingPipeline";
import {
  createCompactLabelLayouts,
  createCanonicalResetPipeline,
  createCanonicalClaimPipeline,
  createRemapLabelPipeline,
} from "./pipelines/compactLabelPipeline";
import {
  createGridVizPipeline,
  createGridVizLayouts,
  DECODED_TAG_ID_UNKNOWN,
  GridDataSchema,
  QuadData,
  MAX_INSTANCES,
  type GridVizFailInterrogateMode,
} from "./pipelines/gridVizPipeline";
import { tryComputeHomography } from "../lib/geometry";
import {
  type DetectedQuad,
  filterNestedQuads,
  extractRegions,
  validateAndFilterQuads,
} from "./contour";

/** Max quads drawn in grid mode; must match `MAX_INSTANCES` in gridVizPipeline (buffer + draw). */
export const MAX_DETECTED_TAGS = MAX_INSTANCES;

export type DisplayMode = "edges" | "nms" | "labels" | "grayscale" | "debug" | "grid";

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE FACTORY
// ═══════════════════════════════════════════════════════════════════════════
export function createCameraPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  canvas: HTMLCanvasElement,
  histCanvas: HTMLCanvasElement,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  // Configure contexts on canvases
  const context = root.configureContext({ canvas, alphaMode: "premultiplied" });
  const histContext = root.configureContext({ canvas: histCanvas });
  // console.log(`[camera] presentationFormat=${presentationFormat}, canvas=${canvas.width}x${canvas.height}`);

  // ═══════════════════════════════════════════════════════════════════════
  // RESOURCES
  // ═══════════════════════════════════════════════════════════════════════

  // Intermediate RGBA texture for external → usable format
  const grayTex = root
    .createTexture({ size: [width, height], format: "rgba8unorm", dimension: "2d" })
    .$usage("storage", "sampled", "render");

  const sampler = root.createSampler({ minFilter: "linear", magFilter: "linear" });

  const grayBuffer = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage("storage");

  const sobelBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage("storage");

  const filteredBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage("storage");

  // Staging buffer for CPU readback of filtered edge gradients
  const filteredStaging = root.device.createBuffer({
    size: width * height * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const dilatedEdgeBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage("storage");

  const thresholdBuffer = root.createBuffer(d.f32).$usage("uniform");

  const histogramSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS);
  const histogramBuffer = root.createBuffer(histogramSchema).$usage("storage");

  const thresholdBinBuffer = root.createBuffer(d.u32).$usage("uniform");

  const pointerJumpBuffer0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");
  const pointerJumpBuffer1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const pointerJumpAtomicBuffer = root
    .createBuffer(d.arrayOf(d.atomic(d.u32), width * height))
    .$usage("storage");

  const {
    initLayout: pointerJumpInitLayout,
    stepLayout: pointerJumpStepLayout,
    labelsToAtomicLayout: pointerJumpLabelsToAtomicLayout,
    parentTightenLayout: pointerJumpParentTightenLayout,
    atomicToLabelsLayout: pointerJumpAtomicToLabelsLayout,
  } = createPointerJumpLayouts(root);
  const pointerJumpInitPipeline = createPointerJumpInitPipeline(
    root,
    pointerJumpInitLayout,
    width,
    height,
  );
  const pointerJumpStepPipeline = createPointerJumpStepPipeline(
    root,
    pointerJumpStepLayout,
    width,
    height,
  );
  const pointerJumpLabelsToAtomicPipeline = createPointerJumpLabelsToAtomicPipeline(
    root,
    pointerJumpLabelsToAtomicLayout,
    width,
    height,
  );
  const pointerJumpParentTightenPipeline = createPointerJumpParentTightenPipeline(
    root,
    pointerJumpParentTightenLayout,
    width,
    height,
  );
  const pointerJumpAtomicToLabelsPipeline = createPointerJumpAtomicToLabelsPipeline(
    root,
    pointerJumpAtomicToLabelsLayout,
    width,
    height,
  );

  const pointerJumpInitBindGroup = root.createBindGroup(pointerJumpInitLayout, {
    edgeBuffer: filteredBuffer,
    labelBuffer: pointerJumpBuffer0,
  });

  const pointerJumpPingPongBindGroups = [
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer0,
      writeBuffer: pointerJumpBuffer1,
    }),
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer1,
      writeBuffer: pointerJumpBuffer0,
    }),
  ];

  const pointerJumpLabelsToAtomicBindGroups = [
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ];
  const pointerJumpParentTightenBindGroups = [
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ];
  const pointerJumpAtomicToLabelsBindGroups = [
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer0,
    }),
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer1,
    }),
  ];

  // ─── Extent tracking (per-component bounding boxes) ─────────────────────
  // Compact labeling: raw pixel-index labels → compact IDs (0..N-1).
  // canonicalRoot: one entry per possible root pixel index (area entries, max ~921K).
  // compactLabelBuffer: remapped labels after compact pass.
  // extentBuffer: sized for MAX_EXTENT_COMPONENTS entries (compact IDs < MAX_EXTENT_COMPONENTS).
  const area = width * height;

  const canonicalRootBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), area)).$usage("storage");

  const compactLabelBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage("storage");

  // Staging buffers for CPU readback (requires COPY_DST | MAP_READ)
  const compactLabelStaging = root.device.createBuffer({
    size: area * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const compactCounterBuffer = root.createBuffer(d.atomic(d.u32)).$usage("storage");

  const { trackLayout: extentTrackLayout } = createExtentTrackingLayouts(root);

  const extentResetLayout = tgpu.bindGroupLayout({
    extentBuffer: { storage: d.arrayOf(ExtentEntry), access: "mutable" },
  });

  const extentBuffer = root
    .createBuffer(d.arrayOf(ExtentEntry, MAX_EXTENT_COMPONENTS))
    .$usage("storage");

  const extentResetPipeline = createExtentResetPipeline(
    root,
    extentResetLayout,
    MAX_EXTENT_COMPONENTS,
  );
  const extentTrackPipeline = createExtentTrackPipeline(
    root,
    extentTrackLayout,
    width,
    height,
    MAX_EXTENT_COMPONENTS,
  );

  const extentResetBindGroup = root.createBindGroup(extentResetLayout, {
    extentBuffer,
  });
  // Extent tracking on compact labels
  const extentTrackBindGroup = root.createBindGroup(extentTrackLayout, {
    labelBuffer: compactLabelBuffer,
    extentBuffer,
  });

  // Compact labeling: remap raw pixel-index labels to compact IDs (0..N-1)
  const { resetLayout, claimLayout, remapLayout } = createCompactLabelLayouts(root);
  const compactResetPipeline = createCanonicalResetPipeline(root, resetLayout, area);
  const compactClaimPipeline = createCanonicalClaimPipeline(
    root,
    claimLayout,
    width,
    height,
    MAX_EXTENT_COMPONENTS,
  );
  const compactRemapPipeline = createRemapLabelPipeline(root, remapLayout, width, height);

  // compactCounter: single u32 atomic counter for next compact ID
  const compactResetBindGroup = root.createBindGroup(resetLayout, {
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  });
  const compactClaimBindGroup = root.createBindGroup(claimLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  });
  const compactRemapBindGroup = root.createBindGroup(remapLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactLabelBuffer: compactLabelBuffer,
    canonicalRoot: canonicalRootBuffer,
  });

  // compactLabelBuffer is the remapped output used by extent tracking

  // ─── Grid visualization (AprilTag grid overlay) ─────────────────────────
  const quadCornersBuffer = root.createBuffer(GridDataSchema).$usage("storage");

  const { gridVizLayout } = createGridVizLayouts(root, quadCornersBuffer);

  const gridVizDebugModeBuffer = root.createBuffer(d.u32).$usage("uniform");
  gridVizDebugModeBuffer.write(0);

  const gridVizPipeline = createGridVizPipeline(
    root,
    gridVizLayout,
    width,
    height,
    presentationFormat,
  );

  const gridVizBindGroup = root.createBindGroup(gridVizLayout, {
    quads: quadCornersBuffer,
    failInterrogate: gridVizDebugModeBuffer,
  });

  // ═══════════════════════════════════════════════════════════════════════
  // LAYOUTS & PIPELINES
  // ═══════════════════════════════���═══════════════════════════════════════

  const {
    copyLayout,
    grayTexToBufferLayout,
    sobelLayout,
    histogramLayout,
    histogramResetLayout,
    edgesLayout,
    histogramDisplayLayout,
    edgeFilterLayout,
    edgeDilateLayout,
    labelVizLayout,
    grayRenderLayout,
    sobelRenderLayout,
    filteredRenderLayout,
  } = createLayouts(root, histogramSchema);

  const copyPipeline = createCopyPipeline(root, copyLayout);
  const grayPipeline = createGrayPipeline(root, grayTexToBufferLayout, width, height);
  const sobelPipeline = createSobelPipeline(root, sobelLayout, width, height);
  const histogramResetPipeline = createHistogramResetPipeline(root, histogramResetLayout);
  const histogramPipeline = createHistogramAccumulatePipeline(root, histogramLayout, width, height);
  const edgesPipeline = createEdgesPipeline(root, edgesLayout, width, height, presentationFormat);
  const edgeFilterPipeline = createEdgeFilterPipeline(root, edgeFilterLayout, width, height);
  const edgeDilatePipeline = createEdgeDilatePipeline(root, edgeDilateLayout, width, height);
  const histogramDisplayPipeline = createHistogramRenderPipeline(
    root,
    histogramDisplayLayout,
    presentationFormat,
    width * height,
  );
  const labelVizPipeline = createLabelVizPipeline(
    root,
    labelVizLayout,
    width,
    height,
    presentationFormat,
  );
  const grayRenderPipeline = createGrayRenderPipeline(
    root,
    grayRenderLayout,
    width,
    height,
    presentationFormat,
  );
  const sobelRenderPipeline = createSobelRenderPipeline(
    root,
    sobelRenderLayout,
    width,
    height,
    presentationFormat,
  );
  const filteredRenderPipeline = createFilteredRenderPipeline(
    root,
    filteredRenderLayout,
    width,
    height,
    presentationFormat,
  );

  // ═══════════════════════════════════════════════════════════════════════
  // BIND GROUPS
  // ═══════════════════════════════════════════════════════════════════════

  // Copy (recreated per-frame for external texture)
  const copyLayoutTemplate = copyLayout;

  const grayTexToBufferBindGroup = root.createBindGroup(grayTexToBufferLayout, {
    grayTex: grayTex,
    grayBuffer: grayBuffer,
  });

  const sobelBindGroup = root.createBindGroup(sobelLayout, {
    grayBuffer: grayBuffer,
    sobelBuffer: sobelBuffer,
  });

  const histogramResetBindGroup = root.createBindGroup(histogramResetLayout, {
    histogram: histogramBuffer,
  });

  const histogramComputeBindGroup = root.createBindGroup(histogramLayout, {
    sobelBuffer: sobelBuffer,
    histogram: histogramBuffer,
  });

  const edgesBindGroup = root.createBindGroup(edgesLayout, {
    sobelBuffer: sobelBuffer,
    filteredBuffer: filteredBuffer,
  });

  // Note: edgesDilatedBindGroup is the same as edgesBindGroup since we removed dilation.
  // It binds filteredBuffer (NMS output) for display — labels mode already reads filteredBuffer.
  const edgesDilatedBindGroup = edgesBindGroup;

  const edgeFilterBindGroup = root.createBindGroup(edgeFilterLayout, {
    sobelBuffer: sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer: filteredBuffer,
  });

  const edgeDilateBindGroup = root.createBindGroup(edgeDilateLayout, {
    src: filteredBuffer,
    grad: filteredBuffer,
    threshold: thresholdBuffer,
    dst: dilatedEdgeBuffer,
  });

  const histogramDisplayBindGroup = root.createBindGroup(histogramDisplayLayout, {
    histogram: histogramBuffer,
    thresholdBin: thresholdBinBuffer,
  });

  const grayRenderBindGroup = root.createBindGroup(grayRenderLayout, {
    grayBuffer: grayBuffer,
  });

  const sobelRenderBindGroup = root.createBindGroup(sobelRenderLayout, {
    sobelBuffer: sobelBuffer,
  });

  const filteredRenderBindGroup = root.createBindGroup(filteredRenderLayout, {
    filteredBuffer: filteredBuffer,
  });

  const labelVizBindGroup = root.createBindGroup(labelVizLayout, {
    labelBuffer: pointerJumpBuffer0,
  });

  return {
    context,
    histContext,
    grayTex,
    grayBuffer,
    sobelBuffer,
    filteredBuffer,
    compactLabelStaging,
    filteredStaging,
    dilatedEdgeBuffer,
    thresholdBuffer,
    thresholdBinBuffer,
    histogramBuffer,
    copyPipeline,
    copyLayoutTemplate,
    grayPipeline,
    grayTexToBufferBindGroup,
    sobelPipeline,
    sobelBindGroup,
    histogramResetPipeline,
    histogramResetBindGroup,
    histogramPipeline,
    histogramComputeBindGroup,
    edgesPipeline,
    edgesBindGroup,
    edgesDilatedBindGroup,
    edgeFilterPipeline,
    edgeFilterBindGroup,
    edgeDilatePipeline,
    edgeDilateBindGroup,
    histogramDisplayPipeline,
    histogramDisplayBindGroup,
    labelVizPipeline,
    labelVizBindGroup,
    labelVizLayout,
    sampler,
    width,
    height,
    histWidth: 512,
    histHeight: 120,
    // Grayscale render
    grayRenderPipeline,
    grayRenderBindGroup,
    grayRenderLayout,
    sobelRenderPipeline,
    sobelRenderBindGroup,
    sobelRenderLayout,
    filteredRenderPipeline,
    filteredRenderBindGroup,
    filteredRenderLayout,
    pointerJumpBuffer0,
    pointerJumpBuffer1,
    pointerJumpAtomicBuffer,
    pointerJumpInitPipeline,
    pointerJumpInitBindGroup,
    pointerJumpStepPipeline,
    pointerJumpPingPongBindGroups,
    pointerJumpLabelsToAtomicPipeline,
    pointerJumpLabelsToAtomicBindGroups,
    pointerJumpParentTightenPipeline,
    pointerJumpParentTightenBindGroups,
    pointerJumpAtomicToLabelsPipeline,
    pointerJumpAtomicToLabelsBindGroups,
    extentBuffer,
    extentResetPipeline,
    extentResetBindGroup,
    extentTrackPipeline,
    extentTrackBindGroup,
    canonicalRootBuffer,
    compactLabelBuffer,
    compactResetPipeline,
    compactResetBindGroup,
    compactClaimPipeline,
    compactClaimBindGroup,
    compactRemapPipeline,
    compactRemapBindGroup,
    // Grid visualization
    gridVizPipeline,
    gridVizLayout,
    gridVizBindGroup,
    quadCornersBuffer,
    gridVizDebugModeBuffer,
  };
}

export const MAX_EXTENT_COMPONENTS = 16384;
export const MAX_COMPONENTS = 16384; // alias for CalibrationView readback
export const EXTENT_FIELDS = 4; // 4 fields per extent entry: minX, minY, maxX, maxY

export const MAX_U32 = 0xffffffff;

export type CameraPipeline = ReturnType<typeof createCameraPipeline>;

// ═══════════════════════════════════════════════════════════════════════════
// PER-FAME PROCESSING
// ═══════════════════════════════════════════════════════════════════════════
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  displayMode: DisplayMode = "edges",
  onError?: (msg: string) => void,
) {
  // onError?.(`[camera] processFrame mode=${displayMode}`);
  const copyBindGroup = root.createBindGroup(pipeline.copyLayoutTemplate, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  });

  // Update uniforms
  pipeline.thresholdBuffer.write(threshold);
  const thresholdBin = Math.round(threshold * 255);
  pipeline.thresholdBinBuffer.write(thresholdBin);

  const enc = root.device.createCommandEncoder({ label: "camera frame" });

  // RENDER: Copy external → grayTex (MUST happen before compute)
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3);

  // COMPUTE: Gray → Sobel → Histogram + pointer-jump labeling (when Labels/Debug)
  let finalLabelBuffer = pipeline.pointerJumpBuffer0;
  {
    const computePass = enc.beginComputePass({ label: "gray + sobel + histogram + filter" });
    const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height);
    const area = pipeline.width * pipeline.height;

    pipeline.grayPipeline
      .with(computePass)
      .with(pipeline.grayTexToBufferBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    pipeline.sobelPipeline
      .with(computePass)
      .with(pipeline.sobelBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    pipeline.histogramResetPipeline
      .with(computePass)
      .with(pipeline.histogramResetBindGroup)
      .dispatchWorkgroups(HISTOGRAM_BINS);

    pipeline.histogramPipeline
      .with(computePass)
      .with(pipeline.histogramComputeBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    pipeline.edgeFilterPipeline
      .with(computePass)
      .with(pipeline.edgeFilterBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    // Pointer-jump connected component labeling
    pipeline.pointerJumpInitPipeline
      .with(computePass)
      .with(pipeline.pointerJumpInitBindGroup)
      .dispatchWorkgroups(wgX, wgY);
    let pj = 0;
    for (let s = 0; s < POINTER_JUMP_ITERATIONS; s++) {
      pipeline.pointerJumpStepPipeline
        .with(computePass)
        .with(pipeline.pointerJumpPingPongBindGroups[pj])
        .dispatchWorkgroups(wgX, wgY);
      pj ^= 1;
      pipeline.pointerJumpLabelsToAtomicPipeline
        .with(computePass)
        .with(pipeline.pointerJumpLabelsToAtomicBindGroups[pj])
        .dispatchWorkgroups(wgX, wgY);
      pipeline.pointerJumpParentTightenPipeline
        .with(computePass)
        .with(pipeline.pointerJumpParentTightenBindGroups[pj])
        .dispatchWorkgroups(wgX, wgY);
      pipeline.pointerJumpAtomicToLabelsPipeline
        .with(computePass)
        .with(pipeline.pointerJumpAtomicToLabelsBindGroups[pj])
        .dispatchWorkgroups(wgX, wgY);
    }
    finalLabelBuffer = pj === 0 ? pipeline.pointerJumpBuffer0 : pipeline.pointerJumpBuffer1;

    // Compact labeling: 3-pass remap of pixel-index labels to compact IDs (0..N-1).
    // Always runs — needed for extent tracking to work (labels must fit in extent buffer).
    // 1. Reset canonicalRoot to INVALID
    pipeline.compactResetPipeline
      .with(computePass)
      .with(pipeline.compactResetBindGroup)
      .dispatchWorkgroups(Math.ceil(area / COMPUTE_WORKGROUP_SIZE));
    // 2. Claim: each root (label == pixel idx) stores its own index as compact ID
    pipeline.compactClaimPipeline
      .with(computePass)
      .with(pipeline.compactClaimBindGroup)
      .dispatchWorkgroups(wgX, wgY);
    // 3. Remap: L[i] = canonicalRoot[label] (compact ID, fits in extent buffer)
    pipeline.compactRemapPipeline
      .with(computePass)
      .with(pipeline.compactRemapBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    // Use compact labels everywhere downstream
    finalLabelBuffer = pipeline.compactLabelBuffer;

    // Track extents on compact labels
    pipeline.extentResetPipeline
      .with(computePass)
      .with(pipeline.extentResetBindGroup)
      .dispatchWorkgroups(Math.ceil(MAX_EXTENT_COMPONENTS / COMPUTE_WORKGROUP_SIZE));
    pipeline.extentTrackPipeline
      .with(computePass)
      .with(pipeline.extentTrackBindGroup)
      .dispatchWorkgroups(wgX, wgY);

    computePass.end();
  }

  // RENDER: Display mode selection + Histogram
  // onError?.(`[camera] render mode=${displayMode}`);
  if (displayMode === "edges") {
    try {
      pipeline.sobelRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.sobelRenderBindGroup)
        .draw(3);
    } catch (e) {
      const msg = `[camera] sobelRender failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  } else if (displayMode === "nms") {
    try {
      pipeline.edgesPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.edgesBindGroup)
        .draw(3);
    } catch (e) {
      const msg = `[camera] edgesPipeline (nms) failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  } else if (displayMode === "labels") {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: finalLabelBuffer,
    });
    try {
      pipeline.labelVizPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(labelVizBindGroup)
        .draw(3);
    } catch (e) {
      const msg = `[camera] labelVizPipeline failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  } else if (displayMode === "debug") {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: finalLabelBuffer,
    });
    try {
      pipeline.labelVizPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(labelVizBindGroup)
        .draw(3);
    } catch (e) {
      const msg = `[camera] labelVizPipeline (debug) failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  } else if (displayMode === "grid") {
    // Grayscale base with grid overlay on top (alpha blend)
    pipeline.grayRenderPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context, loadOp: "load", storeOp: "store" })
      .with(pipeline.grayRenderBindGroup)
      .draw(3);

    try {
      // console.log('[camera] gridViz: drawing full-canvas overlay');
      pipeline.gridVizPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context, loadOp: "load", storeOp: "store" })
        .with(pipeline.gridVizBindGroup)
        .draw(4, MAX_DETECTED_TAGS); // triangle strip quad per instance
    } catch (e) {
      const msg = `[camera] gridViz failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  } else {
    try {
      pipeline.grayRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.grayRenderBindGroup)
        .draw(3);
    } catch (e) {
      const msg = `[camera] grayRender fallback failed: ${e}`;
      console.error(msg);
      onError?.(msg);
    }
  }

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS);

  root.device.queue.submit([enc.finish()]);
}

/**
 * Read the extent buffer. Call periodically (not every frame) to get bounding boxes.
 */
export type ExtentRow = d.Infer<typeof ExtentEntry>;

export async function readExtentBuffer(pipeline: CameraPipeline): Promise<ExtentRow[]> {
  return pipeline.extentBuffer.read();
}

/**
 * Read extent buffer and filter to valid quad candidates.
 * Uses the same bboxes that appear in the debug view.
 */
export async function readExtentDataForQuads(pipeline: CameraPipeline): Promise<ExtentRow[]> {
  const all = await pipeline.extentBuffer.read();
  return all.filter((e) => e.minX !== MAX_U32);
}

// ─────────────────────────────────────────────────────────────────────────────
// Update quad corners buffer for grid visualization (instanced rendering)
// ─────────────────────────────────────────────────────────────────────────────

/** Write homography matrix (mat3x3) + debug data per quad to the GPU buffer.
 * H maps unit square → detected quad. Vertex shader applies: (x,y,z) = H * (u,v,1).
 * H is column-major mat3x3f: [c0.x, c0.y, c0.z, 0, c1..., c2...]
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  quads: DetectedQuad[],
  showFallbacks: boolean = true,
  log: (msg: string) => void = () => {},
): void {
  const filtered = showFallbacks
    ? quads
    : quads.filter((q) => q.hasCorners && typeof q.decodedTagId === "number");
  const count = Math.min(filtered.length, MAX_INSTANCES);

  const data: QuadData[] = [];
  for (let i = 0; i < count; i++) {
    const quad = filtered[i];
    const H = tryComputeHomography(quad.corners);
    const debug = quad.cornerDebug;
    const tagId =
      quad.vizTagId !== undefined && quad.vizTagId !== null
        ? quad.vizTagId >>> 0
        : DECODED_TAG_ID_UNKNOWN;

    data.push({
      homography: H
        ? d.mat3x3f(
            // transpose the matrix
            H[0],
            H[3],
            H[6],
            H[1],
            H[4],
            H[7],
            H[2],
            H[5],
            1,
          )
        : d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: debug ? debug.failureCode : 0,
        edgePixelCount: debug ? debug.edgePixelCount / 100 : 0,
        minR2: debug ? debug.minR2 : 0,
        intersectionCount: debug ? debug.intersectionCount : 0,
      },
      decodedTagId: d.u32(tagId),
    });
  }

  // Pad remaining slots with zeros
  for (let i = count; i < MAX_INSTANCES; i++) {
    data.push({
      homography: d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: 0,
        edgePixelCount: 0,
        minR2: 0,
        intersectionCount: 0,
      },
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
    });
  }

  pipeline.quadCornersBuffer.write(data);
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(
  pipeline: CameraPipeline,
  mode: GridVizFailInterrogateMode,
): void {
  pipeline.gridVizDebugModeBuffer.write(mode);
}

export { filterNestedQuads } from "./contour";

export async function detectContours(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
): Promise<{
  quads: DetectedQuad[];
  extentData: ExtentRow[];
  dilatedGradients: Float32Array;
  labelData: Uint32Array;
}> {
  try {
    // Copy storage buffers → staging buffers (staging has MAP_READ, storage does not)
    const enc = root.device.createCommandEncoder({ label: "readback" });
    const labelStorage = pipeline.compactLabelBuffer.buffer as GPUBuffer;
    enc.copyBufferToBuffer(labelStorage, 0, pipeline.compactLabelStaging, 0, labelStorage.size);
    const edgeStorage = pipeline.filteredBuffer.buffer as GPUBuffer;
    enc.copyBufferToBuffer(edgeStorage, 0, pipeline.filteredStaging, 0, edgeStorage.size);
    root.device.queue.submit([enc.finish()]);
    await root.device.queue.onSubmittedWorkDone();

    // Read from staging buffers — no TypeGPU wrapper, no per-element toString()
    await pipeline.compactLabelStaging.mapAsync(GPUMapMode.READ);
    const labelData = new Uint32Array(pipeline.compactLabelStaging.getMappedRange());
    const labelDataCopy = new Uint32Array(labelData);
    pipeline.compactLabelStaging.unmap();

    await pipeline.filteredStaging.mapAsync(GPUMapMode.READ);
    const dilatedGradients = new Float32Array(pipeline.filteredStaging.getMappedRange());
    const dilatedCopy = new Float32Array(dilatedGradients);
    pipeline.filteredStaging.unmap();

    // CPU-side: extract regions and fit quads
    const regions = extractRegions(labelDataCopy, pipeline.width, pipeline.height, dilatedCopy);
    const maxArea = pipeline.width * pipeline.height * 0.5;
    const quads = validateAndFilterQuads(
      regions,
      dilatedCopy,
      labelDataCopy,
      pipeline.width,
      400,
      maxArea,
    ).filter((q) => q.area < pipeline.width * pipeline.height * 0.25);

    // Read extent buffer
    const extentData: ExtentRow[] = await pipeline.extentBuffer.read();

    return { quads, extentData, dilatedGradients: dilatedCopy, labelData: labelDataCopy };
  } catch (e) {
    console.error("[detectContours] Error:", e);
    throw e;
  }
}

// (empty space)

export { computeThreshold };

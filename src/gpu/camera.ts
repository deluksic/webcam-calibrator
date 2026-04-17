// Camera pipeline — fully GPU-based, single submit
import { tgpu, d, std } from 'typegpu';
import { sqrt, atomicAdd, atomicStore, atomicLoad } from 'typegpu/std';

import {
  EDGES_VIEW_BINARY_MASK,
  HISTOGRAM_BINS,
  POINTER_JUMP_ITERATIONS,
  COMPUTE_WORKGROUP_SIZE,
  computeDispatch2d,
  computeThreshold,
} from './pipelines/constants';
import { createLayouts } from './pipelines/layouts';
import { createCopyPipeline } from './pipelines/copyPipeline';
import { createGrayPipeline } from './pipelines/grayPipeline';
import { createSobelPipeline } from './pipelines/sobelPipeline';
import { createHistogramResetPipeline, createHistogramAccumulatePipeline } from './pipelines/histogramPipelines';
import { createEdgesPipeline } from './pipelines/edgesPipeline';
import { createEdgeFilterPipeline } from './pipelines/edgeFilterPipeline';
import { createHistogramRenderPipeline } from './pipelines/histogramRenderPipeline';
import { createGrayRenderPipeline } from './pipelines/grayRenderPipeline';
import { createSobelRenderPipeline } from './pipelines/sobelRenderPipeline';
import { createFilteredRenderPipeline } from './pipelines/filteredRenderPipeline';
import { createLabelVizPipeline } from './pipelines/labelVizPipeline';
import { createEdgeDilatePipeline } from './pipelines/edgeDilatePipeline';
import {
  createPointerJumpLayouts,
  createPointerJumpInitPipeline,
  createPointerJumpStepPipeline,
  createPointerJumpLabelsToAtomicPipeline,
  createPointerJumpParentTightenPipeline,
  createPointerJumpAtomicToLabelsPipeline,
} from './pipelines/pointerJumpPipeline';
import {
  createExtentTrackingLayouts,
  createExtentResetPipeline,
  createExtentTrackPipeline,
  ExtentEntry,
} from './pipelines/extentTrackingPipeline';
import { createCompactLabelLayouts, createCanonicalResetPipeline, createCanonicalClaimPipeline, createRemapLabelPipeline } from './pipelines/compactLabelPipeline';
import { createGridVizPipeline, createGridVizLayouts, gridCornersSchema, MAX_INSTANCES } from './pipelines/gridVizPipeline';
import { computeProjectiveWeights, type Point } from '../lib/geometry';
import {
  type DetectedQuad,
  filterNestedQuads,
  extractRegions,
  validateAndFilterQuads,
} from './contour';

export const MAX_DETECTED_TAGS = 64;

export type DisplayMode = 'edges' | 'nms' | 'labels' | 'grayscale' | 'debug' | 'grid';

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE FACTORY
// ═══════════════════════════════════════════════════════════════════════════
export function createCameraPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  canvas: HTMLCanvasElement,
  histCanvas: HTMLCanvasElement,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat
) {
  // Configure contexts on canvases
  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
  const histContext = root.configureContext({ canvas: histCanvas });
  // console.log(`[camera] presentationFormat=${presentationFormat}, canvas=${canvas.width}x${canvas.height}`);

  // ═══════════════════════════════════════════════════════════════════════
  // RESOURCES
  // ═══════════════════════════════════════════════════════════════════════

  // Intermediate RGBA texture for external → usable format
  const grayTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('storage', 'sampled', 'render');

  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  const grayBuffer = root
    .createBuffer(d.arrayOf(d.f32, width * height))
    .$usage('storage');

  const sobelBuffer = root
    .createBuffer(d.arrayOf(d.vec2f, width * height))
    .$usage('storage');

  const filteredBuffer = root
    .createBuffer(d.arrayOf(d.vec2f, width * height))
    .$usage('storage');

  // Staging buffer for CPU readback of filtered edge gradients
  const filteredStaging = root.device.createBuffer({
    size: width * height * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const dilatedEdgeBuffer = root
    .createBuffer(d.arrayOf(d.vec2f, width * height))
    .$usage('storage');

  const thresholdBuffer = root
    .createBuffer(d.f32)
    .$usage('uniform');

  const histogramSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS);
  const histogramBuffer = root.createBuffer(histogramSchema).$usage('storage');

  const thresholdBinBuffer = root
    .createBuffer(d.u32)
    .$usage('uniform');

  const pointerJumpBuffer0 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage('storage');
  const pointerJumpBuffer1 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage('storage');

  const pointerJumpAtomicBuffer = root
    .createBuffer(d.arrayOf(d.atomic(d.u32), width * height))
    .$usage('storage');

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

  const canonicalRootBuffer = root
    .createBuffer(d.arrayOf(d.atomic(d.u32), area))
    .$usage('storage');

  const compactLabelBuffer = root
    .createBuffer(d.arrayOf(d.u32, area))
    .$usage('storage');

  // Staging buffers for CPU readback (requires COPY_DST | MAP_READ)
  const compactLabelStaging = root.device.createBuffer({
    size: area * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const compactCounterBuffer = root
    .createBuffer(d.atomic(d.u32))
    .$usage('storage');

  const { trackLayout: extentTrackLayout } = createExtentTrackingLayouts(root);

  const extentResetLayout = tgpu.bindGroupLayout({
    extentBuffer: { storage: d.arrayOf(ExtentEntry), access: 'mutable' },
  });

  const extentBuffer = root
    .createBuffer(d.arrayOf(ExtentEntry, MAX_EXTENT_COMPONENTS))
    .$usage('storage');

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
  const compactRemapPipeline = createRemapLabelPipeline(
    root,
    remapLayout,
    width,
    height,
  );

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
  const quadCornersBuffer = root
    .createBuffer(gridCornersSchema)
    .$usage('storage');

  const { gridVizLayout } = createGridVizLayouts(root, quadCornersBuffer);

  const gridVizPipeline = createGridVizPipeline(
    root,
    gridVizLayout,
    width,
    height,
    presentationFormat,
  );

  const gridVizBindGroup = root.createBindGroup(gridVizLayout, {
    quadCorners: quadCornersBuffer,
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
  const edgesPipeline = createEdgesPipeline(
    root,
    edgesLayout,
    width,
    height,
    presentationFormat,
  );
  const edgeFilterPipeline = createEdgeFilterPipeline(root, edgeFilterLayout, width, height);
  const edgeDilatePipeline = createEdgeDilatePipeline(root, edgeDilateLayout, width, height);
  const histogramDisplayPipeline = createHistogramRenderPipeline(root, histogramDisplayLayout, presentationFormat, width * height);
  const labelVizPipeline = createLabelVizPipeline(root, labelVizLayout, width, height, presentationFormat);
  const grayRenderPipeline = createGrayRenderPipeline(root, grayRenderLayout, width, height, presentationFormat);
  const sobelRenderPipeline = createSobelRenderPipeline(root, sobelRenderLayout, width, height, presentationFormat);
  const filteredRenderPipeline = createFilteredRenderPipeline(root, filteredRenderLayout, width, height, presentationFormat);

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
  };
}

export const MAX_EXTENT_COMPONENTS = 16384;
export const MAX_COMPONENTS = 16384; // alias for CalibrationView readback
export const EXTENT_FIELDS = 4; // 4 fields per extent entry: minX, minY, maxX, maxY

export const MAX_U32 = 0xFFFFFFFF;

export type CameraPipeline = ReturnType<typeof createCameraPipeline>;

// ═══════════════════════════════════════════════════════════════════════════
// PER-FAME PROCESSING
// ═══════════════════════════════════════════════════════════════════════════
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  displayMode: DisplayMode = 'edges',
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

  const enc = root.device.createCommandEncoder({ label: 'camera frame' });

  // RENDER: Copy external → grayTex (MUST happen before compute)
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3);

  // COMPUTE: Gray → Sobel → Histogram + pointer-jump labeling (when Labels/Debug)
  let finalLabelBuffer = pipeline.pointerJumpBuffer0;
  {
    const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' });
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
  if (displayMode === 'edges') {
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
  } else if (displayMode === 'nms') {
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
  } else if (displayMode === 'labels') {
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
  } else if (displayMode === 'debug') {
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
  } else if (displayMode === 'grid') {
    // Grayscale base with grid overlay on top (alpha blend)
    pipeline.grayRenderPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
      .with(pipeline.grayRenderBindGroup)
      .draw(3);

    try {
      // console.log('[camera] gridViz: drawing full-canvas overlay');
      pipeline.gridVizPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
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

export async function readExtentBuffer(
  pipeline: CameraPipeline,
): Promise<ExtentRow[]> {
  return pipeline.extentBuffer.read();
}

/**
 * Read extent buffer and filter to valid quad candidates.
 * Uses the same bboxes that appear in the debug view.
 */
export async function readExtentDataForQuads(
  pipeline: CameraPipeline,
): Promise<ExtentRow[]> {
  const all = await pipeline.extentBuffer.read();
  return all.filter(e => e.minX !== MAX_U32);
}

// ─────────────────────────────────────────────────────────────────────────────
// Update quad corners buffer for grid visualization (instanced rendering)
// ─────────────────────────────────────────────────────────────────────────────

/** Per-quad data for projective grid rendering.
 * 3 vec4f per quad: corners + weights.
 */
export const QuadCornersEntry = d.struct({
  x0: d.f32, y0: d.f32, x1: d.f32, y1: d.f32, // TL, TR
  x2: d.f32, y2: d.f32, x3: d.f32, y3: d.f32, // BL, BR
  w0: d.f32, w1: d.f32, w2: d.f32, w3: d.f32, // weights
});

/** Plain data object for writing to the buffer. */
export interface QuadCornersData {
  x0: number; y0: number; x1: number; y1: number;
  x2: number; y2: number; x3: number; y3: number;
  w0: number; w1: number; w2: number; w3: number;
}

/** Write quad corner data to the GPU buffer.
 * Accepts 4 corner points per quad and computes projective weights.
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  bboxes: { minX: number; minY: number; maxX: number; maxY: number }[],
): void {
  const count = Math.min(bboxes.length, MAX_DETECTED_TAGS);
  // 3 vec4f per quad = 12 f32 values
  const buf = new ArrayBuffer(MAX_INSTANCES * 3 * 16);
  const view = new Float32Array(buf);
  for (let i = 0; i < count; i++) {
    const b = bboxes[i];
    const corners: Point[] = [
      { x: b.minX, y: b.minY }, // TL
      { x: b.maxX, y: b.minY }, // TR
      { x: b.minX, y: b.maxY }, // BL
      { x: b.maxX, y: b.maxY }, // BR
    ];
    const [w0, w1, w2, w3] = computeProjectiveWeights(corners);

    // Debug log first 3 quads with NDC coords
    if (i < 3) {
      const sw0 = Number(w0), sw1 = Number(w1), sw2 = Number(w2), sw3 = Number(w3);
      const halfW = pipeline.width * 0.5, halfH = pipeline.height * 0.5;
      const ndcTL = [(corners[0].x * 2 / halfW) - 1, 1 - (corners[0].y * 2 / halfH)];
      const ndcTR = [(corners[1].x * 2 / halfW) - 1, 1 - (corners[1].y * 2 / halfH)];
      const ndcBL = [(corners[2].x * 2 / halfW) - 1, 1 - (corners[2].y * 2 / halfH)];
      const ndcBR = [(corners[3].x * 2 / halfW) - 1, 1 - (corners[3].y * 2 / halfH)];
      console.log(`Quad ${i}: px=[(${corners[0].x.toFixed(0)},${corners[0].y.toFixed(0)}) (${corners[1].x.toFixed(0)},${corners[1].y.toFixed(0)}) (${corners[2].x.toFixed(0)},${corners[2].y.toFixed(0)}) (${corners[3].x.toFixed(0)},${corners[3].y.toFixed(0)})] ndc=[(${ndcTL[0].toFixed(2)},${ndcTL[1].toFixed(2)}) (${ndcTR[0].toFixed(2)},${ndcTR[1].toFixed(2)}) (${ndcBL[0].toFixed(2)},${ndcBL[1].toFixed(2)}) (${ndcBR[0].toFixed(2)},${ndcBR[1].toFixed(2)})]`);
    }

    const base = i * 12;
    view[base + 0] = corners[0].x;  // x0
    view[base + 1] = corners[0].y;  // y0
    view[base + 2] = corners[1].x;  // x1
    view[base + 3] = corners[1].y;  // y1
    view[base + 4] = corners[2].x;  // x2
    view[base + 5] = corners[2].y;  // y2
    view[base + 6] = corners[3].x;  // x3
    view[base + 7] = corners[3].y;  // y3
    view[base + 8] = w0;
    view[base + 9] = w1;
    view[base + 10] = w2;
    view[base + 11] = w3;
  }
  pipeline.quadCornersBuffer.write(buf);
}

export { filterNestedQuads } from './contour';

export async function detectContours(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
): Promise<{ quads: DetectedQuad[], extentData: ExtentRow[], dilatedGradients: Float32Array, labelData: Uint32Array }> {
  // Copy storage buffers → staging buffers (staging has MAP_READ, storage does not)
  const enc = root.device.createCommandEncoder({ label: 'readback' });
  const labelStorage = pipeline.compactLabelBuffer.buffer as GPUBuffer;
  enc.copyBufferToBuffer(labelStorage, 0, pipeline.compactLabelStaging, 0, labelStorage.size);
  const edgeStorage = pipeline.filteredBuffer.buffer as GPUBuffer;
  enc.copyBufferToBuffer(edgeStorage, 0, pipeline.filteredStaging, 0, edgeStorage.size);
  root.device.queue.submit([enc.finish()]);
  await root.device.queue.onSubmittedWorkDone();

  // Read from staging buffers — no TypeGPU wrapper, no per-element toString()
  await pipeline.compactLabelStaging.mapAsync(GPUMapMode.READ);
  const labelData = new Uint32Array(pipeline.compactLabelStaging.getMappedRange());
  pipeline.compactLabelStaging.unmap();

  await pipeline.filteredStaging.mapAsync(GPUMapMode.READ);
  const dilatedGradients = new Float32Array(pipeline.filteredStaging.getMappedRange());
  pipeline.filteredStaging.unmap();

  // CPU-side: extract regions and fit quads
  const regions = extractRegions(labelData, pipeline.width, pipeline.height, dilatedGradients);
  const quads = validateAndFilterQuads(regions, dilatedGradients, pipeline.width);

  // Read extent buffer
  const extentData: ExtentRow[] = await pipeline.extentBuffer.read();

  return { quads, extentData, dilatedGradients, labelData };
}

// (empty space)

export { computeThreshold };
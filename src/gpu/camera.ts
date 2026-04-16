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
import { createGridVizPipeline, createGridVizLayouts, MAX_DETECTED_TAGS } from './pipelines/gridVizPipeline';
import type { DetectedQuad } from './contour';

export type DisplayMode = 'edges' | 'nms' | 'edgesDilated' | 'labels' | 'grayscale' | 'debug' | 'grid';

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
    .createBuffer(d.arrayOf(d.f32, width * height))
    .$usage('storage');

  const dilatedEdgeBuffer = root
    .createBuffer(d.arrayOf(d.f32, width * height))
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
    edgeBuffer: dilatedEdgeBuffer,
    labelBuffer: pointerJumpBuffer0,
  });

  const pointerJumpPingPongBindGroups = [
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: dilatedEdgeBuffer,
      readBuffer: pointerJumpBuffer0,
      writeBuffer: pointerJumpBuffer1,
    }),
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: dilatedEdgeBuffer,
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
      edgeBuffer: dilatedEdgeBuffer,
      labelRead: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: dilatedEdgeBuffer,
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
  const { gridVizLayout } = createGridVizLayouts(root);

  const quadCornersBuffer = root
    .createBuffer(d.arrayOf(d.f32, MAX_DETECTED_TAGS * 8))
    .$usage('storage');

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

  const edgesDilatedBindGroup = root.createBindGroup(edgesLayout, {
    sobelBuffer: sobelBuffer,
    filteredBuffer: dilatedEdgeBuffer,
  });

  const edgeFilterBindGroup = root.createBindGroup(edgeFilterLayout, {
    sobelBuffer: sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer: filteredBuffer,
  });

  const edgeDilateBindGroup = root.createBindGroup(edgeDilateLayout, {
    src: filteredBuffer,
    grad: sobelBuffer,
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
    quadCornersBuffer,
    gridVizPipeline,
    gridVizLayout,
    gridVizBindGroup,
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
  displayMode: DisplayMode = 'edges'
) {
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

    pipeline.edgeDilatePipeline
      .with(computePass)
      .with(pipeline.edgeDilateBindGroup)
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
  if (displayMode === 'edges') {
    pipeline.sobelRenderPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.sobelRenderBindGroup)
      .draw(3);
  } else if (displayMode === 'nms') {
    pipeline.edgesPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.edgesBindGroup)
      .draw(3);
  } else if (displayMode === 'edgesDilated') {
    pipeline.edgesPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.edgesDilatedBindGroup)
      .draw(3);
  } else if (displayMode === 'labels') {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: finalLabelBuffer,
    });
    pipeline.labelVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(labelVizBindGroup)
      .draw(3);
  } else if (displayMode === 'debug') {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: finalLabelBuffer,
    });
    pipeline.labelVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(labelVizBindGroup)
      .draw(3);
  } else if (displayMode === 'grid') {
    // Render grayscale base, then grid overlay on top (additive blend for grid)
    pipeline.grayRenderPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.grayRenderBindGroup)
      .draw(3);
    // Grid: use 'over' blend so transparent pixels don't draw
    pipeline.gridVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.gridVizBindGroup)
      .draw(3);
  } else {
    pipeline.grayRenderPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.grayRenderBindGroup)
      .draw(3);
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

export async function detectContours(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
): Promise<{ quads: DetectedQuad[], extentData: ExtentRow[], edgeData: Uint8Array, labelData: Uint32Array }> {
  // processFrame already ran pointer-jump + compact labeling.
  // We just read the results — no GPU re-run needed.
  const [labelRaw, edgeRaw] = await Promise.all([
    pipeline.compactLabelBuffer.read(),
    pipeline.dilatedEdgeBuffer.read(),
  ]);

  // Convert to flat typed arrays
  const labelData = new Uint32Array(labelRaw.length);
  for (let i = 0; i < labelRaw.length; i++) {
    labelData[i] = labelRaw[i];
  }

  const edgeData = new Uint8Array(edgeRaw.length);
  for (let i = 0; i < edgeRaw.length; i++) {
    edgeData[i] = edgeRaw[i];
  }

  // TODO: read sobelBuffer on demand for corner detection, or refactor corners to use edgeData
  const sobelData = new Float32Array(0);

  // CPU-side: extract regions and fit quads
  const { extractRegions, validateAndFilterQuads } = await import('./contour');
  const regions = extractRegions(labelData, pipeline.width, pipeline.height, sobelData);
  const quads = validateAndFilterQuads(regions, sobelData, pipeline.width);

  // Read extent buffer
  const extentData: ExtentRow[] = await pipeline.extentBuffer.read();

  return { quads, extentData, edgeData, labelData };
}

/**
 * Filter out quads nested inside larger quads.
 * A quad is nested if its centroid falls inside another quad.
 * Returns only the outer quads.
 */
function filterNestedQuads(quads: DetectedQuad[]): DetectedQuad[] {
  const outer: DetectedQuad[] = [];
  for (let i = 0; i < quads.length; i++) {
    const a = quads[i];
    // Compute centroid
    const cx = (a.corners[0].x + a.corners[1].x + a.corners[2].x + a.corners[3].x) / 4;
    const cy = (a.corners[0].y + a.corners[1].y + a.corners[2].y + a.corners[3].y) / 4;
    let contained = false;
    for (let j = 0; j < quads.length; j++) {
      if (i === j) continue;
      const b = quads[j];
      // Barycentric test for point-in-quad
      const tlX = b.corners[0].x, tlY = b.corners[0].y;
      const trX = b.corners[1].x, trY = b.corners[1].y;
      const brX = b.corners[2].x, brY = b.corners[2].y;
      const blX = b.corners[3].x, blY = b.corners[3].y;
      const e0x = trX - tlX, e0y = trY - tlY;
      const e1x = blX - tlX, e1y = blY - tlY;
      const denom = e0x * e1y - e1x * e0y;
      if (Math.abs(denom) < 1e-6) continue;
      const u = ((cx - tlX) * e1y - (cy - tlY) * e1x) / denom;
      const v = ((cx - tlX) * e0y - (cy - tlY) * e0x) / (-denom);
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        contained = true;
        break;
      }
    }
    if (!contained) outer.push(a);
  }
  return outer;
}

/**
 * Update the quad corners buffer with detected quads.
 * Only includes quads with valid corner detection (hasCorners=true).
 * Filters out quads nested inside larger quads.
 * Writes up to MAX_DETECTED_TAGS quads (8 f32 values each: tl, tr, br, bl x,y).
 * Unused slots are filled with sentinel values.
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  quads: DetectedQuad[],
): void {
  // Only include corner-detected quads
  const cornerQuads = quads.filter(q => q.hasCorners);
  // Filter nested
  const outerQuads = filterNestedQuads(cornerQuads);

  console.log(`[updateQuadCorners] total=${quads.length} hasCorners=${cornerQuads.length} outer=${outerQuads.length}`);
  if (outerQuads.length > 0) {
    const q = outerQuads[0];
    console.log(`[updateQuadCorners] first quad: ${JSON.stringify(q.corners)}`);
  }

  const data = new Float32Array(MAX_DETECTED_TAGS * 8);
  let count = 0;
  for (const quad of outerQuads) {
    if (count >= MAX_DETECTED_TAGS) break;
    const offset = count * 8;
    // Order: TL, TR, BR, BL
    data[offset + 0] = quad.corners[0].x;
    data[offset + 1] = quad.corners[0].y;
    data[offset + 2] = quad.corners[1].x;
    data[offset + 3] = quad.corners[1].y;
    data[offset + 4] = quad.corners[2].x;
    data[offset + 5] = quad.corners[2].y;
    data[offset + 6] = quad.corners[3].x;
    data[offset + 7] = quad.corners[3].y;
    count++;
  }
  // Fill remaining slots with sentinel
  for (let i = count; i < MAX_DETECTED_TAGS; i++) {
    const offset = i * 8;
    data[offset + 0] = 0xFFFFFFFF;
    data[offset + 1] = 0xFFFFFFFF;
    data[offset + 2] = 0xFFFFFFFF;
    data[offset + 3] = 0xFFFFFFFF;
    data[offset + 4] = 0xFFFFFFFF;
    data[offset + 5] = 0xFFFFFFFF;
    data[offset + 6] = 0xFFFFFFFF;
    data[offset + 7] = 0xFFFFFFFF;
  }
  pipeline.quadCornersBuffer.write(data);
}

export { computeThreshold };
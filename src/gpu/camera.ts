// Camera pipeline — fully GPU-based, single submit
import { tgpu, d, std } from 'typegpu';
import { sqrt, atomicAdd, atomicStore, atomicLoad } from 'typegpu/std';

import {
  EDGES_VIEW_BINARY_MASK,
  HISTOGRAM_BINS,
  POINTER_JUMP_ITERATIONS,
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
import type { DetectedQuad } from './contour';

export type DisplayMode = 'edges' | 'edgesDilated' | 'labels' | 'grayscale' | 'debug';

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
  const context = root.configureContext({ canvas });
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

  // ═══════════════════════════════════════════════════════════════════════
  // LAYOUTS & PIPELINES
  // ═══════════════════════════════════════════════════════════════════════

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
    EDGES_VIEW_BINARY_MASK,
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
    filteredBuffer: filteredBuffer,
  });

  const edgesDilatedBindGroup = root.createBindGroup(edgesLayout, {
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
  };
}

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

    const wantComponentViz = displayMode === 'labels' || displayMode === 'debug';

    const runPointerJump = () => {
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
    };

    if (wantComponentViz) {
      runPointerJump();
    }

    computePass.end();
  }

  // RENDER: Display mode selection + Histogram
  if (displayMode === 'edges') {
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

export async function detectContours(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
): Promise<{ quads: DetectedQuad[], labelData: Uint32Array }> {
  const enc = root.device.createCommandEncoder({ label: 'contour labels' });
  const computePass = enc.beginComputePass({ label: 'labeling' });
  const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height);

  let finalBuffer: typeof pipeline.pointerJumpBuffer0;

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
  finalBuffer = pj === 0 ? pipeline.pointerJumpBuffer0 : pipeline.pointerJumpBuffer1;

  computePass.end();
  root.device.queue.submit([enc.finish()]);

  const labelData = new Uint32Array(await finalBuffer.read());

  // Read edge buffer for region analysis
  const edgeData = new Float32Array(await pipeline.dilatedEdgeBuffer.read());

  // CPU-side: extract regions and fit quads
  const { extractRegions, validateAndFilterQuads } = await import('./contour');
  const regions = extractRegions(labelData, pipeline.width, pipeline.height, edgeData);
  const quads = validateAndFilterQuads(regions);

  return { quads, labelData };
}

export { computeThreshold };
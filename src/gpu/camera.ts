// Camera pipeline — fully GPU-based, single submit
import { tgpu, d, std } from 'typegpu';
import { sqrt, atomicAdd, atomicStore, atomicLoad } from 'typegpu/std';

import { HISTOGRAM_BINS, computeThreshold } from './pipelines/constants';
import { createLayouts } from './pipelines/layouts';
import { createCopyPipeline } from './pipelines/copyPipeline';
import { createGrayPipeline } from './pipelines/grayPipeline';
import { createSobelPipeline } from './pipelines/sobelPipeline';
import { createHistogramResetPipeline, createHistogramAccumulatePipeline } from './pipelines/histogramPipelines';
import { createEdgesPipeline } from './pipelines/edgesPipeline';
import { createEdgeFilterPipeline } from './pipelines/edgeFilterPipeline';
import { createHistogramRenderPipeline } from './pipelines/histogramRenderPipeline';

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
    .createBuffer(d.arrayOf(d.f32, width * height))
    .$usage('storage');

  const filteredBuffer = root
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
  } = createLayouts(root, histogramSchema);

  const copyPipeline = createCopyPipeline(root, copyLayout);
  const grayPipeline = createGrayPipeline(root, grayTexToBufferLayout, width, height);
  const sobelPipeline = createSobelPipeline(root, sobelLayout, width, height);
  const histogramResetPipeline = createHistogramResetPipeline(root, histogramResetLayout);
  const histogramPipeline = createHistogramAccumulatePipeline(root, histogramLayout, width, height);
  const edgesPipeline = createEdgesPipeline(root, edgesLayout, width, height, presentationFormat);
  const edgeFilterPipeline = createEdgeFilterPipeline(root, edgeFilterLayout, width, height);
  const histogramDisplayPipeline = createHistogramRenderPipeline(root, histogramDisplayLayout, presentationFormat, width * height, 0);

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

  const edgeFilterBindGroup = root.createBindGroup(edgeFilterLayout, {
    sobelBuffer: sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer: filteredBuffer,
  });

  const histogramDisplayBindGroup = root.createBindGroup(histogramDisplayLayout, {
    histogram: histogramBuffer,
    thresholdBin: thresholdBinBuffer,
  });

  return {
    context,
    histContext,
    grayTex,
    grayBuffer,
    sobelBuffer,
    filteredBuffer,
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
    edgeFilterPipeline,
    edgeFilterBindGroup,
    histogramDisplayPipeline,
    histogramDisplayBindGroup,
    sampler,
    width,
    height,
    histWidth: 512,
    histHeight: 120,
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
  threshold: number
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

  // ═══════════════════════════════════════════════════════════════════════
  // RENDER: Copy external → grayTex (MUST happen before compute)
  // ═══════════════════════════════════════════════════════════════════════
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3);

  // ═══════════════════════════════════════════════════════════════════════
  // COMPUTE: Gray → Sobel → Histogram
  // ═══════════════════════════════════════════════════════════════════════
  {
    const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' });

    pipeline.grayPipeline
      .with(computePass)
      .with(pipeline.grayTexToBufferBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    pipeline.sobelPipeline
      .with(computePass)
      .with(pipeline.sobelBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    pipeline.histogramResetPipeline
      .with(computePass)
      .with(pipeline.histogramResetBindGroup)
      .dispatchWorkgroups(HISTOGRAM_BINS);

    pipeline.histogramPipeline
      .with(computePass)
      .with(pipeline.histogramComputeBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    pipeline.edgeFilterPipeline
      .with(computePass)
      .with(pipeline.edgeFilterBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    computePass.end();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // RENDER: Edges + Histogram visualization
  // ═══════════════════════════════════════════════════════════════════════
  pipeline.edgesPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.context })
    .with(pipeline.edgesBindGroup)
    .draw(3);

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS);

  root.device.queue.submit([enc.finish()]);
}

export { computeThreshold };

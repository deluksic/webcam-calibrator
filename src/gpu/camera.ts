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
import { createGrayRenderPipeline } from './pipelines/grayRenderPipeline';
import { createSobelRenderPipeline } from './pipelines/sobelRenderPipeline';
import { createFilteredRenderPipeline } from './pipelines/filteredRenderPipeline';
import { createLabelVizPipeline } from './pipelines/labelVizPipeline';
import { createContourLayouts, createLabelInitPipeline, createJfaPropagatePipeline, detectQuads, type DetectedQuad } from './contour';

export type DisplayMode = 'edges' | 'labels' | 'grayscale' | 'debug';

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

  // JFA label buffers for contour detection (ping-pong)
  const labelBuffer0 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage('storage');

  const labelBuffer1 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage('storage');

  // Debug: buffer to check propagation wrote something
  const jfaDebugBuffer = root
    .createBuffer(d.u32)
    .$usage('storage');

  // JFA layouts and pipelines
  const { labelInitLayout, jfaLayout } = createContourLayouts(root);
  const labelInitPipeline = createLabelInitPipeline(root, labelInitLayout, width, height);
  const jfaResult = createJfaPropagatePipeline(root, jfaLayout, width, height);
  const jfaPropagatePipeline = jfaResult.pipeline;
  const jfaDebugPipeline = jfaResult.debugPipeline;

  // JFA bind groups (include offset uniform)
  const jfaOffsetBuffer = root.createBuffer(d.i32).$usage('uniform');
  const labelInitBindGroup = root.createBindGroup(labelInitLayout, {
    edgeBuffer: filteredBuffer,
    labelBuffer: labelBuffer0,
  });

  const jfaPingPongBindGroups = [
    root.createBindGroup(jfaLayout, {
      readBuffer: labelBuffer0,
      writeBuffer: labelBuffer1,
      offset: jfaOffsetBuffer,
    }),
    root.createBindGroup(jfaLayout, {
      readBuffer: labelBuffer1,
      writeBuffer: labelBuffer0,
      offset: jfaOffsetBuffer,
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

  const edgeFilterBindGroup = root.createBindGroup(edgeFilterLayout, {
    sobelBuffer: sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer: filteredBuffer,
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
    labelBuffer: labelBuffer0,
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
    labelVizPipeline,
    labelVizBindGroup,
    labelBuffer0,
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
    // Sobel render (debug)
    sobelRenderPipeline,
    sobelRenderBindGroup,
    sobelRenderLayout,
    // Filtered render (debug: edge filter output)
    filteredRenderPipeline,
    filteredRenderBindGroup,
    filteredRenderLayout,
    // JFA contour detection
    labelBuffer1,
    labelInitPipeline,
    labelInitBindGroup,
    jfaPropagatePipeline,
    jfaDebugPipeline,
    jfaOffsetBuffer,
    jfaPingPongBindGroups,
    jfaDebugBuffer,
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

  // COMPUTE: Gray → Sobel → Histogram + JFA
  let finalLabelBuffer = pipeline.labelBuffer0;
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

    // JFA: Initialize labels from edge mask
    pipeline.labelInitPipeline
      .with(computePass)
      .with(pipeline.labelInitBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    // JFA propagate passes (skip for debug to test init only)
    const maxRange = Math.floor(Math.max(pipeline.width, pipeline.height) / 2);
    let offset = maxRange;
    let sourceIdx = 0;

    if (displayMode !== 'debug') {
      while (offset >= 1) {
        pipeline.jfaOffsetBuffer.write(offset);
        pipeline.jfaPropagatePipeline
          .with(computePass)
          .with(pipeline.jfaPingPongBindGroups[sourceIdx])
          .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));
        sourceIdx ^= 1;
        offset = Math.floor(offset / 2);
      }
    }

    // Set final buffer based on last sourceIdx
    // In debug mode, skip propagate so sourceIdx=0, debugPipeline writes to labelBuffer1
    finalLabelBuffer = (displayMode === 'debug' || sourceIdx === 1) ? pipeline.labelBuffer0 : pipeline.labelBuffer1;

    // DEBUG: run debug pipeline that counts neighbors
    // Writes to labelBuffer1 (read=0, write=1 in ping-pong index 0)
    pipeline.jfaDebugPipeline
      .with(computePass)
      .with(pipeline.jfaPingPongBindGroups[0])
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    // In debug mode, data is in labelBuffer1 from debug pipeline
    finalLabelBuffer = pipeline.labelBuffer1;

    computePass.end();
  }

  // RENDER: Display mode selection + Histogram
  if (displayMode === 'edges') {
    pipeline.edgesPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(pipeline.edgesBindGroup)
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
    // Debug: show result of jfaDebugPipeline (neighbor count) - data is in labelBuffer1
    const debugBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: pipeline.labelBuffer1,
    });
    pipeline.labelVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context })
      .with(debugBindGroup)
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
  const enc = root.device.createCommandEncoder({ label: 'jfa contour' });
  const computePass = enc.beginComputePass({ label: 'jfa' });

  // Initialize labels from edge mask
  pipeline.labelInitPipeline
    .with(computePass)
    .with(pipeline.labelInitBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // JFA passes with decreasing step sizes
  const maxRange = Math.floor(Math.max(pipeline.width, pipeline.height) / 2);
  let offset = maxRange;
  let sourceIdx = 0;

  while (offset >= 1) {
    pipeline.jfaOffsetBuffer.write(offset);
    pipeline.jfaPropagatePipeline
      .with(computePass)
      .with(pipeline.jfaPingPongBindGroups[sourceIdx])
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));
    sourceIdx ^= 1;
    offset = Math.floor(offset / 2);
  }

  computePass.end();
  root.device.queue.submit([enc.finish()]);

  // Read back labeled buffer
  const finalBuffer = sourceIdx === 0 ? pipeline.labelBuffer0 : pipeline.labelBuffer1;
  const labelData = new Uint32Array(await finalBuffer.read());

  // Read edge buffer for region analysis
  const edgeData = new Float32Array(await pipeline.filteredBuffer.read());

  // CPU-side: extract regions and fit quads
  const { extractRegions, validateAndFilterQuads } = await import('./contour');
  const regions = extractRegions(labelData, pipeline.width, pipeline.height, edgeData);
  const quads = validateAndFilterQuads(regions);

  return { quads, labelData };
}

export { computeThreshold };
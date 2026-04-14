// Camera pipeline — TypeGPU frame capture + grayscale + Sobel + Threshold + Histogram
// Pass 1: copy external camera texture → RGBA intermediate (render pass)
// Pass 2: compute RGBA → rgba8unorm grayscale (compute pass)
// Pass 3: compute grayscale → Sobel gradient magnitude (compute pass)
// Pass 4: compute histogram using 256 parallel workgroups (one per bin)
// Pass 5: read histogram back to CPU and compute threshold
// Pass 6: apply threshold (compute pass)
// Pass 7: render thresholded edges → canvas (render pass)

import { tgpu, d, common, std } from 'typegpu';
import { sqrt, atomicAdd, atomicStore, atomicLoad } from 'typegpu/std';
import type { TgpuTexture, TgpuRenderPipeline, TgpuBindGroupLayout, TgpuBindGroup, TgpuSampler, TgpuBuffer } from 'typegpu';

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;
const HISTOGRAM_BINS = 256;
const PIXELS_PER_THREAD = 32;

// ─── Pipeline factory ────────────────────────────────────────────────────
export function createCameraPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat
) {
  // Configure context on the canvas
  const context = root.configureContext({ canvas });

  // ── Textures ─────────────────────────────────────────────────────────────
  const rgbaTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('sampled', 'storage', 'render');

  const grayTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('storage', 'sampled');

  const sobelTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('storage', 'sampled');

  const edgesTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('storage', 'sampled');

  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  // ── Buffers ─────────────────────────────────────────────────────────────
  // Threshold uniform buffer
  const thresholdBuffer = root.createBuffer(d.f32, 0.0).$usage('uniform');

  // Histogram buffer: 256 atomic uint32 counters
  const histogramSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS);
  const histogramBuffer = root.createBuffer(histogramSchema).$usage('storage');

  // ── Layouts ─────────────────────────────────────────────────────────────
  const copyLayout = tgpu.bindGroupLayout({
    cameraTex: { externalTexture: d.textureExternal() },
    sampler: { sampler: 'filtering' },
  });

  const grayLayout = tgpu.bindGroupLayout({
    rgbaTex: { texture: d.texture2d(d.f32) },
    grayTex: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
  });

  const sobelLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32) },
    sobelTex: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
  });

  const histogramLayout = tgpu.bindGroupLayout({
    sobelTex: { texture: d.texture2d(d.f32) },
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  const thresholdLayout = tgpu.bindGroupLayout({
    sobelTex: { texture: d.texture2d(d.f32) },
    edgesTex: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
    threshold: { uniform: d.f32 },
  });

  const displayLayout = tgpu.bindGroupLayout({
    outputTex: { texture: d.texture2d(d.f32) },
    sampler: { sampler: 'filtering' },
  });

  // ── Pass 1: copy camera → rgba ──────────────────────────────────────────
  const copyFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    return std.textureSampleBaseClampToEdge(copyLayout.$.cameraTex, copyLayout.$.sampler, i.uv);
  });

  const copyPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: copyFrag,
    targets: { format: 'rgba8unorm' },
  });

  // ── Pass 2: rgba → grayscale ────────────────────────────────────────────
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }
    const color = std.textureLoad(grayLayout.$.rgbaTex, input.gid.xy, 0);
    const gray = color.r * WR + color.g * WG + color.b * WB;
    std.textureStore(grayLayout.$.grayTex, input.gid.xy, d.vec4f(gray, gray, gray, 1.0));
  });

  const grayPipeline = root.createComputePipeline({ compute: grayKernel });

  // ── Pass 3: grayscale → Sobel magnitude ─────────────────────────────────
  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const pos = d.vec2i(input.gid.xy);
    const tl = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, -1)), 0).r;
    const t  = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(0, -1)), 0).r;
    const tr = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, -1)), 0).r;
    const ml = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, 0)), 0).r;
    const mr = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, 0)), 0).r;
    const bl = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, 1)), 0).r;
    const b  = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(0, 1)), 0).r;
    const br = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, 1)), 0).r;

    const gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    const gy = (bl + 2.0 * b  + br) - (tl + 2.0 * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    const normalized = magnitude * (1.0 / 512.0);

    std.textureStore(sobelLayout.$.sobelTex, input.gid.xy, d.vec4f(normalized, 0, 0, 1));
  });

  const sobelPipeline = root.createComputePipeline({ compute: sobelKernel });

  // ── Pass 4: histogram reset kernel ─────────────────────────────────────
  // First pass: reset all histogram bins to zero using atomic store
  const histogramResetKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu';
    const binIdx = input.gid.x;
    if (binIdx >= d.u32(HISTOGRAM_BINS)) { return; }
    // Reset to zero using atomic store operation
    atomicStore(histogramLayout.$.histogram[binIdx], d.u32(0));
  });

  const histogramResetPipeline = root.createComputePipeline({ compute: histogramResetKernel });

  // ── Pass 5: histogram with atomic operations ───────────────────────────
  // Many workgroups, each thread processes pixels and atomically increments bins
  // Uses atomicAdd to safely increment from multiple threads
  const histogramKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    const zero = d.u32(0);

    // Get this thread's share of pixels
    const tileWidth = d.u32(16);
    const tileHeight = d.u32(16);
    const numBins = d.u32(HISTOGRAM_BINS);

    // Process 16x16 tile of pixels
    const startX = input.gid.x * tileWidth;
    const startY = input.gid.y * tileHeight;

    for (let dy = zero; dy < tileHeight; dy = dy + d.u32(1)) {
      for (let dx = zero; dx < tileWidth; dx = dx + d.u32(1)) {
        const px = startX + dx;
        const py = startY + dy;

        // Skip out-of-bounds
        if (px >= d.u32(width) || py >= d.u32(height)) { continue; }

        // Load magnitude and compute bin
        const mag = std.textureLoad(histogramLayout.$.sobelTex, d.vec2i(px, py), 0).r;
        const bin = d.u32(mag * d.f32(HISTOGRAM_BINS));
        let clampedBin = bin;
        if (bin >= numBins) { clampedBin = numBins - d.u32(1); }

        // Atomically increment the bin
        atomicAdd(histogramLayout.$.histogram[clampedBin], d.u32(1));
      }
    }
  });

  const histogramPipeline = root.createComputePipeline({ compute: histogramKernel });

  // ── Pass 5: apply threshold ──────────────────────────────────────────────
  const thresholdKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }
    const thresh = thresholdLayout.$.threshold;
    const mag = std.textureLoad(thresholdLayout.$.sobelTex, input.gid.xy, 0).r;
    let edge = 0.0;
    if (mag > thresh) { edge = 1.0; }
    std.textureStore(thresholdLayout.$.edgesTex, input.gid.xy, d.vec4f(edge, edge, edge, 1.0));
  });

  // Debug kernel to read first 10 histogram bins (for verification)
  const histogramDebugKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x > d.u32(9)) { return; }
    const val = atomicLoad(histogramLayout.$.histogram[input.gid.x]);
  });
  const histogramDebugPipeline = root.createComputePipeline({ compute: histogramDebugKernel });

  const thresholdPipeline = root.createComputePipeline({ compute: thresholdKernel });

  // ── Pass 6: display ──────────────────────────────────────────────────────
  const displayFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const c = std.textureSampleBaseClampToEdge(displayLayout.$.outputTex, displayLayout.$.sampler, i.uv);
    return c;
  });

  const displayPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: displayFrag,
    targets: { format: presentationFormat },
  });

  // ── Bind groups ─────────────────────────────────────────────────────────
  const grayBindGroup = root.createBindGroup(grayLayout, {
    rgbaTex: rgbaTex,
    grayTex: grayTex,
  });

  const sobelBindGroup = root.createBindGroup(sobelLayout, {
    grayTex: grayTex,
    sobelTex: sobelTex,
  });

  const histogramResetBindGroup = root.createBindGroup(histogramLayout, {
    sobelTex: sobelTex,
    histogram: histogramBuffer,
  });

  const histogramBindGroup = root.createBindGroup(histogramLayout, {
    sobelTex: sobelTex,
    histogram: histogramBuffer,
  });

  const thresholdBindGroup = root.createBindGroup(thresholdLayout, {
    sobelTex: sobelTex,
    edgesTex: edgesTex,
    threshold: thresholdBuffer,
  });

  // Display bind groups
  const displayBindGroupEdges = root.createBindGroup(displayLayout, {
    outputTex: edgesTex,
    sampler: sampler,
  });

  const displayBindGroupSobel = root.createBindGroup(displayLayout, {
    outputTex: sobelTex,
    sampler: sampler,
  });

  const displayBindGroupGray = root.createBindGroup(displayLayout, {
    outputTex: grayTex,
    sampler: sampler,
  });

  const displayBindGroupOriginal = root.createBindGroup(displayLayout, {
    outputTex: rgbaTex,
    sampler: sampler,
  });

  return {
    context,
    rgbaTex,
    grayTex,
    sobelTex,
    edgesTex,
    histogramBuffer,
    thresholdBuffer,
    copyPipeline,
    grayPipeline,
    grayBindGroup,
    sobelPipeline,
    sobelBindGroup,
    histogramResetPipeline,
    histogramResetBindGroup,
    histogramPipeline,
    histogramBindGroup,
    thresholdPipeline,
    thresholdBindGroup,
    displayPipeline,
    displayBindGroupEdges,
    displayBindGroupSobel,
    displayBindGroupGray,
    displayBindGroupOriginal,
    histogramDebugPipeline,
    copyLayout,
    sampler,
    width,
    height,
  };
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>;

// ─── Display modes ──────────────────────────────────────────────────────
export type DisplayMode = 'edges' | 'sobel' | 'grayscale' | 'original';

// ─── Compute adaptive threshold from histogram ──────────────────────────
export function computeThreshold(histogramData: number[], percentile: number = 0.85): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0);
  const targetCount = totalPixels * (1 - percentile);

  let cumulative = 0;
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i];
    if (cumulative >= targetCount) {
      // Return Sobel threshold value (bin / 512, since Sobel is normalized by 1/512)
      return (i + 1) / 512.0;
    }
  }
  return 0.5; // fallback
}

// ─── Per-frame processing ──────────────────────────────────────────────
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  displayMode: DisplayMode = 'edges'
) {
  // Create external texture bind group per-frame
  const copyBindGroup = root.createBindGroup(pipeline.copyLayout, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  });

  // ── Pass 1: render external → rgba ──────────────────────────────────────
  pipeline.copyPipeline.withColorAttachment({ view: pipeline.rgbaTex.createView() }).with(copyBindGroup).draw(3);

  // ── Pass 2: rgba → grayscale ────────────────────────────────────────────
  pipeline.grayPipeline
    .with(pipeline.grayBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 3: grayscale → Sobel ───────────────────────────────────────────
  pipeline.sobelPipeline
    .with(pipeline.sobelBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 4: reset histogram and apply threshold ───────────────────────
  pipeline.histogramResetPipeline
    .with(pipeline.histogramResetBindGroup)
    .dispatchWorkgroups(HISTOGRAM_BINS);

  pipeline.thresholdPipeline
    .with(pipeline.thresholdBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 5: display ─────────────────────────────────────────────────────
  const displayBindGroup = displayMode === 'edges' ? pipeline.displayBindGroupEdges :
                           displayMode === 'sobel' ? pipeline.displayBindGroupSobel :
                           displayMode === 'grayscale' ? pipeline.displayBindGroupGray :
                           pipeline.displayBindGroupOriginal;
  pipeline.displayPipeline.withColorAttachment({ view: pipeline.context }).with(displayBindGroup).draw(3);
}

// ─── Async version with histogram-based adaptive threshold ─────────────
export async function processFrameAsync(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  displayMode: DisplayMode = 'edges'
) {
  // Create external texture bind group per-frame
  const copyBindGroup = root.createBindGroup(pipeline.copyLayout, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  });

  // ── Pass 1: render external → rgba ──────────────────────────────────────
  pipeline.copyPipeline.withColorAttachment({ view: pipeline.rgbaTex.createView() }).with(copyBindGroup).draw(3);

  // ── Pass 2: rgba → grayscale ────────────────────────────────────────────
  pipeline.grayPipeline
    .with(pipeline.grayBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 3: grayscale → Sobel ───────────────────────────────────────────
  pipeline.sobelPipeline
    .with(pipeline.sobelBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 4: reset histogram and compute new histogram ──────────────────
  // Reset all bins to zero first
  pipeline.histogramResetPipeline
    .with(pipeline.histogramResetBindGroup)
    .dispatchWorkgroups(HISTOGRAM_BINS);

  // Process all pixels with atomic increments
  pipeline.histogramPipeline
    .with(pipeline.histogramBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Read histogram and compute threshold ─────────────────────────────────
  const histogramResult = await pipeline.histogramBuffer.read();
  const threshold = computeThreshold(histogramResult, 0.85);

  // Update threshold uniform
  pipeline.thresholdBuffer.write(threshold);

  // ── Pass 5: apply threshold ─────────────────────────────────────────────
  pipeline.thresholdPipeline
    .with(pipeline.thresholdBindGroup)
    .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

  // ── Pass 6: display ─────────────────────────────────────────────────────
  const displayBindGroup = displayMode === 'edges' ? pipeline.displayBindGroupEdges :
                           displayMode === 'sobel' ? pipeline.displayBindGroupSobel :
                           displayMode === 'grayscale' ? pipeline.displayBindGroupGray :
                           pipeline.displayBindGroupOriginal;
  pipeline.displayPipeline.withColorAttachment({ view: pipeline.context }).with(displayBindGroup).draw(3);
}
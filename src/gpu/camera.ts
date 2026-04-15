// Camera pipeline — fully GPU-based, single submit
// 1. Render: external → grayTex (copy via render pass)
// 2. Compute: grayTex → grayBuffer (f32)
// 3. Compute: grayBuffer → sobelBuffer (f32)
// 4. Compute: sobelBuffer → histogramBuffer (atomic u32[256])
// 5. Render: sobelBuffer → canvas (edges)
// 6. Render: histogramBuffer → histogramCanvas (histogram visualization)

import { tgpu, d, common, std } from 'typegpu';
import { sqrt, atomicAdd, atomicStore, atomicLoad } from 'typegpu/std';

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;
const HISTOGRAM_BINS = 256;
const HIST_WIDTH = 512;
const HIST_HEIGHT = 120;

// ─── Pipeline factory ────────────────────────────────────────────────────
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

  // ── Textures ─────────────────────────────────────────────────────────────
  // Intermediate RGBA texture for external → usable format
  const grayTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('storage', 'sampled', 'render');

  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  // ── Buffers ─────────────────────────────────────────────────────────────
  const grayBuffer = root
    .createBuffer(d.arrayOf(d.f32, width * height))
    .$usage('storage');

  const sobelBuffer = root
    .createBuffer(d.arrayOf(d.f32, width * height))
    .$usage('storage');

  // Histogram buffer: 256 atomic uint32 counters
  const histogramSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS);
  const histogramBuffer = root.createBuffer(histogramSchema).$usage('storage');

  // ── Layouts ─────────────────────────────────────────────────────────────

  // Copy layout (render: external → grayTex)
  const copyLayout = tgpu.bindGroupLayout({
    cameraTex: { externalTexture: d.textureExternal() },
    sampler: { sampler: 'filtering' },
  });

  // Gray tex → buffer layout
  const grayTexToBufferLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32), access: 'readonly' },
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  });

  // Sobel layout (compute: buffer → buffer)
  const sobelLayout = tgpu.bindGroupLayout({
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  });

  // Histogram layout (compute: buffer → atomic buffer)
  const histogramLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  // Histogram reset layout
  const histogramResetLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  // Display layout (edges)
  const edgesLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  });

  // Histogram display layout
  const histogramDisplayLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  // ── Pass 1: copy external texture → grayTex ────────────────────────────
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

  // ── Pass 2: grayTex → grayBuffer ────────────────────────────────────────
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const color = std.textureLoad(grayTexToBufferLayout.$.grayTex, input.gid.xy, 0);
    const gray = color.r * d.f32(WR) + color.g * d.f32(WG) + color.b * d.f32(WB);

    const idx = input.gid.y * d.u32(width) + input.gid.x;
    grayTexToBufferLayout.$.grayBuffer[idx] = gray;
  });

  const grayPipeline = root.createComputePipeline({ compute: grayKernel });

  // ── Pass 3: compute Sobel from grayscale buffer ─────────────────────────
  function sobelLoad(px: number, py: number, w: number) {
    'use gpu';
    if (px >= w || py >= w) { return d.f32(0); }
    return sobelLayout.$.grayBuffer[py * w + px];
  }

  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const x = input.gid.x;
    const y = input.gid.y;
    const w = d.u32(width);

    const tl = sobelLoad(x - d.u32(1), y - d.u32(1), w);
    const t  = sobelLoad(x, y - d.u32(1), w);
    const tr = sobelLoad(x + d.u32(1), y - d.u32(1), w);
    const ml = sobelLoad(x - d.u32(1), y, w);
    const mr = sobelLoad(x + d.u32(1), y, w);
    const bl = sobelLoad(x - d.u32(1), y + d.u32(1), w);
    const b  = sobelLoad(x, y + d.u32(1), w);
    const br = sobelLoad(x + d.u32(1), y + d.u32(1), w);

    const gx = (tr + d.f32(2.0) * mr + br) - (tl + d.f32(2.0) * ml + bl);
    const gy = (bl + d.f32(2.0) * b  + br) - (tl + d.f32(2.0) * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * w + x] = magnitude;
  });

  const sobelPipeline = root.createComputePipeline({ compute: sobelKernel });

  // ── Pass 4: histogram reset ─────────────────────────────────────────────
  const histogramResetKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu';
    const binIdx = input.gid.x;
    if (binIdx >= d.u32(HISTOGRAM_BINS)) { return; }
    atomicStore(histogramResetLayout.$.histogram[binIdx], d.u32(0));
  });

  const histogramResetPipeline = root.createComputePipeline({ compute: histogramResetKernel });

  // ── Pass 5: histogram with atomic operations ───────────────────────────
  const histogramKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    const zero = d.u32(0);
    const tileWidth = d.u32(16);
    const tileHeight = d.u32(16);
    const numBins = d.u32(HISTOGRAM_BINS);

    const startX = input.gid.x * tileWidth;
    const startY = input.gid.y * tileHeight;

    for (let dy = zero; dy < tileHeight; dy = dy + d.u32(1)) {
      for (let dx = zero; dx < tileWidth; dx = dx + d.u32(1)) {
        const px = startX + dx;
        const py = startY + dy;

        if (px >= d.u32(width) || py >= d.u32(height)) { continue; }

        const idx = py * d.u32(width) + px;
        const mag = histogramLayout.$.sobelBuffer[idx];
        const bin = d.u32(mag * d.f32(HISTOGRAM_BINS));
        let clampedBin = bin;
        if (bin >= numBins) { clampedBin = numBins - d.u32(1); }

        atomicAdd(histogramLayout.$.histogram[clampedBin], d.u32(1));
      }
    }
  });

  const histogramPipeline = root.createComputePipeline({ compute: histogramKernel });

  // ── Pass 6: render edges to canvas ──────────────────────────────────────
  const edgesFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    // Compute pixel position from UV
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const mag = edgesLayout.$.sobelBuffer[idx];
    return d.vec4f(mag, mag, mag, d.f32(1));
  });

  const edgesPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: edgesFrag,
    targets: { format: presentationFormat },
  });

  // ── Pass 7: render histogram bars using instanceIndex ───────────────────
  // 256 instances, each bar renders as a vertical rectangle
  // No vertex buffer needed - instance index directly accesses the histogram buffer
  const histogramVert = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      uv: d.vec2f,
      barIndex: d.location(1, d.f32),
      position: d.builtin.position,
    },
  })((i) => {
    'use gpu';
    const vertInBar = i.vertexIndex % d.u32(6);
    const localU = d.f32(vertInBar % d.u32(2));
    const localV = d.f32(vertInBar / d.u32(2));

    const histW = d.f32(HIST_WIDTH);
    const histH = d.f32(HIST_HEIGHT);
    const numBars = d.f32(HISTOGRAM_BINS);
    const barW = histW / numBars;

    const barPxX = (d.f32(i.instanceIndex) * barW) + (localU * barW);
    // Flip vertical: barPxY = 0 at bottom, histH at top
    // localV = 0 → bottom of bar rectangle, localV = 1 → top of bar rectangle
    const barPxY = (d.f32(1) - localV) * histH;
    const clipX = (barPxX / histW) * d.f32(2.0) - d.f32(1.0);
    const clipY = d.f32(1.0) - (barPxY / histH) * d.f32(2.0);
    // UV.y: 0 at bottom (bar bottom), 1 at top (bar top)

    return {
      uv: d.vec2f(localU, localV),
      barIndex: d.f32(i.instanceIndex),
      position: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
    };
  });

  const histogramFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f), barIndex: d.location(1, d.f32) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const bin = d.u32(i.barIndex);
    const count = atomicLoad(histogramDisplayLayout.$.histogram[bin]);

    // Normalize bar height - will be updated via uniform buffer
    const maxCount = d.f32(50000); // Placeholder, actual value set via uniform
    const normalizedHeight = d.f32(count) / maxCount;

    // Clip bars above their height (make them empty/transparent)
    if (i.uv.y > normalizedHeight) {
      return d.vec4f(d.f32(0.1), d.f32(0.1), d.f32(0.15), d.f32(0));
    }

    // Blue bars for histogram
    return d.vec4f(d.f32(0.29), d.f32(0.62), d.f32(1.0), d.f32(1));
  });

  const histogramDisplayPipeline = root.createRenderPipeline({
    vertex: histogramVert,
    fragment: histogramFrag,
    targets: { format: presentationFormat },
  });

  // ── Bind groups ─────────────────────────────────────────────────────────
  // Copy (recreated per-frame for external texture)
  const copyLayoutTemplate = copyLayout;

  // Gray tex → buffer (static)
  const grayTexToBufferBindGroup = root.createBindGroup(grayTexToBufferLayout, {
    grayTex: grayTex,
    grayBuffer: grayBuffer,
  });

  // Sobel (static)
  const sobelBindGroup = root.createBindGroup(sobelLayout, {
    grayBuffer: grayBuffer,
    sobelBuffer: sobelBuffer,
  });

  // Histogram reset (static)
  const histogramResetBindGroup = root.createBindGroup(histogramResetLayout, {
    histogram: histogramBuffer,
  });

  // Histogram compute (static)
  const histogramComputeBindGroup = root.createBindGroup(histogramLayout, {
    sobelBuffer: sobelBuffer,
    histogram: histogramBuffer,
  });

  // Edges display (static)
  const edgesBindGroup = root.createBindGroup(edgesLayout, {
    sobelBuffer: sobelBuffer,
  });

  // Histogram display (static)
  const histogramDisplayBindGroup = root.createBindGroup(histogramDisplayLayout, {
    histogram: histogramBuffer,
  });

  return {
    context,
    histContext,
    grayTex,
    grayBuffer,
    sobelBuffer,
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
    histogramDisplayPipeline,
    histogramDisplayBindGroup,
    sampler,
    width,
    height,
    histWidth: HIST_WIDTH,
    histHeight: HIST_HEIGHT,
  };
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>;

// ─── Compute adaptive threshold from histogram ──────────────────────────
export function computeThreshold(histogramData: number[], percentile: number = 0.85): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0);
  const targetCount = totalPixels * (1 - percentile);

  let cumulative = 0;
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i];
    if (cumulative >= targetCount) {
      return (i + 1) / 512.0;
    }
  }
  return 0.5;
}

// ─── Per-frame processing (single submit) ───────────────────────────────
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement
) {
  // Create external texture bind group per-frame
  const copyBindGroup = root.createBindGroup(pipeline.copyLayoutTemplate, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  });

  // Single command encoder for all passes
  const enc = root.device.createCommandEncoder({ label: 'camera frame' });

  // ── Render pass: Copy external → grayTex (must happen FIRST before compute reads) ──
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3);

  // ── Compute passes ───────────────────────────────────────���──────────────
  {
    const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram' });

    // 1. Gray tex → buffer
    pipeline.grayPipeline
      .with(computePass)
      .with(pipeline.grayTexToBufferBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    // 2. Sobel
    pipeline.sobelPipeline
      .with(computePass)
      .with(pipeline.sobelBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    // 3. Reset histogram
    pipeline.histogramResetPipeline
      .with(computePass)
      .with(pipeline.histogramResetBindGroup)
      .dispatchWorkgroups(HISTOGRAM_BINS);

    // 4. Compute histogram
    pipeline.histogramPipeline
      .with(computePass)
      .with(pipeline.histogramComputeBindGroup)
      .dispatchWorkgroups(Math.ceil(pipeline.width / 16), Math.ceil(pipeline.height / 16));

    computePass.end();
  }

  // ── Render passes ────────────────────────────────────────────────────────

  // 2. Render edges to canvas
  pipeline.edgesPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.context })
    .with(pipeline.edgesBindGroup)
    .draw(3);

  // 3. Render histogram bars (256 instances × 6 vertices)
  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS);

  root.device.queue.submit([enc.finish()]);
}

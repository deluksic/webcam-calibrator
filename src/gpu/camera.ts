// Camera pipeline — TypeGPU frame capture + grayscale + Sobel
// Pass 1: copy external camera texture → RGBA intermediate (render pass)
// Pass 2: compute RGBA → rgba8unorm grayscale (compute pass)
// Pass 3: compute grayscale → Sobel gradient magnitude (compute pass)
// Pass 4: render Sobel magnitude → canvas (render pass)

import { tgpu, d, common, std } from 'typegpu';
import { sqrt, atan2 } from 'typegpu/std';
import type { TgpuTexture, TgpuRenderPipeline, TgpuBindGroupLayout, TgpuBindGroup, TgpuSampler } from 'typegpu';

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;

// Sobel kernels
const SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
const SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

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

  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

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

  // ── Pass 3: grayscale → Sobel magnitude ────────────────────────────────
  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    // Convert gid to vec2i for textureLoad offsets
    const pos = d.vec2i(input.gid.xy);

    // Sobel with clamped coordinates
    const tl = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, -1)), 0).r;
    const t  = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(0, -1)), 0).r;
    const tr = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, -1)), 0).r;
    const ml = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, 0)), 0).r;
    const mr = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, 0)), 0).r;
    const bl = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(-1, 1)), 0).r;
    const b  = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(0, 1)), 0).r;
    const br = std.textureLoad(sobelLayout.$.grayTex, pos.add(d.vec2i(1, 1)), 0).r;

    // Sobel X: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    const gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    // Sobel Y: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    const gy = (bl + 2.0 * b  + br) - (tl + 2.0 * t  + tr);

    // Magnitude
    const magnitude = sqrt(gx * gx + gy * gy);
    // Normalize to 0-1 range
    const normalized = magnitude * (1.0 / 512.0);

    // Store magnitude in R, direction in G (for debugging)
    const angle = atan2(gy, gx);
    std.textureStore(sobelLayout.$.sobelTex, input.gid.xy, d.vec4f(normalized, (angle + 3.14159) / 6.28318, 0, 1));
  });

  const sobelPipeline = root.createComputePipeline({ compute: sobelKernel });

  // ── Pass 4: display ────────────────────────────────────────────────────
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

  // Display bind groups for each mode
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

  const displayBindGroup = root.createBindGroup(displayLayout, {
    outputTex: sobelTex,
    sampler: sampler,
  });

  return {
    context,
    rgbaTex,
    grayTex,
    sobelTex,
    copyPipeline,
    grayPipeline,
    grayBindGroup,
    sobelPipeline,
    sobelBindGroup,
    displayPipeline,
    displayBindGroupSobel,
    displayBindGroupGray,
    displayBindGroupOriginal,
    copyLayout,
    sampler,
    width,
    height,
  };
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>;

// ─── Display modes ──────────────────────────────────────────────────────
export type DisplayMode = 'sobel' | 'grayscale' | 'original';

// ─── Per-frame processing ─────────────────────────────────────���──────────────
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  displayMode: DisplayMode = 'sobel'
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

  // ── Pass 4: display ─────────────────────────────────────────────────────
  const displayBindGroup = displayMode === 'sobel' ? pipeline.displayBindGroupSobel :
                          displayMode === 'grayscale' ? pipeline.displayBindGroupGray :
                          pipeline.displayBindGroupOriginal;
  pipeline.displayPipeline.withColorAttachment({ view: pipeline.context }).with(displayBindGroup).draw(3);
}

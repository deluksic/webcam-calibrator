// Camera pipeline — TypeGPU frame capture + grayscale
// Pass 1: copy external camera texture → RGBA intermediate (render pass)
// Pass 2: compute RGBA → rgba8unorm grayscale (compute pass)
// Pass 3: render grayscale → canvas (render pass)

import { tgpu, d, common, std } from 'typegpu';
import type { TgpuTexture, TgpuRenderPipeline, TgpuBindGroupLayout, TgpuBindGroup, TgpuSampler } from 'typegpu';
import { computeDispatch2d } from './pipelines/constants';

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;

// ─── Pipeline factory ────────────────────────────────────────────────────
export function createGrayscalePipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat
) {
  // Configure context on the canvas
  const context = root.configureContext({ canvas });

  // ── Textures ──────────────────��──────────────────────────────────────────
  const rgbaTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('sampled', 'storage', 'render');

  const grayTex = root
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

  const displayLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32) },
    sampler: { sampler: 'filtering' },
  });

  // ── Copy pipeline (camera → rgba) ──────────────────────────────���───────
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

  // ── Compute pipeline (rgba → grayscale) ────────────────────────────────
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) { return; }
    const color = std.textureLoad(grayLayout.$.rgbaTex, input.gid.xy, 0);
    const gray = color.r * WR + color.g * WG + color.b * WB;
    std.textureStore(grayLayout.$.grayTex, input.gid.xy, d.vec4f(gray, gray, gray, 1.0));
  });

  const grayPipeline = root.createComputePipeline({ compute: grayKernel });

  // ── Display pipeline (grayscale → canvas) ───────────────────────────────
  const displayFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const g = std.textureSampleBaseClampToEdge(displayLayout.$.grayTex, displayLayout.$.sampler, i.uv);
    return g;
  });

  const displayPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: displayFrag,
    targets: { format: presentationFormat },
  });

  // ── Bind groups (static) ────────────────────────────────────────────────
  const grayBindGroup = root.createBindGroup(grayLayout, {
    rgbaTex: rgbaTex,
    grayTex: grayTex,
  });

  const displayBindGroup = root.createBindGroup(displayLayout, {
    grayTex: grayTex,
    sampler: sampler,
  });

  return {
    context,
    rgbaTex,
    grayTex,
    copyPipeline,
    grayPipeline,
    grayBindGroup,
    displayPipeline,
    displayBindGroup,
    copyLayout,
    sampler,
    width,
    height,
  };
}

export type GrayscalePipeline = ReturnType<typeof createGrayscalePipeline>;

// ─── Per-frame processing ────────────────────────────────────────────────────
export function processFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: GrayscalePipeline,
  video: HTMLVideoElement
) {
  // Create external texture bind group per-frame
  const copyBindGroup = root.createBindGroup(pipeline.copyLayout, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  });

  // ── Pass 1: render external → rgba ──────────────────────────────────────
  pipeline.copyPipeline.withColorAttachment({ view: pipeline.rgbaTex.createView() }).with(copyBindGroup).draw(3);

  // ── Pass 2: compute rgba → grayscale ────────────────────────────────────
  const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height);
  pipeline.grayPipeline
    .with(pipeline.grayBindGroup)
    .dispatchWorkgroups(wgX, wgY);

  // ── Pass 3: display grayscale → canvas ──────────────────────────────────
  pipeline.displayPipeline.withColorAttachment({ view: pipeline.context }).with(pipeline.displayBindGroup).draw(3);
}

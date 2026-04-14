// Camera pipeline — TypeGPU frame capture + grayscale
// Pass 1: copy external camera texture → RGBA intermediate (render pass)
// Pass 2: compute RGBA → r8unorm grayscale (compute pass)

import * as tgpu from 'typegpu';
import { d } from 'typegpu';
import { common } from 'typegpu';
import { getRoot } from './init';

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;

let rgbaTex: tgpu.TgpuTexture | null = null;
let grayTex: tgpu.TgpuTexture | null = null;
let texSizeBuffer: tgpu.TgpuBuffer | null = null;

let copyPipeline: tgpu.TgpuRenderPipeline | null = null;
let copyLayout: tgpu.TgpuBindGroupLayout | null = null;
let copyBindGroup: tgpu.TgpuBindGroup | null = null;
let copySampler: tgpu.TgpuSampler | null = null;

let grayPipeline: tgpu.TgpuComputePipeline | null = null;
let grayLayout: tgpu.TgpuBindGroupLayout | null = null;
let grayBindGroup: tgpu.TgpuBindGroup | null = null;

let displayPipeline: tgpu.TgpuRenderPipeline | null = null;
let displayLayout: tgpu.TgpuBindGroupLayout | null = null;
let displayBindGroup: tgpu.TgpuBindGroup | null = null;
let displaySampler: tgpu.TgpuSampler | null = null;

let canvasCtx: GPUCanvasContext | null = null;

let currentWidth = 0;
let currentHeight = 0;

export async function init(width: number, height: number, canvas: HTMLCanvasElement) {
  const root = getRoot();
  const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
  if (!ctx) throw new Error('Canvas not configured for WebGPU');
  ctx.configure({
    device: root.device,
    format: navigator.gpu!.getPreferredCanvasFormat(),
    alphaMode: 'opaque',
  });
  canvasCtx = ctx;
  currentWidth = width;
  currentHeight = height;

  // ── RGBA intermediate texture ──────────────────────────────────────────
  rgbaTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm', dimension: '2d' })
    .$usage('sampled', 'storage', 'render');

  // ── Grayscale output texture ───────────────────────────────────────────
  grayTex = root
    .createTexture({ size: [width, height], format: 'r8unorm', dimension: '2d' })
    .$usage('storage');

  // ── Uniform for tex dimensions (per-frame writes) ───────────────────────
  texSizeBuffer = root.createUniform(d.vec2u);

  // ── Sampler (created once, reused every frame) ─────────────────────────
  copySampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  // ══ Pass 1: copy camera → RGBA ════════════════════════════════════════════

  const copyFrag = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((i) => {
    'use gpu';
    return textureSample(i.cameraTex, i.samp, i.uv);
  }).$uses({ cameraTex: d.textureExternal(), samp: d.sampler() });

  copyPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: copyFrag,
    targets: [{ format: 'rgba8unorm' }],
  });

  copyLayout = tgpu.bindGroupLayout({
    cameraTex: { externalTexture: {} },
    samp: { sampler: {} },
  });

  // Per-frame: bind group with camera external texture (raw GPUExternalTexture)
  copyBindGroup = root.createBindGroup(copyLayout, {
    cameraTex: null as any, // populated per-frame
    samp: copySampler,
  });

  // ══ Pass 2: compute RGBA → grayscale ════════════════════════════════════

  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= input.texSize.x || input.gid.y >= input.texSize.y) { return; }
    const color = textureLoad(input.rgbaTex, input.gid.xy, 0);
    const gray = color.r * WR + color.g * WG + color.b * WB;
    input.grayTex.store(input.gid.xy, gray);
  }).$uses({
    rgbaTex: d.texture2d(d.f32),
    grayTex: d.textureStorage2d('r8unorm', 'write-only'),
    texSize: d.vec2u,
  });

  grayPipeline = root.createComputePipeline(grayKernel);

  // ── Grayscale bind group (reused every frame — texture views are stable) ──
  grayLayout = tgpu.bindGroupLayout({
    rgbaTex: { texture: d.f32 },
    grayTex: { storageTexture: 'r8unorm' },
    texSize: { uniform: d.vec2u },
  });

  const rgbaView = rgbaTex.createView();
  const grayView = grayTex.createView();

  grayBindGroup = root.createBindGroup(grayLayout, {
    rgbaTex: rgbaView,
    grayTex: grayView,
    texSize: texSizeBuffer,
  });

  // ══ Display pass: render grayscale to canvas texture ═══════════════════════
  displaySampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  const displayFrag = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((i) => {
    'use gpu';
    const g = textureSample(i.grayTex, i.samp, i.uv);
    return vec4f(g.r, g.r, g.r, 1.0);
  }).$uses({
    grayTex: d.texture2d(d.f32),
    samp: d.sampler(),
  });

  displayPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: displayFrag,
    targets: [{ format: 'bgra8unorm' }],
  });

  displayLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.f32 },
    samp: { sampler: {} },
  });

  displayBindGroup = root.createBindGroup(displayLayout, {
    grayTex: grayView,
    samp: displaySampler,
  });
}

// ─── Per-frame processing ────────────────────────────────────────────────────
export function processFrame(video: HTMLVideoElement) {
  const root = getRoot();

  if (!copyPipeline || !grayPipeline || !grayBindGroup || !copyBindGroup ||
      !displayPipeline || !displayBindGroup) {
    throw new Error('Pipeline not initialized');
  }

  // Update texSize uniform
  texSizeBuffer!.write(d.vec2u(currentWidth, currentHeight));

  // Import current video frame as external texture
  const extTex = root.device.importExternalTexture({ source: video });

  // Update external texture binding
  (copyBindGroup as any)._resources.cameraTex = extTex;

  const encoder = root.device.createCommandEncoder();

  const rgbaView = rgbaTex!.createView('render');

  // ── Pass 1: render external → rgba ──────────────────────────────────────
  copyPipeline
    .with(encoder)
    .withColorAttachment({ view: rgbaView, loadOp: 'clear', storeOp: 'store' })
    .with(copyBindGroup)
    .draw(3);

  // ── Pass 2: compute rgba → grayscale ────────────────────────────────────
  const computePass = encoder.beginComputePass();
  grayPipeline!
    .with(computePass)
    .with(grayBindGroup)
    .dispatchWorkgroups(Math.ceil(currentWidth / 16), Math.ceil(currentHeight / 16));
  computePass.end();

  // ── Pass 3: display grayscale → canvas ───────────────────────────────────
  const canvasTexture = canvasCtx!.getCurrentTexture();
  const canvasView = canvasTexture.createView();
  displayPipeline
    .with(encoder)
    .withColorAttachment({ view: canvasView, loadOp: 'clear', storeOp: 'store' })
    .with(displayBindGroup)
    .draw(3);

  // Submit
  root.device.queue.submit([encoder.finish()]);
}

export function getGrayscaleTexture() { return grayTex; }
export function getRgbaTexture() { return rgbaTex; }
export function getFrameSize() { return { width: currentWidth, height: currentHeight }; }

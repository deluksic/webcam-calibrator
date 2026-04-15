// Contour detection: JFA connected components + quad fitting on CPU
import { tgpu, d, std } from 'typegpu';

export const COMPONENT_LABEL_INVALID = 0xFFFFFFFF;

export interface DetectedQuad {
  corners: [number, number][];  // 4 corner points
  label: number;
  count: number;
  aspectRatio: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Layouts
// ─────────────────────────────────────────────────────────────────────────────

export function createContourLayouts(root: Awaited<ReturnType<typeof tgpu.init>>) {
  // Initialize labels from edge buffer
  const labelInitLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  });

  // JFA ping-pong layout
  const jfaLayout = tgpu.bindGroupLayout({
    readBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    writeBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  });

  return { labelInitLayout, jfaLayout };
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline 1: Initialize labels from edge mask
// Edge pixels get their pixel index as unique label (seed)
// Non-edge pixels get INVALID
// ─────────────────────────────────────────────────────────────────────────────

export function createLabelInitPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelInitLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const initKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const idx = input.gid.y * d.u32(width) + input.gid.x;
    const edgeMag = labelInitLayout.$.edgeBuffer[idx];

    // Edge pixel: use a common label (1) for all edge pixels
    // JFA will propagate this label to connected neighbors
    // Non-edge: INVALID marker
    if (edgeMag > d.f32(0)) {
      labelInitLayout.$.labelBuffer[idx] = d.u32(1);
    } else {
      labelInitLayout.$.labelBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID);
    }
  });

  return root.createComputePipeline({ compute: initKernel });
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline 2: JFA propagation pass
// Propagate labels to neighbors, keep lowest label (nearest seed approximation)
// ─────────────────────────────────────────────────────────────────────────────

export function createJfaPropagatePipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  jfaLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const offsetUniform = root.createUniform(d.i32);

  const propagateKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const x = d.i32(input.gid.x);
    const y = d.i32(input.gid.y);
    const w = d.i32(width);
    const h = d.i32(height);
    const offset = offsetUniform.$;
    const wU32 = d.u32(w);

    // Current pixel's label
    let bestLabel = jfaLayout.$.readBuffer[d.u32(y) * wU32 + d.u32(x)];

    // Check 8 neighbors (inlined to avoid nested function)
    const offsets = [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0],           [1, 0],
      [-1, 1],  [0, 1],  [1, 1],
    ];

    for (let i = 0; i < 8; i++) {
      const ox = offsets[i][0];
      const oy = offsets[i][1];
      const nx = x + ox * offset;
      const ny = y + oy * offset;
      // Clamp to bounds
      let sx = nx;
      if (nx < d.i32(0)) { sx = d.i32(0); }
      if (nx >= w) { sx = w - d.i32(1); }
      let sy = ny;
      if (ny < d.i32(0)) { sy = d.i32(0); }
      if (ny >= h) { sy = h - d.i32(1); }
      const nIdx = d.u32(sy) * wU32 + d.u32(sx);
      const nLabel = jfaLayout.$.readBuffer[nIdx];
      if (nLabel !== d.u32(COMPONENT_LABEL_INVALID)) {
        if (bestLabel === d.u32(COMPONENT_LABEL_INVALID)) {
          bestLabel = nLabel;
        }
      }
    }

    jfaLayout.$.writeBuffer[d.u32(y) * wU32 + d.u32(x)] = bestLabel;
  });

  const pipeline = root.createComputePipeline({ compute: propagateKernel });
  return { pipeline, offsetUniform };
}

// ─────────────────────────────────────────────────────────────────────────────
// JFA Runner: Execute all passes, return final label buffer
// ─────────────────────────────────────────────────────────────────────────────

export async function runJfa(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelInitPipeline: ReturnType<typeof createLabelInitPipeline>,
  jfaPropagate: ReturnType<typeof createJfaPropagatePipeline>,
  initBindGroup: any,
  pingPongBindGroups: any[],
  labelBuffer: any,
  width: number,
  height: number,
) {
  const enc = root.device.createCommandEncoder({ label: 'jfa' });
  const computePass = enc.beginComputePass({ label: 'jfa compute' });

  // Initialize labels from edge mask
  labelInitPipeline
    .with(computePass)
    .with(initBindGroup)
    .dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));

  // JFA passes with decreasing step sizes
  const maxRange = Math.floor(Math.max(width, height) / 2);
  let offset = maxRange;
  let sourceIdx = 0;

  while (offset >= 1) {
    jfaPropagate.offsetUniform.write(offset);
    jfaPropagate.pipeline
      .with(computePass)
      .with(pingPongBindGroups[sourceIdx])
      .dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    sourceIdx ^= 1;  // Swap source/dest
    offset = Math.floor(offset / 2);
  }

  computePass.end();

  // Copy final labels to output buffer
  // Note: After JFA, labels are in writeBuffer of last iteration
  // We need to read from the correct buffer
  const finalBuffer = pingPongBindGroups[sourceIdx].$.writeBuffer;

  root.device.queue.submit([enc.finish()]);

  return finalBuffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side processing: Extract quads from labeled image
// ─────────────────────────────────────────────────────────────────────────────

export interface RegionData {
  label: number;
  count: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  pixels: [number, number][];
}

export function extractRegions(
  labelData: Uint32Array,
  width: number,
  height: number,
  edgeData: Float32Array,
): Map<number, RegionData> {
  const regions = new Map<number, RegionData>();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const label = labelData[idx];

      if (label === COMPONENT_LABEL_INVALID) continue;

      if (!regions.has(label)) {
        regions.set(label, {
          label,
          count: 0,
          minX: x,
          minY: y,
          maxX: x,
          maxY: y,
          pixels: [],
        });
      }

      const region = regions.get(label)!;
      region.count++;
      region.minX = Math.min(region.minX, x);
      region.minY = Math.min(region.minY, y);
      region.maxX = Math.max(region.maxX, x);
      region.maxY = Math.max(region.maxY, y);

      // Store pixel position for corner detection
      if (region.pixels.length < 500) {
        region.pixels.push([x, y]);
      }
    }
  }

  return regions;
}

export function fitQuadToRegion(region: RegionData): [number, number][] | null {
  const { minX, minY, maxX, maxY } = region;

  // Simple approach: Use corner pixels from the edge set
  // Find the 4 extreme points that could be corners

  const boundingBoxWidth = maxX - minX;
  const boundingBoxHeight = maxY - minY;

  // Aspect ratio check - AprilTags are roughly square
  const aspectRatio = boundingBoxWidth / boundingBoxHeight;
  if (aspectRatio < 0.5 || aspectRatio > 2.0) {
    return null;
  }

  // For each side of the bounding box, find the edge pixel closest to that edge
  // This approximates where the tag border is
  let leftEdge = maxX;
  let rightEdge = minX;
  let topEdge = maxY;
  let bottomEdge = minY;

  for (const [px, py] of region.pixels) {
    // Left edge (min x)
    if (px - minX < leftEdge - minX) leftEdge = px;
    // Right edge (max x)
    if (maxX - px < maxX - rightEdge) rightEdge = px;
    // Top edge (min y)
    if (py - minY < topEdge - minY) topEdge = py;
    // Bottom edge (max y)
    if (maxY - py < maxY - bottomEdge) bottomEdge = py;
  }

  // Return the estimated corner positions
  return [
    [leftEdge, topEdge],
    [rightEdge, topEdge],
    [rightEdge, bottomEdge],
    [leftEdge, bottomEdge],
  ];
}

export function validateAndFilterQuads(
  regions: Map<number, RegionData>,
  minArea: number = 400,  // Min ~20x20 pixels
  maxArea: number = 200000, // Max ~450x450 pixels
): DetectedQuad[] {
  const quads: DetectedQuad[] = [];

  for (const [, region] of regions) {
    const width = region.maxX - region.minX;
    const height = region.maxY - region.minY;
    const area = width * height;

    // Size filter
    if (area < minArea || area > maxArea) continue;

    // Aspect ratio check (should be ~1:1 for AprilTag)
    const aspectRatio = width / height;
    if (aspectRatio < 0.6 || aspectRatio > 1.7) continue;

    // Edge density check - should have consistent edge pixels
    const perimeter = 2 * (width + height);
    const edgeDensity = region.count / perimeter;
    if (edgeDensity < 0.5 || edgeDensity > 5) continue;

    // Fit quad
    const corners = fitQuadToRegion(region);
    if (!corners) continue;

    quads.push({
      corners,
      label: region.label,
      count: region.count,
      aspectRatio,
    });
  }

  return quads;
}

export async function detectQuads(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgeBuffer: any,
  labelBuffer0: any,
  labelBuffer1: any,
  labelInitPipeline: any,
  jfaPropagate: any,
  initBindGroup: any,
  pingPongBindGroups: any[],
  width: number,
  height: number,
): Promise<{ quads: DetectedQuad[], labelData: Uint32Array }> {
  // Run JFA
  const finalBuffer = await runJfa(
    root,
    labelInitPipeline,
    jfaPropagate,
    initBindGroup,
    pingPongBindGroups,
    labelBuffer0,
    width,
    height,
  );

  // Read back labeled image
  const labelData = new Uint32Array(await finalBuffer.read());

  // Also read edge buffer for region analysis
  const edgeData = new Float32Array(await edgeBuffer.read());

  // Extract regions and fit quads
  const regions = extractRegions(labelData, width, height, edgeData);
  const quads = validateAndFilterQuads(regions);

  return { quads, labelData };
}
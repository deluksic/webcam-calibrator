// Grid visualization pipeline: draw 6x6 grid lines over detected AprilTags
import { tgpu, d, std } from 'typegpu';
import { common } from 'typegpu';
import { fract, abs, clamp, min } from 'typegpu/std';

export const GRID_DIVISIONS = 6;
export const GRID_LINE_WIDTH = 0.06; // fraction of cell size
export const MAX_DETECTED_TAGS = 64;

export function createGridVizLayouts(
  root: Awaited<ReturnType<typeof tgpu.init>>,
) {
  const gridVizLayout = tgpu.bindGroupLayout({
    quadCorners: { storage: d.arrayOf(d.f32), access: 'readonly' },
  });

  return { gridVizLayout };
}

export function createGridVizPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  gridVizLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const wi = d.i32(width);
    const hi = d.i32(height);
    const maxPx = d.f32(wi - d.i32(1));
    const maxPy = d.f32(hi - d.i32(1));

    // Pixel coordinates (as float for computation)
    const px = clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx);
    const py = clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy);

    // Compute grid distance across all quads
    // Use min() to combine results (lower = closer to grid line = draw it)
    // Start with "infinity" (very large = no grid line)
    const sentinel = d.f32(0xFFFFFFFF);
    let minGridDist = d.f32(1000); // large sentinel

    // Iterate through quads
    for (const qi of std.range(MAX_DETECTED_TAGS)) {
      const base = d.i32(qi) * d.i32(8);

      const tlX = gridVizLayout.$.quadCorners[base + 0];
      const tlY = gridVizLayout.$.quadCorners[base + 1];
      const trX = gridVizLayout.$.quadCorners[base + 2];
      const trY = gridVizLayout.$.quadCorners[base + 3];
      const brX = gridVizLayout.$.quadCorners[base + 4];
      const brY = gridVizLayout.$.quadCorners[base + 5];
      const blX = gridVizLayout.$.quadCorners[base + 6];
      const blY = gridVizLayout.$.quadCorners[base + 7];

      // Skip if corners not set (check if TL is sentinel)
      if (abs(tlX - sentinel) < d.f32(0.5)) { continue; }
      if (abs(tlY - sentinel) < d.f32(0.5)) { continue; }

      // Edge vectors
      const e0x = trX - tlX; // top edge
      const e0y = trY - tlY;
      const e1x = blX - tlX; // left edge
      const e1y = blY - tlY;

      // Compute u,v using cross products (barycentric-like for quads)
      const denom = e0x * e1y - e1x * e0y;
      const u = ((px - tlX) * e1y - (py - tlY) * e1x) / denom;
      const v = ((px - tlX) * e0y - (py - tlY) * e0x) / denom;

      // Check if inside quad (with epsilon for edge cases)
      const eps = d.f32(0.001);
      const inside = (u >= -eps) && (u <= 1.0 + eps) && (v >= -eps) && (v <= 1.0 + eps);

      if (inside) {
        // Map position to 6x6 grid
        const gridX = u * d.f32(GRID_DIVISIONS);
        const gridY = v * d.f32(GRID_DIVISIONS);

        // Distance to nearest grid line (using fract)
        // For grid lines at integer positions: fract(x) near 0 or near 1
        const fx = fract(gridX);
        const distX = min(fx, d.f32(1) - fx);
        const fy = fract(gridY);
        const distY = min(fy, d.f32(1) - fy);

        // Combine: pixel is on grid if close to ANY line
        const gridDist = min(distX, distY);

        // Keep minimum grid distance across all quads
        minGridDist = min(minGridDist, gridDist);
      }
    }

    // Draw grid line if we're close enough to a grid line
    const halfWidth = d.f32(GRID_LINE_WIDTH / 2);
    const isGridLine = minGridDist < halfWidth;

    if (isGridLine) {
      return d.vec4f(d.f32(0), d.f32(1), d.f32(1), d.f32(1)); // cyan
    }

    // Not in any quad - return transparent
    return d.vec4f(d.f32(0), d.f32(0), d.f32(0), d.f32(0));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: gridVizFrag,
    targets: {
      format: presentationFormat,
      blend: {
        color: {
          operation: 'add',
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
        },
        alpha: {
          operation: 'add',
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
        },
      },
    },
  });
}

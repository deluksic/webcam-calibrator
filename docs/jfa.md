# Jump Flood Algorithm (JFA) for Connected Component Labeling

## Overview

JFA is a GPU-efficient algorithm for computing distance transforms and connected component labeling. It converges in O(log n) passes where n is the maximum distance, making it much faster than naive BFS for large images.

**Note**: TypeGPU doesn't have a built-in JFA implementation. We'll implement it using TypeGPU's compute shader capabilities.

## Algorithm

JFA works by iteratively propagating nearest seed information across the image using progressively smaller step sizes:

```
Pass 1: step = n/2  — Check neighbors at distance n/2
Pass 2: step = n/4  — Check neighbors at distance n/4
Pass 3: step = n/8  — Check neighbors at distance n/8
...
Pass k: step = 1    — Final pass with step = 1
```

For a W×H image, we need ceil(log2(max(W, H))) passes. For 1280×720, that's 11 passes.

### Seed Initialization

Each pixel is initialized with a seed ID:
- Edge pixels (filteredEdges > threshold): unique seed ID (e.g., pixel index)
- Non-edge pixels: INVALID_MARKER (e.g., 0xFFFFFFFF)

### Propagation Rules

For each pixel, check all 8 neighbors within the current step distance:
1. If any neighbor has a valid seed AND is closer than current best → adopt that seed
2. If no closer neighbor found, keep current seed

### 8-Way Neighborhood Offsets

```wgsl
const DIRECTIONS: vec2i[] = [
  (-step, -step), (0, -step), (+step, -step),
  (-step, 0),                 (+step, 0),
  (-step, +step), (0, +step), (+step, +step),
];
```

### Final Output

After all passes, each pixel contains the seed ID of its connected component. Pixels with INVALID_MARKER are non-edge (background).

## TypeGPU Implementation

### Key TypeGPU API Patterns (from jump-flood-distance example)

1. **Guarded Compute Pipeline**: Use `root.createGuardedComputePipeline((x, y) => { 'use gpu'; ... })` for GPU kernels
2. **Dispatch**: Use `dispatchThreads(width, height)` not `dispatchWorkgroups`
3. **Fixed Iteration**: Use `tgpu.unroll([-1, 0, 1])` for neighbor loops (compile-time unrolling)
4. **Texture Operations**:
   - `std.textureDimensions(layout.$.textureName)` → vec2i
   - `std.textureLoad(layout.$.textureName, position)` → loads texel
   - `std.textureStore(layout.$.textureName, position, value)` → writes texel
5. **Safe Bounds**: Use `std.clamp(pos, min, max)` for boundary-safe sampling
6. **Conditional Select**: Use `std.select(condition, trueVal, falseVal)` instead of ternary

### Ping-Pong Texture Setup

```ts
// Two textures for ping-pong: alternate read/write
const textures = [0, 1].map(() =>
  root
    .createTexture({
      size: [width, height],
      format: 'rgba16float',
    })
    .$usage('storage'),
);

// Layouts for ping-pong
const pingPongLayout = tgpu.bindGroupLayout({ readView: d.textureStorage2d, writeView: d.textureStorage2d });

const pingPongBindGroups = [0, 1].map((i) =>
  root.createBindGroup(pingPongLayout, {
    readView: textures[i],
    writeView: textures[1 - i],
  }),
);
```

### Pipeline Structure

```
src/gpu/pipelines/
├── jfaInitPipeline.ts      — Initialize seeds from edges
├── jfaPassPipeline.ts     — Single JFA iteration pass
└── jfaPipeline.ts         — Orchestrates N passes
```

### jfaInitPipeline

Input: filteredBuffer (f32)
Output: ping-pong texture with seed coordinates

```ts
const initFromMaskLayout = tgpu.bindGroupLayout({
  maskTexture: d.textureStorage2d<'r32uint', 'read-only'>,
  writeView: d.textureStorage2d<'rgba16float', 'write-only'>,
});

const initFromMask = root.createGuardedComputePipeline((x, y) => {
  'use gpu';
  const size = std.textureDimensions(initFromMaskLayout.$.writeView);
  const pos = d.vec2f(x, y);
  const uv = pos.div(d.vec2f(size));

  const mask = std.textureLoad(initFromMaskLayout.$.maskTexture, d.vec2i(x, y)).x;

  const inside = mask > 0;
  const invalid = d.vec2f(-1);

  // Store UV coordinates: xy = inside seed, zw = outside seed (-1 = invalid)
  const insideCoord = std.select(invalid, uv, inside);
  const outsideCoord = std.select(uv, invalid, inside);

  std.textureStore(
    initFromMaskLayout.$.writeView,
    d.vec2i(x, y),
    d.vec4f(insideCoord, outsideCoord),
  );
});
```

### jfaPassPipeline

Input: ping-pong texture (rgba16float) from previous pass
Output: ping-pong texture with updated seeds

Key patterns:
- Step size controlled via uniform (written before each pass)
- 8-way neighborhood with `tgpu.unroll([-1, 0, 1])` for fixed-size loop
- Bounds-safe sampling with `std.clamp`
- Track both "inside" and "outside" nearest seeds (for distance field)

```ts
// Offset uniform controls step size for this pass
const offsetUniform = root.createUniform(d.i32);

const sampleWithOffset = (tex, pos, offset) => {
  'use gpu';
  const dims = std.textureDimensions(tex);
  const samplePos = pos.add(offset);

  // Bounds check
  const outOfBounds =
    samplePos.x < 0 ||
    samplePos.y < 0 ||
    samplePos.x >= d.i32(dims.x) ||
    samplePos.y >= d.i32(dims.y);

  // Safe sampling with clamp
  const safePos = std.clamp(samplePos, d.vec2i(0), d.vec2i(dims.sub(1)));
  const loaded = std.textureLoad(tex, safePos);

  return {
    inside: loaded.xy,
    outside: loaded.zw,
    outOfBounds,
  };
};

const jumpFlood = root.createGuardedComputePipeline((x, y) => {
  'use gpu';
  const offset = offsetUniform.$;  // Step size from uniform
  const size = std.textureDimensions(pingPongLayout.$.readView);
  const pos = d.vec2f(x, y);

  let bestInsideCoord = d.vec2f(-1);
  let bestOutsideCoord = d.vec2f(-1);
  let bestInsideDist = 1e20;
  let bestOutsideDist = 1e20;

  // 8-way neighborhood with fixed iteration
  for (const dx of tgpu.unroll([-1, 0, 1])) {
    for (const dy of tgpu.unroll([-1, 0, 1])) {
      const sample = sampleWithOffset(
        pingPongLayout.$.readView,
        d.vec2i(x, y),
        d.vec2i(dx * offset, dy * offset),
      );

      if (sample.inside.x >= 0) {
        const dInside = std.distance(pos, sample.inside.mul(d.vec2f(size)));
        if (dInside < bestInsideDist) {
          bestInsideDist = dInside;
          bestInsideCoord = d.vec2f(sample.inside);
        }
      }

      if (sample.outside.x >= 0) {
        const dOutside = std.distance(pos, sample.outside.mul(d.vec2f(size)));
        if (dOutside < bestOutsideDist) {
          bestOutsideDist = dOutside;
          bestOutsideCoord = d.vec2f(sample.outside);
        }
      }
    }
  }

  std.textureStore(
    pingPongLayout.$.writeView,
    d.vec2i(x, y),
    d.vec4f(bestInsideCoord, bestOutsideCoord),
  );
});

// JFA execution loop
const maxRange = Math.floor(Math.max(width, height) / 2);
let offset = maxRange;

while (offset >= 1) {
  offsetUniform.write(offset);  // Update step size
  jumpFlood.with(pingPongBindGroups[sourceIdx]).dispatchThreads(width, height);
  swap();
  offset = Math.floor(offset / 2);
}
```

### jfaPipeline (Orchestrator)

The JFA loop runs in the compute pass, alternating between ping-pong textures:

```ts
export function runJfa(
  root,
  jfaInitPipeline,
  jfaInitBindGroup,
  jumpFloodPipeline,
  pingPongBindGroups,
  offsetUniform,
  width,
  height,
) {
  let sourceIdx = 0;

  // Initialize seeds from edge mask
  jfaInitPipeline.with(jfaInitBindGroup).dispatchThreads(width, height);

  sourceIdx = 0;

  // JFA passes with decreasing step sizes
  const maxRange = Math.floor(Math.max(width, height) / 2);
  let offset = maxRange;

  while (offset >= 1) {
    offsetUniform.write(offset);
    jumpFloodPipeline
      .with(pingPongBindGroups[sourceIdx])
      .dispatchThreads(width, height);
    sourceIdx ^= 1;  // Swap source/dest
    offset = Math.floor(offset / 2);
  }

  return sourceIdx;  // Index of final result texture
}
```

### processFrame Integration

```ts
function processFrame(root, pipeline, video, threshold) {
  const enc = root.device.createCommandEncoder({ label: 'camera frame' });

  // ... existing gray + sobel + histogram + filter ...

  {
    const computePass = enc.beginComputePass({ label: 'jfa' });

    // Initialize seeds from edge mask
    jfaInitPipeline.with(jfaInitBindGroup).dispatchThreads(width, height);

    // JFA passes
    let sourceIdx = 0;
    const maxRange = Math.floor(Math.max(width, height) / 2);
    let offset = maxRange;

    while (offset >= 1) {
      offsetUniform.write(offset);
      jumpFloodPipeline
        .with(pingPongBindGroups[sourceIdx])
        .dispatchThreads(width, height);
      sourceIdx ^= 1;
      offset = Math.floor(offset / 2);
    }

    computePass.end();
  }

  // ... rest of processing ...
}
```

## Application to AprilTag Detection

After JFA completes:
1. Identify connected components by grouping pixels with same seed ID
2. For each component, find bounding box
3. Filter to quadrilaterals (4 corners)
4. Fit quad to edge chains
5. Decode tag pattern

## Complexity

- **Passes**: ceil(log2(max(W, H))) ≈ 11 for 1280×720
- **Per pass**: O(W×H) with 8 neighbor checks
- **Total**: O(W×H × log(max(W, H)))

Compared to BFS: O(W×H × component_size) per component

## References

- [JFA Paper](https://www.cs.drexel.edu/~p解car/papers/SPA_1995.pdf): "Jump Flooding: An Ecient Algorithm for Gradient Ray Casting and Other Propagation Problems on Graphics Hardware"
- Generalization to Connected Components: Same principle, seeds propagate through connected regions

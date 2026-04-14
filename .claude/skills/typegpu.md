# TypeGPU 0.11 Patterns

## Import

```typescript
import { tgpu, d, common, std } from 'typegpu';
```

## Named Exports

Use named `tgpu` export, not default:
```typescript
import { tgpu } from 'typegpu';  // ✓
import tgpu from 'typegpu';       // ✗
```

## Vite Plugin

Required for shader transformation. Add BEFORE `vite-plugin-solid`:

```typescript
import typegpuPlugin from 'unplugin-typegpu/vite';

export default {
  plugins: [typegpuPlugin(), solidPlugin()],
};
```

## Fragment Shader

```typescript
const frag = tgpu.fragmentFn({
  in: { uv: d.location(0, d.vec2f) },
  out: d.vec4f,
})((i) => {
  'use gpu';
  return std.textureSampleBaseClampToEdge(layout.$.texture, layout.$.sampler, i.uv);
});
```

## Compute Shader

```typescript
const kernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [16, 16, 1],
})((input) => {
  'use gpu';
  // input.gid is the global invocation ID
});
```

## Create Pipeline

```typescript
const pipeline = root.createRenderPipeline({
  vertex: common.fullScreenTriangle,
  fragment: frag,
  targets: { format: 'rgba8unorm' },  // NOT array
});
```

## Compute Pipeline

```typescript
const pipeline = root.createComputePipeline({ compute: kernel });
```

## Bind Group Layout

```typescript
const layout = tgpu.bindGroupLayout({
  texture: { texture: d.texture2d(d.f32) },
  storageTex: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
  uniform: { uniform: d.vec2u },
  sampler: { sampler: 'filtering' },  // 'filtering' | 'non-filtering' | 'comparison'
  external: { externalTexture: d.textureExternal() },
});
```

## Create Bind Group

```typescript
const bindGroup = root.createBindGroup(layout, {
  texture: textureOrView,
  sampler: sampler,
});
```

## Texture Usage

```typescript
const tex = root.createTexture({ size, format, dimension })
  .$usage('sampled', 'storage', 'render');  // add needed usages
```

## Canvas Context

Configure BEFORE drawing:

```typescript
const context = root.configureContext({ canvas });

// Later in render:
pipeline.withColorAttachment({ view: context }).with(bindGroup).draw(3);
```

## Dispatch Compute

```typescript
pipeline.with(bindGroup).dispatchWorkgroups(x, y);
```

## Accessing Layout Resources in Shaders

Use `layout.$` accessor:
```typescript
std.textureLoad(layout.$.texture, coords, 0);
std.textureStore(layout.$.storageTex, coords, value);
```

## Key Differences from v0.10

1. Named export `{ tgpu }` instead of `import * as tgpu`
2. `targets: { format }` not `targets: [{ format }]`
3. Compute: `{ compute: kernel }` descriptor pattern
4. `root.configureContext({ canvas })` for canvas rendering
5. Shader intrinsics injected at transform time (skipLibCheck handles types)

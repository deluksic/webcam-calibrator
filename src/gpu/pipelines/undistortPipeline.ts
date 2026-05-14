import type { ColorAttachment, TgpuRoot, TgpuTextureView } from 'typegpu'
import { d, tgpu, std, common } from 'typegpu'

import { PinholeIntrinsicsGpu, RationalDistortion8Gpu } from '@/gpu/schemas/cameraGpuUniforms'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'

const UndistortUniformStruct = d.struct({
  intrinsics: PinholeIntrinsicsGpu,
  distortion: RationalDistortion8Gpu,
  imageSize: d.vec2f,
  /** Uniform block size padding. */
  _pad: d.vec2f,
})

const undistortUniformLayout = tgpu.bindGroupLayout({
  params: { uniform: UndistortUniformStruct },
})

const sourceLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d() },
  srcSampler: { sampler: 'filtering' },
})

export function allocUndistortUniform(root: TgpuRoot) {
  return root.createBuffer(UndistortUniformStruct).$usage('uniform')
}

export type UndistortUniformGpuBuffer = ReturnType<typeof allocUndistortUniform>

export function createUndistortPipeline(
  root: TgpuRoot,
  sourceTextureView: TgpuTextureView,
  presentationFormat: GPUTextureFormat,
  uniform: UndistortUniformGpuBuffer,
) {
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  })

  const paramsBg = root.createBindGroup(undistortUniformLayout, { params: uniform })
  const sourceBg = root.createBindGroup(sourceLayout, {
    // Grayscale ingest view is a sampled 2D texture; TypeGPU's union is wider than bind layout inference.
    source: sourceTextureView as never,
    srcSampler: sampler,
  })

  const frag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })(({ pos }) => {
    'use gpu'
    const p = undistortUniformLayout.$.params
    const intr = p.intrinsics
    const dist = p.distortion

    const uOut = pos.x
    const vOut = pos.y

    const xn = (uOut - intr.cx) / intr.fx
    const yn = (vOut - intr.cy) / intr.fy

    const r2 = xn * xn + yn * yn
    const r4 = r2 * r2
    const r6 = r4 * r2

    const radialNum = d.f32(1) + dist.k1 * r2 + dist.k2 * r4 + dist.k3 * r6
    const radialDen = d.f32(1) + dist.k4 * r2 + dist.k5 * r4 + dist.k6 * r6

    const xd = (xn * radialNum) / radialDen + d.f32(2) * dist.p1 * xn * yn + dist.p2 * (r2 + d.f32(2) * xn * xn)
    const yd = (yn * radialNum) / radialDen + dist.p1 * (r2 + d.f32(2) * yn * yn) + d.f32(2) * dist.p2 * xn * yn

    const uSrc = xd * intr.fx + intr.cx
    const vSrc = yd * intr.fy + intr.cy
    const uv = d.vec2f(uSrc / p.imageSize.x, vSrc / p.imageSize.y)
    return std.textureSample(sourceLayout.$.source, sourceLayout.$.srcSampler, uv)
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })

  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(paramsBg).with(sourceBg).draw(3)
  }

  return { encodeToCanvas, uniform }
}

export function writeUndistortUniform(
  buf: UndistortUniformGpuBuffer,
  params: {
    K: CameraIntrinsics
    distortion: RationalDistortion8
    width: number
    height: number
  },
): void {
  const [k1, k2, p1, p2, k3, k4, k5, k6] = params.distortion
  buf.write({
    intrinsics: {
      fx: params.K.fx,
      fy: params.K.fy,
      cx: params.K.cx,
      cy: params.K.cy,
    },
    distortion: {
      k1,
      k2,
      p1,
      p2,
      k3,
      k4,
      k5,
      k6,
    },
    imageSize: d.vec2f(params.width, params.height),
    _pad: d.vec2f(0, 0),
  })
}

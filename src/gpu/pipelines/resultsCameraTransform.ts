import type { TgpuBindGroup, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import type { Mat4Arg, Vec3Arg } from 'wgpu-matrix'

import { buildOrbitClipMatrix } from '@/lib/orbitOrthoMath'

const ResultsCameraUniformStruct = d.struct({
  clipHomogeneousMatrixFromBoard: d.mat4x4f,
  viewportHalfSizePixels: d.vec2f,
})

export const resultsCameraBindLayout = tgpu
  .bindGroupLayout({
    transform: { uniform: ResultsCameraUniformStruct },
  })
  .$idx(0)

export type ResultsCameraBindGroup = TgpuBindGroup<typeof resultsCameraBindLayout.entries>

export function allocResultsCameraUniform(root: TgpuRoot) {
  return root.createBuffer(ResultsCameraUniformStruct).$usage('uniform')
}

export type ResultsCameraUniformGpuBuffer = ReturnType<typeof allocResultsCameraUniform>

const clipFromBoardScratch: Mat4Arg = Array.from({ length: 16 }, () => 0)

function mat4x4fFromColMajor(m: Mat4Arg) {
  return d.mat4x4f(
    m[0]!,
    m[1]!,
    m[2]!,
    m[3]!,
    m[4]!,
    m[5]!,
    m[6]!,
    m[7]!,
    m[8]!,
    m[9]!,
    m[10]!,
    m[11]!,
    m[12]!,
    m[13]!,
    m[14]!,
    m[15]!,
  )
}

/** Writes shared orbit projection once per frame (bind group 0 for all results passes). */
export function writeResultsCameraTransform(input: {
  aspectWidthOverHeight: number
  orbitEyeDirUnit: Vec3Arg
  baseOrthoExtentY: number
  orthoZoom: number
  viewportWidthPx: number
  viewportHeightPx: number
  cameraUniform: ResultsCameraUniformGpuBuffer
}): void {
  const orthoExtentY = input.baseOrthoExtentY / input.orthoZoom
  buildOrbitClipMatrix(
    { x: 0, y: 0, z: 0 },
    input.aspectWidthOverHeight,
    input.orbitEyeDirUnit,
    orthoExtentY,
    clipFromBoardScratch,
  )

  const clipMatrixCpu = mat4x4fFromColMajor(clipFromBoardScratch)
  const vw = Math.max(1, input.viewportWidthPx)
  const vh = Math.max(1, input.viewportHeightPx)
  input.cameraUniform.write({
    clipHomogeneousMatrixFromBoard: clipMatrixCpu,
    viewportHalfSizePixels: d.vec2f(vw * 0.5, vh * 0.5),
  })
}

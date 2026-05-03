import { circle, circleVertexCount } from '@typegpu/geometry'
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { length, mix, mul, select, smoothstep } from 'typegpu/std'

import { resultsCameraBindLayout, type ResultsCameraBindGroup } from '@/gpu/pipelines/resultsCameraTransform'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { cornersToVec3fArray } from '@/gpu/pipelines/resultsSceneCpu'
import type { CalibrationOk } from '@/workers/calibration.worker'

/** Triangulation subdiv for `circle()`; see `circleVertexCount`. */
export const MARKER_DISK_SUBDIV = 2

/** Radius in framebuffer pixels (~CSS pixel at 1:1 backing store); disks are billboarded so size stays fixed on screen. */
export const RESULTS_MARKER_DISK_RADIUS_PX = 6

export const MAX_RESULTS_MARKER_POINTS = 8192

/** One calibration target in board space (storage buffer element). */
export const MarkerCenter = d.struct({
  positionBoardUnits: d.vec3f,
})

export type MarkerCenterRow = d.Infer<typeof MarkerCenter>

const MarkerCentersSchema = d.arrayOf(MarkerCenter, MAX_RESULTS_MARKER_POINTS)

const MarkerPassUniformStruct = d.struct({
  markerDiskRadiusPixels: d.f32,
  displayedMarkerCount: d.u32,
  orbitMarkerUniformReserve: d.vec2f,
})

export const markersBindLayout = tgpu
  .bindGroupLayout({
    markers: { uniform: MarkerPassUniformStruct },
    centers: {
      storage: MarkerCentersSchema,
      access: 'readonly',
    },
  })
  .$idx(1)

export type MarkersGpuBindValues = ExtractBindGroupInputFromLayout<typeof markersBindLayout.entries>

/**
 * When true: marker pass draws one bright clip-space triangle and ignores projection/circle math.
 * Confirms MSAA color + resolve + bind group wiring. Set to `false` after you see the triangle.
 */
export const RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE = false as const

/** NDC depth offset (× w) so marker disks win over coplanar tag quads; WebGPU nearer = smaller z. */
const RESULTS_MARKER_DEPTH_BIAS_NDC = 0.0005 as const

function allocMarkerPassUniform(root: TgpuRoot) {
  return root.createBuffer(MarkerPassUniformStruct).$usage('uniform')
}

function allocMarkersCenters(root: TgpuRoot) {
  return root.createBuffer(MarkerCentersSchema).$usage('storage')
}

export type MarkerPassUniformGpuBuffer = ReturnType<typeof allocMarkerPassUniform>
export type MarkersCentersBuffer = ReturnType<typeof allocMarkersCenters>

export function writeMarkerPassUniform(buf: MarkerPassUniformGpuBuffer, pointCount: number): void {
  buf.write({
    markerDiskRadiusPixels: d.f32(RESULTS_MARKER_DISK_RADIUS_PX),
    displayedMarkerCount: d.u32(Math.max(0, Math.floor(pointCount))),
    orbitMarkerUniformReserve: d.vec2f(0, 0),
  })
}

export function markerCenterWritesForGpu(ok: CalibrationOk): MarkerCenterRow[] {
  const rows: MarkerCenterRow[] = []
  outer: for (const t of ok.updatedTargets) {
    const cs = cornersToVec3fArray(t.corners)
    for (const c of cs) {
      if (rows.length >= MAX_RESULTS_MARKER_POINTS) {
        break outer
      }
      rows.push(MarkerCenter({ positionBoardUnits: c }))
    }
  }
  const dead = MarkerCenter({
    positionBoardUnits: d.vec3f(0, 0, -1e9),
  })
  for (let i = rows.length; i < MAX_RESULTS_MARKER_POINTS; i++) {
    rows.push(dead)
  }
  return rows
}

function markerInstancesForEncode(requestedMarkers: number) {
  if (RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE) {
    return 1 as const
  }
  return Math.max(requestedMarkers, 0)
}

/** Pipeline + pass bind group + encode in one closure (see {@link createGridVizStage}). */
export function createMarkerResultsStage(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const markerUniform = allocMarkerPassUniform(root)
  const centersBuf = allocMarkersCenters(root)
  const markersBg = root.createBindGroup(markersBindLayout, { markers: markerUniform, centers: centersBuf })

  if (RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE) {
    const vertDiag = tgpu.vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: { clipPos: d.builtin.position },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      if (instanceIndex > d.u32(0)) {
        return { clipPos: d.vec4f(0, 0, 2, 1) }
      }
      const vx = select(d.f32(-0.92), d.f32(0.92), vertexIndex === d.u32(2))
      const vy = select(d.f32(-0.92), d.f32(0.92), vertexIndex === d.u32(1))
      return { clipPos: d.vec4f(vx, vy, d.f32(0.5), d.f32(1)) }
    })

    const fragDiag = tgpu.fragmentFn({
      out: d.vec4f,
    })(() => {
      'use gpu'
      return d.vec4f(1, 0, 0, 1)
    })

    const pipeline = root.createRenderPipeline({
      vertex: vertDiag,
      fragment: fragDiag,
      targets: {
        format: presentationFormat,
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })

    const encodeToPass = (pass: GPURenderPassEncoder, _cameraBg: ResultsCameraBindGroup, _markerInstances: number) => {
      pipeline.with(pass).draw(3, 1)
    }
    return { markerUniform, centersBuf, encodeToPass }
  }

  const diskVerts = circleVertexCount(MARKER_DISK_SUBDIV)
  const vert = tgpu
    .vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: {
        clipPos: d.builtin.position,
        /** Unit disk offset (circle perimeter verts); `length` after interpolation ≈ 0 at center, 1 at rim — same as axis `uv.y` for outline. */
        diskUnit: d.vec2f,
      },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      const cam = resultsCameraBindLayout.$.transform
      const m = markersBindLayout.$.markers
      const row = markersBindLayout.$.centers[instanceIndex]!
      const pc = row.positionBoardUnits
      const centerClip = mul(cam.clipHomogeneousMatrixFromBoard, d.vec4f(pc.x, -pc.y, -pc.z, d.f32(1)))
      const unit = circle(vertexIndex)
      const half = cam.viewportHalfSizePixels
      const rPx = m.markerDiskRadiusPixels
      const dNdcX = (unit.x * rPx) / half.x
      const dNdcY = (unit.y * rPx) / half.y
      const w = centerClip.w
      return {
        clipPos: d.vec4f(
          centerClip.x + dNdcX * w,
          centerClip.y + dNdcY * w,
          centerClip.z - d.f32(RESULTS_MARKER_DEPTH_BIAS_NDC) * w,
          centerClip.w,
        ),
        diskUnit: unit,
      }
    })
    .$uses({ camera: resultsCameraBindLayout, markers: markersBindLayout })

  const frag = tgpu.fragmentFn({
    in: { diskUnit: d.vec2f },
    out: d.vec4f,
  })(({ diskUnit }) => {
    'use gpu'
    let rgb = d.vec3f(0.15, 0.92, 1.0)
    const iy = length(diskUnit)
    const outlineMix = smoothstep(0.7, 0.8, iy)
    const outlineRgb = d.vec3f(0.05, 0.05, 0.07)
    rgb = mix(rgb, outlineRgb, outlineMix)
    return d.vec4f(rgb, 1)
  })

  const pipeline = root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: {
      format: presentationFormat,
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  const encodeToPass = (pass: GPURenderPassEncoder, cameraBg: ResultsCameraBindGroup, markerInstances: number) => {
    const count = markerInstancesForEncode(markerInstances)
    pipeline.with(pass).with(cameraBg).with(markersBg).draw(diskVerts, count)
  }
  return { markerUniform, centersBuf, encodeToPass }
}

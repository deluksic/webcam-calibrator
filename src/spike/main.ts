/** runntime self-check: load the ROI net on runntime core, compare it against a
 *  PyTorch reference, and time it. Results go to #out as JSON so a headless
 *  Chrome run can read them. */
import { tensor, toArray } from '@runntime/core'

import { initSharedGPU } from '@/gpu/init'
import { loadRoiNet } from '@/gpu/roi/roiRunntime'

const { performance } = globalThis

const H = 192
const W = 256
const ITERATIONS = 30

const out = document.querySelector('#out') as HTMLPreElement

const report: Record<string, unknown> = {}
const log = (message: string): void => {
  out.textContent = `${JSON.stringify(report, null, 2)}\n\n${message}`
}

const fail = (error: unknown): void => {
  report.ok = false
  report.error = error instanceof Error ? `${error.message}\n${error.stack}` : String(error)
  log('FAILED')
  ;(window as unknown as { spikeDone: boolean }).spikeDone = true
}

const base = import.meta.env.BASE_URL

const fetchF32 = async (name: string): Promise<Float32Array> => {
  const response = await fetch(`${base}spike/${name}`)
  if (!response.ok) {
    throw new Error(`${name}: HTTP ${response.status}`)
  }
  return new Float32Array(await response.arrayBuffer())
}

const compare = (got: Float32Array, reference: Float32Array): Record<string, unknown> => {
  let maxAbs = 0
  let sumAbs = 0
  let intersection = 0
  let union = 0
  for (let i = 0; i < reference.length; i++) {
    const d = Math.abs(got[i]! - reference[i]!)
    maxAbs = Math.max(maxAbs, d)
    sumAbs += d
    const p = got[i]! > 0.5
    const r = reference[i]! > 0.5
    if (p && r) {
      intersection++
    }
    if (p || r) {
      union++
    }
  }
  return {
    maxAbs,
    meanAbs: sumAbs / reference.length,
    iouAt05: union ? intersection / union : 1,
    max: Math.max(...got),
  }
}

const main = async (): Promise<void> => {
  log('init')
  // Deliberately goes through the app's own init, so this page also verifies
  // that the shared device and ruNNtime's default root are wired up.
  const { root, device } = await initSharedGPU()
  report.adapter = {
    f16: device.features.has('shader-f16'),
    timestampQuery: device.features.has('timestamp-query'),
  }

  const input = await fetchF32('input.f32.bin')
  const reference = await fetchF32('ref_mask.f32.bin')
  if (input.length !== H * W) {
    throw new Error(`input ${input.length} != ${H * W}`)
  }

  log('loading weights')
  const net = await loadRoiNet(root, `${base}models/roi_net.safetensors`)
  report.runntime = { loadMs: Math.round(net.loadMs), tensors: net.tensorCount }

  const x = tensor(root, input, { elems: H * W, dtype: 'f32', dims: [1, H, W] })
  const compileStart = performance.now()
  await toArray(net.evalMask(x))
  ;(report.runntime as Record<string, unknown>).compileMs = Math.round(
    performance.now() - compileStart,
  )
  ;(report.runntime as Record<string, unknown>).compare = compare(
    await toArray(net.evalMask(x)),
    reference,
  )

  const start = performance.now()
  for (let i = 0; i < ITERATIONS; i++) {
    net.evalMask(x)
  }
  await device.queue.onSubmittedWorkDone()
  ;(report.runntime as Record<string, unknown>).msPerRun = Number(
    ((performance.now() - start) / ITERATIONS).toFixed(3),
  )

  // Does a new input height recompile? The app changes H when the camera
  // resolution changes.
  const smallH = 112
  const x2 = tensor(root, input.slice(0, smallH * W), {
    elems: smallH * W,
    dtype: 'f32',
    dims: [1, smallH, W],
  })
  const shapeStart = performance.now()
  await toArray(net.evalMask(x2))
  const firstAtNewShape = performance.now() - shapeStart
  const steadyStart = performance.now()
  for (let i = 0; i < 10; i++) {
    net.evalMask(x2)
  }
  await device.queue.onSubmittedWorkDone()
  report.heightChange = {
    firstEvalMs: Math.round(firstAtNewShape),
    steadyMsPerRun: Number(((performance.now() - steadyStart) / 10).toFixed(3)),
  }

  report.ok = true
  log('done')
  ;(window as unknown as { spikeDone: boolean }).spikeDone = true
}

main().catch(fail)

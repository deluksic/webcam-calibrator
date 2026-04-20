// TypeGPU initialization — singleton GPU root
import { tgpu } from 'typegpu'

const { navigator } = globalThis

let rootPromise: Promise<Awaited<ReturnType<typeof tgpu.init>>> | null = null

export async function initGPU(): Promise<Awaited<ReturnType<typeof tgpu.init>>> {
  if (rootPromise) return rootPromise

  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is not supported. Please use Chrome 113+, Edge 113+, ' + 'or Firefox Nightly with WebGPU enabled.',
    )
  }

  rootPromise = tgpu.init()
  return rootPromise
}

export function getRoot() {
  throw new Error('getRoot is deprecated — use the root from initGPU instead')
}

// TypeGPU initialization — singleton GPU root
import type { TgpuRoot } from 'typegpu'
import { tgpu } from 'typegpu'

const { navigator } = globalThis

let rootPromise: Promise<TgpuRoot> | undefined = undefined

export async function initGPU(): Promise<TgpuRoot> {
  if (rootPromise) {
    return rootPromise
  }

  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported.')
  }

  rootPromise = tgpu.init()
  return rootPromise
}

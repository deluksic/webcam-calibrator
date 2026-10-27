// TypeGPU initialization — one shared device/root for every WebGPU page
import { initRunntime } from '@runntime/core'
import type { TgpuRoot } from 'typegpu'
import { tgpu } from 'typegpu'

const { navigator } = globalThis

/**
 * Root used by Target and Results.
 *
 * Every page shares one device: a cached root that disagreed with the device the
 * capture pages were using could outlive it, and navigating between those pages
 * could leave a canvas configured against a stale device. One root also matches
 * TypeGPU's intended app-wide usage.
 */
export async function initGPU(): Promise<TgpuRoot> {
  const shared = await initSharedGPU()
  return shared.root
}

export type SharedGpu = {
  root: TgpuRoot
  device: GPUDevice
  adapter: GPUAdapter
}

let sharedGpuPromise: Promise<SharedGpu> | undefined

/**
 * Creates the app-wide WebGPU device.
 *
 * `shader-f16` is required: the ROI model runs its convolutions in f16.
 * Adapters without it — Adreno 5xx and most 6xx, NVIDIA Pascal — cannot run the
 * capture pipeline at all.
 */
export function initSharedGPU(): Promise<SharedGpu> {
  if (!sharedGpuPromise) {
    sharedGpuPromise = createSharedGPU().catch((cause: unknown) => {
      sharedGpuPromise = undefined
      throw cause
    })
  }
  return sharedGpuPromise
}

async function createSharedGPU(): Promise<SharedGpu> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported.')
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  })
  if (!adapter) {
    throw new Error('No WebGPU adapter available.')
  }
  if (!adapter.features.has('shader-f16')) {
    throw new Error(
      'This GPU does not support shader-f16, which the ROI model requires. ' +
        'Try a different browser or device.',
    )
  }
  const requiredFeatures: GPUFeatureName[] = ['shader-f16']
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query')
  }
  // The model's kernels bind more than the default eight storage buffers per
  // stage. Request the adapter's actual capacity when it is higher.
  const requiredLimits: Record<string, number> = {}
  if (adapter.limits.maxStorageBuffersPerShaderStage > 8) {
    requiredLimits.maxStorageBuffersPerShaderStage = Math.min(
      adapter.limits.maxStorageBuffersPerShaderStage,
      16,
    )
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits })

  // If the device is lost, drop the cached root so a later pipeline attempt can
  // create a fresh device instead of reusing a dead one forever.
  void device.lost.then(() => {
    sharedGpuPromise = undefined
  })

  const root = tgpu.initFromDevice({ device })
  // ruNNtime's graph API resolves buffers and executions through a default root,
  // so it has to be registered once here. Every page that runs the ROI model
  // depends on this; without it, evalValues fails with "no default device".
  initRunntime(root)
  return { root, device, adapter }
}

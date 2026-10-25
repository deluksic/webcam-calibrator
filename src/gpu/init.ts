// TypeGPU initialization — one shared device/root for every WebGPU page
import type { TgpuRoot } from 'typegpu'
import { tgpu } from 'typegpu'

const { navigator } = globalThis

/**
 * Root used by Target and Results.
 *
 * This deliberately reuses the ONNX/capture device instead of calling
 * `tgpu.init()` a second time. Two independent devices meant a cached root
 * could outlive or disagree with the device the capture pages were using, and
 * navigating between those pages could leave a canvas configured against a
 * stale device. One root also matches TypeGPU's intended app-wide usage.
 */
export async function initGPU(): Promise<TgpuRoot> {
  const shared = await initSharedGPUForOnnx()
  return shared.root
}

export type SharedOnnxGpu = {
  root: TgpuRoot
  device: GPUDevice
  adapter: GPUAdapter
}

let sharedOnnxGpuPromise: Promise<SharedOnnxGpu> | undefined
let originalGpuRequestAdapter: GPU['requestAdapter'] | undefined
let originalAdapterRequestDevice: GPUAdapter['requestDevice'] | undefined

/**
 * Creates one WebGPU device and shares it with ONNX Runtime Web's WebGPU EP.
 *
 * ORT creates its own device unless `navigator.gpu.requestAdapter` and
 * `adapter.requestDevice` return an existing one. This is the same workaround
 * used by the TypeGPU background-segmentation example:
 * https://github.com/software-mansion/TypeGPU/tree/main/apps/typegpu-docs/src/examples/image-processing/background-segmentation
 */
export function initSharedGPUForOnnx(): Promise<SharedOnnxGpu> {
  if (!sharedOnnxGpuPromise) {
    sharedOnnxGpuPromise = createSharedGPUForOnnx().catch((cause: unknown) => {
      sharedOnnxGpuPromise = undefined
      throw cause
    })
  }
  return sharedOnnxGpuPromise
}

async function createSharedGPUForOnnx(): Promise<SharedOnnxGpu> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported.')
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  })
  if (!adapter) {
    throw new Error('No WebGPU adapter available.')
  }
  const requiredFeatures: GPUFeatureName[] = []
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query')
  }
  // Some WebGPU workloads (including ORT's WebGPU EP on some adapters) bind
  // more than the default eight storage buffers per stage. Request the
  // adapter's actual capacity when it is higher.
  const requiredLimits: Record<string, number> = {}
  if (adapter.limits.maxStorageBuffersPerShaderStage > 8) {
    requiredLimits.maxStorageBuffersPerShaderStage = Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16)
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits })
  const gpu = navigator.gpu
  originalGpuRequestAdapter ??= gpu.requestAdapter.bind(gpu)
  originalAdapterRequestDevice ??= adapter.requestDevice.bind(adapter)
  // ORT WebGPU EP does not yet accept an existing device. Returning the same
  // adapter/device makes its GPU tensors share our queue and buffers.
  const gpuWithOverride = gpu as unknown as {
    requestAdapter: (options?: GPURequestAdapterOptions) => Promise<GPUAdapter | null>
  }
  const adapterWithOverride = adapter as unknown as {
    requestDevice: (descriptor?: GPUDeviceDescriptor) => Promise<GPUDevice>
  }
  gpuWithOverride.requestAdapter = async () => adapter
  adapterWithOverride.requestDevice = async () => device

  // If the device is lost, drop the cached root and restore the factories so a
  // later pipeline attempt can create a fresh device instead of reusing a dead
  // one forever.
  void device.lost.then(() => {
    sharedOnnxGpuPromise = undefined
    if (originalGpuRequestAdapter) {
      gpuWithOverride.requestAdapter = originalGpuRequestAdapter
    }
    if (originalAdapterRequestDevice) {
      adapterWithOverride.requestDevice = originalAdapterRequestDevice
    }
  })

  const root = tgpu.initFromDevice({ device })
  // Keep the overrides for the lifetime of the page. ORT creates its session
  // asynchronously, and restoring the factories early lets a later session
  // create a second device. This is the same lifetime as the TypeGPU example's
  // module-level initialization.
  return { root, device, adapter }
}

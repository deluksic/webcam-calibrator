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

  rootPromise = tgpu.init({
    device: {
      optionalFeatures: ['timestamp-query'],
    },
  })
  return rootPromise
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
  const device = await adapter.requestDevice()
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

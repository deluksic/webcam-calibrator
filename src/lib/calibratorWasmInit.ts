import type { InitCalibratorOptions } from '@deluksic/opencv-calibration-wasm'
import { CALIBRATOR_WASM_MAXIMUM_MEMORY_BYTES } from '@deluksic/opencv-calibration-wasm'

/**
 * `maximumMemoryBytes` is the compile-time cap from the wasm build
 * (`CALIBRATOR_WASM_MAXIMUM_MEMORY_BYTES` in `@deluksic/opencv-calibration-wasm`, e.g. 512 MiB in current local `dist/index.js`).
 */
export function calibratorInitOptions(wasmPath: string): InitCalibratorOptions {
  return {
    wasmPath,
    maximumMemoryBytes: CALIBRATOR_WASM_MAXIMUM_MEMORY_BYTES,
  }
}

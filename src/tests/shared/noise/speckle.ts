import { decodeStressAddSpeckle01 } from "../../../lib/decodeStressHarness";

/** Uniform ±amplitude on `[0,1]` intensity; same stream as decode stress for identical seed. */
export function applySpeckle01(intensity: Float32Array, amplitude: number, seed: number): void {
  decodeStressAddSpeckle01(intensity, amplitude, seed);
}

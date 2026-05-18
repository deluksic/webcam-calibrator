import type { TgpuRoot } from 'typegpu'

import type { DetectedQuad } from '@/gpu/detectedQuad'
import { readGpuDetection } from '@/gpu/gpuQuadReadback'
import type { FrameSlot } from '@/gpu/frameSlotPool'

import type { CameraPipeline } from './cameraPipeline'

/** Read GPU tag-detection results from a slot whose compute was already submitted. */
export async function detectForSlot(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): Promise<{
  quads: DetectedQuad[]
  frameId: number
  slot: FrameSlot
}> {
  const quads = await readGpuDetection(root, pipeline)
  return { quads, frameId: slot.frameId, slot }
}

import type { TgpuRoot } from 'typegpu'

import type { DetectedQuad } from '@/gpu/detectedQuad'
import { readGpuDetection } from '@/gpu/gpuQuadReadback'
import type { FrameSlot } from '@/gpu/frameSlotPool'

import type { CameraPipeline } from './cameraPipeline'

/** Read GPU tag-detection results after compute+decode submit. */
export async function detectForSlot(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): Promise<{
  quads: DetectedQuad[]
  quadCount: number
  frameId: number
  slot: FrameSlot
}> {
  const { quads, quadCount } = await readGpuDetection(root, pipeline)
  return { quads, quadCount, frameId: slot.frameId, slot }
}

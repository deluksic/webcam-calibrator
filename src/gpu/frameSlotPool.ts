import type { TgpuRoot } from 'typegpu'
import { d } from 'typegpu'

import type { CameraPipeline } from '@/gpu/cameraPipeline'
import { grayRenderLayout } from '@/gpu/pipelines/grayRenderPipeline'
import type { GrayRenderBindResources } from '@/gpu/pipelines/grayRenderPipeline'

export type FrameSlotState = 'free' | 'inflight' | 'display'

// Inferred types from the factory — avoid spelling out deep TypeGPU generics.
type GraySnapshot = ReturnType<TgpuRoot['createBuffer']>
type GrayRenderBindGroup = ReturnType<TgpuRoot['createBindGroup']>

export interface FrameSlot {
  graySnapshot: GraySnapshot
  labelStaging: GPUBuffer
  filteredStaging: GPUBuffer
  grayRenderBindGroup: GrayRenderBindGroup
  frameId: number
  state: FrameSlotState
}

export interface FrameSlotPool {
  /** Returns a free slot (transitions it to `inflight`) or `undefined` if none available. */
  acquireFreeSlot(): FrameSlot | undefined
  /** Returns an inflight slot back to `free`. Call on error or disposal. */
  releaseSlot(slot: FrameSlot): void
  /**
   * Promotes `slot` to `display` and demotes the previous display slot to `free`.
   * In development, asserts `slot.frameId` is monotonically increasing.
   */
  swapDisplaySlot(slot: FrameSlot): void
  /**
   * Appends three `copyBufferToBuffer` commands into `enc`:
   *   grayBuffer → slot.graySnapshot
   *   compactLabelBuffer → slot.labelStaging
   *   filteredBuffer → slot.filteredStaging
   */
  enqueueCopiesForSlot(enc: GPUCommandEncoder, pip: CameraPipeline, slot: FrameSlot): void
  /** The slot currently shown on the canvas (may be undefined before first detection). */
  displaySlot: FrameSlot | undefined
}

export function createFrameSlotPool(
  root: TgpuRoot,
  options: {
    width: number
    height: number
    grayRenderTimeBuffer: GrayRenderBindResources['timeSec']
    slotCount?: number
  },
): FrameSlotPool {
  const { width, height, grayRenderTimeBuffer, slotCount = 3 } = options
  const area = width * height

  const slots: FrameSlot[] = Array.from({ length: slotCount }, () => {
    const graySnapshot = root.createBuffer(d.arrayOf(d.f32, area)).$usage('storage')
    const labelStaging = root.device.createBuffer({
      size: area * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const filteredStaging = root.device.createBuffer({
      size: area * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const grayRenderBindGroup = root.createBindGroup(grayRenderLayout, {
      grayBuffer: graySnapshot,
      timeSec: grayRenderTimeBuffer,
    })
    return {
      graySnapshot,
      labelStaging,
      filteredStaging,
      grayRenderBindGroup,
      frameId: -1,
      state: 'free' as FrameSlotState,
    }
  })

  let displaySlot: FrameSlot | undefined = undefined

  return {
    get displaySlot() {
      return displaySlot
    },

    acquireFreeSlot() {
      const slot = slots.find((s) => s.state === 'free')
      if (!slot) {
        return undefined
      }
      slot.state = 'inflight'
      return slot
    },

    releaseSlot(slot) {
      slot.state = 'free'
    },

    swapDisplaySlot(slot) {
      if (import.meta.env.DEV && displaySlot !== undefined) {
        if (slot.frameId <= displaySlot.frameId) {
          console.warn(
            `[frameSlotPool] swapDisplaySlot: non-monotonic frameId (new=${slot.frameId}, current=${displaySlot.frameId})`,
          )
        }
      }
      if (displaySlot !== undefined) {
        displaySlot.state = 'free'
      }
      slot.state = 'display'
      displaySlot = slot
    },

    enqueueCopiesForSlot(enc, pip, slot) {
      const grayStorage = pip.gray.buffer.buffer
      const grayDst = slot.graySnapshot.buffer
      enc.copyBufferToBuffer(grayStorage, 0, grayDst, 0, grayStorage.size)

      const labelStorage = pip.compact.compactLabelBuffer.buffer
      enc.copyBufferToBuffer(labelStorage, 0, slot.labelStaging, 0, labelStorage.size)

      const edgeStorage = pip.nms.filteredBuffer.buffer
      enc.copyBufferToBuffer(edgeStorage, 0, slot.filteredStaging, 0, edgeStorage.size)
    },
  }
}

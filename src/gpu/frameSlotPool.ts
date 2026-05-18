export type FrameSlotState = 'free' | 'inflight'

/** Lightweight backpressure token — no GPU resources (gray/grid stay on live pipeline buffers). */
export interface FrameSlot {
  frameId: number
  state: FrameSlotState
}

export interface FrameSlotPool {
  acquireFreeSlot(): FrameSlot | undefined
  releaseSlot(slot: FrameSlot): void
}

let nextFrameId = 0

export function createFrameSlotPool(options?: { slotCount?: number }): FrameSlotPool {
  const slotCount = options?.slotCount ?? 3

  const slots: FrameSlot[] = Array.from({ length: slotCount }, () => ({
    frameId: -1,
    state: 'free' as FrameSlotState,
  }))

  return {
    acquireFreeSlot() {
      const slot = slots.find((s) => s.state === 'free')
      if (!slot) {
        return undefined
      }
      slot.state = 'inflight'
      slot.frameId = nextFrameId++
      return slot
    },

    releaseSlot(slot) {
      slot.state = 'free'
    },
  }
}

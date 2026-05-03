import { describe, expect, it } from 'vitest'

import { objectPointRowKey, unpackObjectPointRowKey } from '@/lib/objectPointRowKey'

describe('objectPointRowKey', () => {
  function roundTrip(tagId: number, cornerId: number) {
    const k = objectPointRowKey(tagId, cornerId)
    expect(unpackObjectPointRowKey(k)).toEqual({ tagId, cornerId })
  }

  it('round-trips dictionary and zero', () => {
    for (const tagId of [0, 1, 42, 586]) {
      for (let c = 0; c < 4; c++) {
        roundTrip(tagId, c)
      }
    }
  })

  it('round-trips negative (custom) ids', () => {
    for (const tagId of [-1, -3, -42, -(2 ** 36)]) {
      for (let c = 0; c < 4; c++) {
        roundTrip(tagId, c)
      }
    }
  })
})

import { describe, expect, it } from 'vitest'

import { canonicalRingDescriptor } from '@/lib/pointIdentity/identity'
import { createRandom } from '@/lib/pointIdentity/random'

const BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'

/**
 * The pre-bit-packing implementation, verbatim. Still reachable in production
 * for rings wider than 32, and used here as the oracle for every width.
 */
function legacyCanonical(
  colors: string,
  chainIds: Int32Array,
  chainCount: number,
  endpoints: string,
  rootChain: number,
): { colors: string; partition: string; endpoints: string } {
  if (chainCount <= 1) {
    return { colors, partition: '0', endpoints }
  }

  const distinctChains: number[] = []
  const distinctLabels: number[] = []
  let bestColors = ''
  let bestPartition = ''
  let bestEndpoints = ''
  let hasBest = false

  for (let shift = 0; shift < chainCount; shift += 1) {
    const candidateColors = colors.slice(shift) + colors.slice(0, shift)
    const candidateEndpoints = endpoints.slice(shift) + endpoints.slice(0, shift)

    distinctChains.length = 0
    distinctLabels.length = 0
    let nextLabel = rootChain >= 0 ? 1 : 0
    let candidatePartition = ''

    for (let index = 0; index < chainCount; index += 1) {
      const chainId = chainIds[(shift + index) % chainCount] ?? -1
      let label = -1
      if (rootChain >= 0 && chainId === rootChain) {
        label = 0
      } else {
        for (let known = 0; known < distinctChains.length; known += 1) {
          if (distinctChains[known] === chainId) {
            label = distinctLabels[known]!
            break
          }
        }
        if (label < 0) {
          label = nextLabel
          nextLabel += 1
          distinctChains.push(chainId)
          distinctLabels.push(label)
        }
      }
      candidatePartition += BASE36[label]
    }

    if (
      !hasBest ||
      candidateColors < bestColors ||
      (candidateColors === bestColors &&
        (candidatePartition < bestPartition ||
          (candidatePartition === bestPartition && candidateEndpoints < bestEndpoints)))
    ) {
      bestColors = candidateColors
      bestPartition = candidatePartition
      bestEndpoints = candidateEndpoints
      hasBest = true
    }
  }

  return { colors: bestColors, partition: bestPartition, endpoints: bestEndpoints }
}

function asText(values: Uint8Array, count: number): string {
  let text = ''
  for (let index = 0; index < count; index += 1) {
    text += values[index] === 1 ? '1' : '0'
  }
  return text
}

function check(
  colors: Uint8Array,
  endpoints: Uint8Array,
  chains: Int32Array,
  count: number,
  rootChain: number,
): void {
  const expected = legacyCanonical(
    asText(colors, count),
    chains,
    count,
    asText(endpoints, count),
    rootChain,
  )
  const actual = canonicalRingDescriptor(colors, endpoints, chains, count, rootChain)
  expect(actual.size).toBe(count)
  expect(actual.colors).toBe(expected.colors)
  expect(actual.partition).toBe(expected.partition)
  expect(actual.endpoints).toBe(expected.endpoints)
}

describe('ring canonicalisation', () => {
  it('matches the string implementation for every ring width', () => {
    const random = createRandom(0x51ed270b)
    for (let count = 0; count <= 40; count += 1) {
      for (let trial = 0; trial < 12; trial += 1) {
        const colors = new Uint8Array(count)
        const endpoints = new Uint8Array(count)
        const chains = new Int32Array(count)
        // A small chain-id pool makes repeats and rotations common.
        const chainPool = 1 + Math.floor(random() * 6)
        for (let index = 0; index < count; index += 1) {
          colors[index] = random() < 0.5 ? 0 : 1
          endpoints[index] = random() < 0.4 ? 0 : 1
          chains[index] = Math.floor(random() * chainPool)
        }
        const rootChain = trial % 3 === 0 ? -1 : chains[0]!
        check(colors, endpoints, chains, count, rootChain)
      }
    }
  })

  it('matches on periodic rings where many rotations tie', () => {
    // Uniform colour means every rotation reaches the same colour word, so the
    // chain partition alone decides the winner.
    for (let count = 1; count <= 40; count += 1) {
      const colors = new Uint8Array(count)
      const endpoints = new Uint8Array(count)
      const chains = new Int32Array(count)
      for (let index = 0; index < count; index += 1) {
        chains[index] = index % 3
      }
      for (const rootChain of [-1, 0, 1, 2]) {
        check(colors, endpoints, chains, count, rootChain)
      }
    }
  })

  it('matches with a root chain that is absent from the ring', () => {
    const count = 9
    const colors = new Uint8Array(count)
    const endpoints = new Uint8Array(count)
    const chains = new Int32Array(count)
    for (let index = 0; index < count; index += 1) {
      colors[index] = index % 2
      endpoints[index] = index % 3 === 0 ? 1 : 0
      chains[index] = index % 2
    }
    check(colors, endpoints, chains, count, 99)
  })

  it('keeps a single-element ring unchanged', () => {
    const colors = Uint8Array.from([1])
    const endpoints = Uint8Array.from([1])
    const chains = Int32Array.from([7])
    const ring = canonicalRingDescriptor(colors, endpoints, chains, 1, 7)
    expect(ring).toEqual({ size: 1, colors: '1', partition: '0', endpoints: '1' })
  })
})

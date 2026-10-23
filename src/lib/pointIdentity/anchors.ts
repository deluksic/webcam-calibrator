import {
  fingerprintKeyAtDepth,
  fingerprintRingKey,
  fingerprintRootKey,
} from '@/lib/pointIdentity/identity'
import type {
  BinaryColor,
  IdentityIndex,
  PatternPoint,
  PointFingerprint,
  ProximityGraph,
  Vec2,
} from '@/lib/pointIdentity/types'

/** A stable design identity recovered from a fingerprint prefix. */
export type ProofPowerComputation = {
  designId?: number
  /** Deepest depth whose design-unique prefix still agrees with the candidate. */
  depth: number
  /** A deeper unique prefix disagreed with an earlier unique prefix. */
  unstable: boolean
}

export type Anchor = {
  observedPosition: number
  designId: number
  proofPower: number
  design: Vec2
  observed: Vec2
  color: BinaryColor
}

export type AnchorCollection = {
  anchors: Anchor[]
  proofPowers: ProofPowerComputation[]
  /** Proof-power threshold that produced the returned anchors. */
  usedProofPower: number
  unstableCount: number
  selectedDepthConflictCount: number
  fallbackUsed: boolean
}

export type AnchorOptions = {
  identityDepth: number
  maxProofDepth: number
  minProofPower: number
  minAnchors: number
  /** Design-space points keyed by their stable design id. */
  designPointsByIndex: ReadonlyMap<number, PatternPoint>
  /** Retry one hop lower when too few anchors survive (floor 3). */
  allowProofFallback?: boolean
}

/**
 * Proof power is the deepest fingerprint prefix that is design-unique and
 * agrees with every earlier design-unique prefix. Ambiguities are allowed; a
 * deeper unique prefix that disagrees with an earlier one makes the point
 * unstable and therefore unsafe as an anchor.
 */
export function computeProofPower(
  fingerprint: PointFingerprint,
  designIndexesByDepth: readonly IdentityIndex[],
  maxProofDepth: number,
): ProofPowerComputation {
  const depthLimit = Math.min(Math.max(0, Math.floor(maxProofDepth)), designIndexesByDepth.length)
  let designId: number | undefined
  let depth = 0
  let unstable = false
  let key = fingerprintRootKey(fingerprint.color, fingerprint.endpoint)

  for (let currentDepth = 1; currentDepth <= depthLimit; currentDepth += 1) {
    const designIndex = designIndexesByDepth[currentDepth - 1]
    if (designIndex === undefined) {
      break
    }
    // Depth 1 is the root color plus the first ring, matching fingerprintKeyAtDepth.
    const ring = fingerprint.rings[currentDepth - 1]
    if (ring !== undefined) {
      key += `|${fingerprintRingKey(ring)}`
    }
    const candidates = designIndex.byKey.get(key)
    if (candidates !== undefined && candidates.length === 1) {
      const candidate = candidates[0]!
      if (designId !== undefined && candidate !== designId) {
        unstable = true
        break
      }
      designId = candidate
      depth = currentDepth
    }
  }

  return { designId, depth, unstable }
}

/**
 * Collect proof-power anchors from a full-depth fingerprint pass.
 *
 * `observedFingerprints` must be the full-depth fingerprints for the observed
 * graph. The selected-depth disagreement check uses the same full-depth
 * fingerprint's prefix, so no second graph traversal is needed.
 */
export function collectAnchors(
  observedGraph: ProximityGraph,
  observedColors: readonly BinaryColor[],
  observedFingerprints: readonly PointFingerprint[],
  designIndexesByDepth: readonly IdentityIndex[],
  options: AnchorOptions,
): AnchorCollection {
  const proofPowers = observedFingerprints.map((fingerprint) =>
    computeProofPower(fingerprint, designIndexesByDepth, options.maxProofDepth),
  )

  const buildAt = (threshold: number): { anchors: Anchor[]; conflicts: number } => {
    const selectedDepthIndex = designIndexesByDepth[options.identityDepth - 1]
    const anchors: Anchor[] = []
    let conflicts = 0

    for (let observedPosition = 0; observedPosition < observedFingerprints.length; observedPosition += 1) {
      const fingerprint = observedFingerprints[observedPosition]!
      const proof = proofPowers[observedPosition]!
      if (proof.designId === undefined || proof.depth < threshold || proof.unstable) {
        continue
      }

      if (selectedDepthIndex !== undefined) {
        const selectedCandidates = selectedDepthIndex.byKey.get(
          fingerprintKeyAtDepth(fingerprint, options.identityDepth),
        )
        if (
          selectedCandidates !== undefined &&
          selectedCandidates.length === 1 &&
          selectedCandidates[0] !== proof.designId
        ) {
          conflicts += 1
          continue
        }
      }

      const observed = observedGraph.points[observedPosition]
      const design = options.designPointsByIndex.get(proof.designId)
      if (observed === undefined || design === undefined) {
        continue
      }

      anchors.push({
        observedPosition,
        designId: proof.designId,
        proofPower: proof.depth,
        design: { x: design.x, y: design.y },
        observed: { x: observed.x, y: observed.y },
        color: observedColors[observedPosition] ?? 0,
      })
    }

    return { anchors, conflicts }
  }

  let usedProofPower = Math.max(1, Math.floor(options.minProofPower))
  let selected = buildAt(usedProofPower)
  let fallbackUsed = false
  if (
    selected.anchors.length < Math.max(1, options.minAnchors) &&
    (options.allowProofFallback ?? true) &&
    usedProofPower > 3
  ) {
    const fallbackThreshold = Math.max(3, usedProofPower - 1)
    if (fallbackThreshold !== usedProofPower) {
      const fallback = buildAt(fallbackThreshold)
      if (fallback.anchors.length > selected.anchors.length) {
        selected = fallback
        usedProofPower = fallbackThreshold
        fallbackUsed = true
      }
    }
  }

  const unstableCount = proofPowers.filter((proof) => proof.unstable).length
  return {
    anchors: selected.anchors,
    proofPowers,
    usedProofPower,
    unstableCount,
    selectedDepthConflictCount: selected.conflicts,
    fallbackUsed,
  }
}

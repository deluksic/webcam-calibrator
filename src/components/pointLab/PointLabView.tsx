import { createMemo, createSignal } from 'solid-js'

import { ControlSlider } from '@/components/pointLab/ControlSlider'
import { PointLabCanvas } from '@/components/pointLab/PointLabCanvas'
import { extractChainRuns, summarizeChainRuns } from '@/lib/pointIdentity/chainRuns'
import type { IcpRefineResult } from '@/lib/pointIdentity/icpRefiner'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  createLocalIcpOptions,
  MAX_FINGERPRINT_DEPTH,
  refineObservedIdentityWithLocalIcp,
} from '@/lib/pointIdentity/localIcpRefiner'
import {
  computeFingerprints,
  createIdentityIndexesByDepth,
  evaluateIdentity,
} from '@/lib/pointIdentity/identity'
import {
  IDENTITY_AMBIGUOUS,
  IDENTITY_CONFLICT,
  IDENTITY_RECOVERED,
  IDENTITY_UNRESOLVED,
  matchObservedIdentity,
  type IdentityStatus,
} from '@/lib/pointIdentity/match'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, outOfPlaneTiltDeg, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { createRandom, randomBinaryColors } from '@/lib/pointIdentity/random'
import type { GraphTuning, MatchOutcome, ProximityGraph, ViewPose } from '@/lib/pointIdentity/types'

import styles from '@/components/pointLab/PointLabView.module.css'

const DEFAULTS = {
  columns: 8,
  pointsPerEdge: 4,
  cameraDistance: 2200,
} as const

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function degrees(value: number): string {
  return `${value.toFixed(0)}°`
}

function outcomeLabel(outcome: MatchOutcome): string {
  switch (outcome.kind) {
    case 'correct':
      return 'correct'
    case 'false-match':
      return 'false match'
    case 'ambiguous':
      return 'ambiguous'
    case 'unresolved':
      return 'unmatched'
  }
}

function outcomeDetail(outcome: MatchOutcome): string {
  switch (outcome.kind) {
    case 'correct':
      return `Resolved to design point #${outcome.matchedIndex}.`
    case 'false-match':
      return `Resolved to #${outcome.matchedIndex} — the wrong identity.`
    case 'ambiguous': {
      const shown = outcome.candidateIndices.slice(0, 5).join(', ')
      const suffix = outcome.candidateIndices.length > 5 ? ', …' : ''
      return `Matches ${outcome.candidateIndices.length} design points: ${shown}${suffix}.`
    }
    case 'unresolved':
      return 'No design point has this color-ring fingerprint.'
  }
}

function outcomeToneClass(outcome: MatchOutcome): string {
  switch (outcome.kind) {
    case 'correct':
      return styles.statusCorrect ?? ''
    case 'false-match':
      return styles.statusFalse ?? ''
    case 'ambiguous':
    case 'unresolved':
      return styles.statusSafe ?? ''
  }
}

function formatRing(ring: { size: number; colors: string; partition: string; endpoints: string }): string {
  return `${ring.size} pt${ring.size === 1 ? '' : 's'} · ${ring.colors} · p:${ring.partition} · e:${ring.endpoints}`
}

function refinedStatusLabel(status: IdentityStatus): string {
  switch (status) {
    case IDENTITY_RECOVERED:
      return 'recovered'
    case IDENTITY_CONFLICT:
      return 'conflict'
    default:
      return 'safe fail'
  }
}

function refinedToneClass(status: IdentityStatus): string {
  if (status === IDENTITY_RECOVERED) {
    return styles.statusCorrect ?? ''
  }
  if (status === IDENTITY_CONFLICT) {
    return styles.statusFalse ?? ''
  }
  return styles.statusSafe ?? ''
}

function degreeById(graph: ProximityGraph, id: number): number {
  const position = graph.points.findIndex((point) => point.index === id)
  return position < 0 ? 0 : (graph.neighbors[position]?.length ?? 0)
}

export function PointLabView() {
  const [columns, setColumns] = createSignal(DEFAULTS.columns)
  const [pointsPerEdge, setPointsPerEdge] = createSignal(DEFAULTS.pointsPerEdge)
  const [colorSeed, setColorSeed] = createSignal(1)

  const [yawDeg, setYawDeg] = createSignal(0)
  const [pitchDeg, setPitchDeg] = createSignal(0)
  const [cameraDistance, setCameraDistance] = createSignal(DEFAULTS.cameraDistance)
  const [sphericalDistortion, setSphericalDistortion] = createSignal(0)

  const [dropoutPercent, setDropoutPercent] = createSignal(DEFAULT_OBSERVATION_MASK.dropoutPercent)
  const [dropoutSeed, setDropoutSeed] = createSignal(DEFAULT_OBSERVATION_MASK.dropoutSeed)
  const [jitterPercent, setJitterPercent] = createSignal(0)
  const [cropLeft, setCropLeft] = createSignal(DEFAULT_OBSERVATION_MASK.cropLeft)
  const [cropRight, setCropRight] = createSignal(DEFAULT_OBSERVATION_MASK.cropRight)
  const [cropTop, setCropTop] = createSignal(DEFAULT_OBSERVATION_MASK.cropTop)
  const [cropBottom, setCropBottom] = createSignal(DEFAULT_OBSERVATION_MASK.cropBottom)

  const [showGraph, setShowGraph] = createSignal(true)
  const [zoomToCursor, setZoomToCursor] = createSignal(true)
  const [selectedIndex, setSelectedIndex] = createSignal<number | undefined>(undefined)
  const [edgeReveal, setEdgeReveal] = createSignal<number | undefined>(undefined)
  const [recoveryEnabled, setRecoveryEnabled] = createSignal(true)
  const [minProofPower, setMinProofPower] = createSignal(7)
  const [hopRadius, setHopRadius] = createSignal(8)
  const [showAnchors, setShowAnchors] = createSignal(true)
  const [showPredictedLattice, setShowPredictedLattice] = createSignal(true)
  const [showPredictedMissing, setShowPredictedMissing] = createSignal(true)

  const changePatternNumber = (setter: (value: number) => void) => (value: number) => {
    setter(value)
    setSelectedIndex(undefined)
  }

  const pattern = createMemo(() =>
    generateHexPattern({
      columns: columns(),
      pointsPerEdge: pointsPerEdge(),
    }),
  )

  const colors = createMemo(() => randomBinaryColors(pattern().points.length, colorSeed()))

  const pose = createMemo<ViewPose>(() => ({
    yawDeg: yawDeg(),
    pitchDeg: pitchDeg(),
    cameraDistance: cameraDistance(),
    focalLength: DEFAULT_VIEW.focalLength,
    sphericalDistortion: sphericalDistortion(),
  }))

  const projected = createMemo(() => projectPattern(pattern().points, pose()))
  const tuning = createMemo<GraphTuning>(() => ({ pointsPerEdge: pointsPerEdge() }))
  const designGraph = createMemo(() => buildProximityGraph(pattern().points, tuning()))
  const observedPoints = createMemo(() => {
    const masked = applyObservationMask(projected(), {
      cropLeft: cropLeft(),
      cropRight: cropRight(),
      cropTop: cropTop(),
      cropBottom: cropBottom(),
      dropoutPercent: dropoutPercent(),
      dropoutSeed: dropoutSeed(),
    })
    const jitter = jitterPercent()
    if (jitter <= 0) {
      return masked
    }
    const spacing = Math.max(designGraph().stats.localScaleMedian, 1e-6)
    const random = createRandom((dropoutSeed() + 0x51ed270b) >>> 0)
    return masked.map((point) => ({
      ...point,
      x: point.x + (random() * 2 - 1) * spacing * (jitter / 100),
      y: point.y + (random() * 2 - 1) * spacing * (jitter / 100),
    }))
  })
  const identityIndexesByDepth = createMemo(() =>
    createIdentityIndexesByDepth(designGraph(), colors(), MAX_FINGERPRINT_DEPTH),
  )
  const designIndex = createMemo(
    () => identityIndexesByDepth()[MAX_FINGERPRINT_DEPTH - 1]!,
  )
  const observedGraph = createMemo(() => buildProximityGraph(observedPoints(), tuning()))
  const observedFingerprints = createMemo(() =>
    computeFingerprints(observedGraph(), colors(), MAX_FINGERPRINT_DEPTH),
  )
  const evaluation = createMemo(() =>
    evaluateIdentity(
      designGraph(),
      observedGraph(),
      designIndex(),
      colors(),
      MAX_FINGERPRINT_DEPTH,
      observedFingerprints(),
    ),
  )
  const observedColors = createMemo(() => observedGraph().points.map((point) => colors()[point.index] ?? 0))
  const baselineMatch = createMemo(() =>
    matchObservedIdentity(designIndex(), observedGraph(), colors(), MAX_FINGERPRINT_DEPTH, {
      fingerprints: observedFingerprints(),
    }),
  )
  const frameBounds = createMemo(() => {
    const points = projected()
    if (points.length === 0) {
      return undefined
    }
    let minX = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY
    for (const point of points) {
      minX = Math.min(minX, point.x)
      maxX = Math.max(maxX, point.x)
      minY = Math.min(minY, point.y)
      maxY = Math.max(maxY, point.y)
    }
    return { minX, minY, maxX, maxY }
  })
  const recovery = createMemo<IcpRefineResult>(() =>
    refineObservedIdentityWithLocalIcp({
      designPattern: pattern(),
      designColors: colors(),
      designGraph: designGraph(),
      designIndexesByDepth: identityIndexesByDepth(),
      observedGraph: observedGraph(),
      observedColors: observedColors(),
      observedFingerprints: observedFingerprints(),
      baseline: baselineMatch(),
      options: createLocalIcpOptions({
        enabled: recoveryEnabled(),
        identityDepth: MAX_FINGERPRINT_DEPTH,
        maxProofDepth: MAX_FINGERPRINT_DEPTH,
        minProofPower: minProofPower(),
        hopRadius: hopRadius(),
        frameBounds: frameBounds(),
      }),
    }),
  )
  const recoverySummary = createMemo(() => {
    const result = recovery()
    const observed = observedGraph().points
    const truthIds = new Set(observed.map((point) => point.index))
    let correct = 0
    let falseMatch = 0
    let recoveredFalse = 0
    let recovered = 0
    let conflicts = 0
    let safe = 0
    let remainingAmbiguous = 0
    let remainingUnresolved = 0
    for (let position = 0; position < result.points.length; position += 1) {
      const point = result.points[position]
      const truth = observed[position]
      if (point === undefined || truth === undefined) {
        continue
      }
      if (point.status === IDENTITY_RECOVERED) {
        recovered += 1
      }
      if (point.status === IDENTITY_AMBIGUOUS) {
        remainingAmbiguous += 1
      }
      if (point.status === IDENTITY_UNRESOLVED) {
        remainingUnresolved += 1
      }
      if (point.status === IDENTITY_CONFLICT) {
        conflicts += 1
      }
      if (point.designId === undefined) {
        safe += 1
      } else if (point.designId === truth.index) {
        correct += 1
      } else {
        falseMatch += 1
        if (point.status === IDENTITY_RECOVERED) {
          recoveredFalse += 1
        }
      }
    }
    let baselineAmbiguous = 0
    let baselineUnresolved = 0
    for (const point of result.baseline.points) {
      if (point.status === IDENTITY_AMBIGUOUS) {
        baselineAmbiguous += 1
      }
      if (point.status === IDENTITY_UNRESOLVED) {
        baselineUnresolved += 1
      }
    }
    let predictedMissingFalse = 0
    for (const missing of result.predictedMissing) {
      if (truthIds.has(missing.designId)) {
        predictedMissingFalse += 1
      }
    }
    const total = result.points.length
    return {
      correct,
      falseMatch,
      recoveredFalse,
      recovered,
      conflicts,
      safe,
      baselineAmbiguous,
      baselineUnresolved,
      remainingAmbiguous,
      remainingUnresolved,
      total,
      predictedMissingFalse,
      accuracy: total === 0 ? 0 : correct / total,
    }
  })

  const expectedChains = createMemo(() => pattern().stats.sampledEdges)
  const chainDiagnostics = createMemo(() =>
    summarizeChainRuns(extractChainRuns(observedGraph()), pointsPerEdge(), expectedChains()),
  )
  const designChainDiagnostics = createMemo(() =>
    summarizeChainRuns(extractChainRuns(designGraph()), pointsPerEdge(), expectedChains()),
  )
  const histogramMax = createMemo(() => Math.max(1, ...chainDiagnostics().histogram.map((bin) => bin.count)))

  const edgeCount = createMemo(() => observedGraph().edges.length)
  const visibleEdgeCount = createMemo(() => {
    const total = edgeCount()
    const value = edgeReveal()
    return value === undefined ? total : Math.min(value, total)
  })
  const isScrubbingEdges = createMemo(() => visibleEdgeCount() < edgeCount())

  const latestEdgeLabel = (): string => {
    if (!isScrubbingEdges()) {
      return 'all edges'
    }
    const count = visibleEdgeCount()
    const edge = count === 0 ? undefined : observedGraph().edges[count - 1]
    if (edge === undefined) {
      return 'no edges shown'
    }
    const first = observedGraph().points[edge[0]]?.index
    const second = observedGraph().points[edge[1]]?.index
    return first === undefined || second === undefined ? 'no edges shown' : `latest #${first}–#${second}`
  }

  const scrubEdges = (value: number): void => {
    setEdgeReveal(value >= edgeCount() ? undefined : value)
  }

  const summary = createMemo(() => evaluation().summary)
  const tilt = createMemo(() => outOfPlaneTiltDeg(pose()))
  const uniquePercent = createMemo(() => {
    const total = designIndex().fingerprints.length
    return total === 0 ? 0 : summary().designUniquePoints / total
  })

  const selectedDetails = createMemo(() => {
    const index = selectedIndex()
    if (index === undefined) {
      return undefined
    }
    const item = evaluation().perPoint.find((entry) => entry.index === index)
    if (item === undefined) {
      return undefined
    }
    const fingerprint = designIndex().fingerprints.find((entry) => entry.index === index)
    const observedPosition = observedGraph().points.findIndex((entry) => entry.index === index)
    const refinedPoint = observedPosition >= 0 ? recovery().points[observedPosition] : undefined
    const isAnchor = recovery().anchors.some((anchor) => anchor.observedPosition === observedPosition)
    return {
      item,
      color: colors()[index] ?? 0,
      designFingerprint: fingerprint,
      designDegree: degreeById(designGraph(), index),
      observedDegree: degreeById(observedGraph(), index),
      refinedPoint,
      isAnchor,
    }
  })

  const graphRejections = createMemo(() => {
    const stats = observedGraph().stats
    return [`degree ${stats.rejections.degree}`, `crossing ${stats.rejections.crossing}`].join(' · ')
  })

  const resetLattice = (): void => {
    setColumns(DEFAULTS.columns)
    setPointsPerEdge(DEFAULTS.pointsPerEdge)
    setSelectedIndex(undefined)
  }

  const reshuffleColors = (): void => {
    setColorSeed((seed) => seed + 1)
  }

  const resampleDropouts = (): void => {
    setDropoutSeed((seed) => seed + 1)
  }

  const resetView = (): void => {
    setYawDeg(0)
    setPitchDeg(0)
    setCameraDistance(DEFAULTS.cameraDistance)
    setSphericalDistortion(0)
  }

  const displayedFalseMatch = (): number => (recoveryEnabled() ? recoverySummary().falseMatch : summary().falseMatch)
  const displayedAccuracy = (): number => (recoveryEnabled() ? recoverySummary().accuracy : summary().accuracy)
  const accuracyTone = (): string =>
    displayedFalseMatch() > 0
      ? (styles.metricDanger ?? '')
      : displayedAccuracy() >= 0.9
        ? (styles.metricSuccess ?? '')
        : (styles.metricWarning ?? '')

  const renderInspector = () => {
    const details = selectedDetails()
    if (details === undefined) {
      return (
        <p class={styles.inspectorPlaceholder}>
          Click any projected point to inspect its detected color, graph neighborhood and color-ring fingerprint.
        </p>
      )
    }

    return (
      <>
        <div class={styles.inspectorTop}>
          <span class={styles.inspectorIndex}>#{details.item.index}</span>
          <span class={`${styles.colorChip} ${details.color === 1 ? styles.colorChipOne : styles.colorChipZero}`}>
            color {details.color}
          </span>
          <span class={`${styles.statusChip} ${outcomeToneClass(details.item.outcome)}`}>
            {outcomeLabel(details.item.outcome)}
          </span>
          {recoveryEnabled() && details.refinedPoint !== undefined ? (
            <span class={`${styles.statusChip} ${refinedToneClass(details.refinedPoint.status)}`}>
              ICP: {refinedStatusLabel(details.refinedPoint.status)}
            </span>
          ) : null}
          <span class={styles.inspectorRoute}>{outcomeDetail(details.item.outcome)}</span>
        </div>
        <div class={styles.inspectorBody}>
          <div class={styles.ringStrip}>
            <span class={styles.ringChip}>own color · {details.item.fingerprint.color}</span>
            {details.item.fingerprint.rings.map((ring, index) => (
              <span class={styles.ringChip}>
                hop {index + 1} · {formatRing(ring)}
              </span>
            ))}
          </div>
          <dl class={styles.inspectorStats}>
            <div>
              <dt>design degree</dt>
              <dd>{details.designDegree}</dd>
            </div>
            <div>
              <dt>observed degree</dt>
              <dd>{details.observedDegree}</dd>
            </div>
            <div>
              <dt>role</dt>
              <dd>{details.item.fingerprint.endpoint ? 'endpoint' : 'interior'}</dd>
            </div>
            <div>
              <dt>key length</dt>
              <dd>{details.item.fingerprint.key.length}</dd>
            </div>
            <div>
              <dt>proof power</dt>
              <dd>{details.refinedPoint?.proofPower ?? '—'}</dd>
            </div>
            <div>
              <dt>anchor</dt>
              <dd>{details.isAnchor ? 'yes' : 'no'}</dd>
            </div>
            <div>
              <dt>ICP residual</dt>
              <dd>
                {details.refinedPoint?.designDistance === undefined
                  ? '—'
                  : details.refinedPoint.designDistance.toFixed(2)}
              </dd>
            </div>
            <div>
              <dt>votes</dt>
              <dd>{details.refinedPoint?.recoveryVotes ?? '—'}</dd>
            </div>
            <div>
              <dt>source</dt>
              <dd>{details.refinedPoint?.source ?? '—'}</dd>
            </div>
            <div>
              <dt>conflict</dt>
              <dd>{details.refinedPoint?.geometryConflict === true ? 'yes' : 'no'}</dd>
            </div>
          </dl>
        </div>
      </>
    )
  }

  return (
    <div class={styles.page}>
      <header class={styles.hero}>
        <div class={styles.heroText}>
          <p class={styles.eyebrow}>Point identity lab</p>
          <h1 class={styles.title}>Can a binary color field identify itself?</h1>
          <p class={styles.lede}>
            A clean-room bench for the color-coded point target. The board is a direct flat-top honeycomb with
            half-offset edge points; pose and proximity graph are synthetic, so only identification is under test.
          </p>
        </div>
        <div class={styles.heroActions}>
          <button type="button" class={styles.primaryButton} onClick={reshuffleColors}>
            Reshuffle colors
          </button>
          <button type="button" class={styles.ghostButton} onClick={resetLattice}>
            Reset lattice
          </button>
        </div>
      </header>

      <section class={styles.metrics} aria-label="Experiment summary">
        <article class={`${styles.metric} ${accuracyTone()}`}>
          <span class={styles.metricLabel}>Correct identity</span>
          <strong class={styles.metricValue}>{percent(displayedAccuracy())}</strong>
          <span class={styles.metricDetail}>
            {recoveryEnabled()
              ? `${recoverySummary().correct} of ${recoverySummary().total} · baseline ${percent(summary().accuracy)} · ${recoverySummary().recovered} recovered`
              : `${summary().correct} of ${summary().total} projected points`}
          </span>
        </article>
        <article class={`${styles.metric} ${displayedFalseMatch() > 0 ? styles.metricDanger : styles.metricNeutral}`}>
          <span class={styles.metricLabel}>False matches</span>
          <strong class={styles.metricValue}>{displayedFalseMatch()}</strong>
          <span class={styles.metricDetail}>
            {recoveryEnabled()
              ? `baseline ${summary().falseMatch} · recovered false ${recoverySummary().recoveredFalse}`
              : `${percent(summary().falseMatchRate)} of the field · dangerous`}
          </span>
        </article>
        <article class={`${styles.metric} ${styles.metricWarning}`}>
          <span class={styles.metricLabel}>Safe failures</span>
          <strong class={styles.metricValue}>
            {recoveryEnabled() ? recoverySummary().safe : summary().safeFail}
          </strong>
          <span class={styles.metricDetail}>
            {recoveryEnabled()
              ? `${recoverySummary().conflicts} conflicts · baseline safe ${summary().safeFail} · ICP ${
                  recovery().accepted ? 'accepted' : recovery().rejectionReason ?? 'idle'
                }`
              : `${summary().unresolved} unmatched · ${summary().ambiguous} ambiguous`}
          </span>
        </article>
        <article class={`${styles.metric} ${styles.metricNeutral}`}>
          <span class={styles.metricLabel}>Unique design IDs</span>
          <strong class={styles.metricValue}>{percent(uniquePercent())}</strong>
          <span class={styles.metricDetail}>{summary().designCollisions} colliding points</span>
        </article>
      </section>

      <div class={styles.workspace}>
        <section class={styles.stageCard}>
          <div class={styles.stageHead}>
            <div>
              <h2 class={styles.stageTitle}>Projected field</h2>
              <p class={styles.stageSub}>
                yaw {degrees(yawDeg())} · pitch {degrees(pitchDeg())} · out-of-plane {tilt().toFixed(1)}°
              </p>
            </div>
            <div class={styles.stageToggles}>
              <label class={styles.toggle}>
                <input
                  type="checkbox"
                  checked={showGraph()}
                  onChange={(event) => setShowGraph(event.currentTarget.checked)}
                />
                Graph
              </label>
              <label class={styles.toggle} title="Wheel over the canvas to zoom around the pointer">
                <input
                  type="checkbox"
                  checked={zoomToCursor()}
                  onChange={(event) => setZoomToCursor(event.currentTarget.checked)}
                />
                Zoom to cursor
              </label>
            </div>
          </div>

          <div class={styles.canvasShell}>
            <PointLabCanvas
              projected={observedPoints()}
              framePoints={projected()}
              colors={colors()}
              graph={observedGraph()}
              evaluation={evaluation()}
              recovery={recovery()}
              showGraph={showGraph()}
              showRecovery={recoveryEnabled()}
              showAnchors={showAnchors()}
              showPredictedLattice={showPredictedLattice()}
              showPredictedMissing={showPredictedMissing()}
              zoomToCursor={zoomToCursor()}
              visibleEdgeCount={visibleEdgeCount()}
              selectedIndex={selectedIndex()}
              onSelect={setSelectedIndex}
            />
          </div>

          <div class={styles.legendBar}>
            <span class={styles.legendGroupLabel}>Detected color</span>
            <span class={styles.legendItem}>
              <span class={`${styles.colorDot} ${styles.colorZero}`} />
              color 0
            </span>
            <span class={styles.legendItem}>
              <span class={`${styles.colorDot} ${styles.colorOne}`} />
              color 1
            </span>
            <span class={styles.legendDivider} />
            <span class={styles.legendGroupLabel}>Failures</span>
            <span class={styles.legendItem}>no ring = correct</span>
            <span class={styles.legendItem}>
              <span class={`${styles.statusRing} ${styles.statusSafe}`} />
              safe fail
            </span>
            <span class={styles.legendItem}>
              <span class={`${styles.statusRing} ${styles.statusFalse}`} />
              false match
            </span>
            <span class={styles.legendDivider} />
            <span class={styles.legendGroupLabel}>ICP</span>
            <span class={styles.legendItem}>
              <span class={`${styles.statusRing} ${styles.statusRecovered}`} />
              recovered
            </span>
            <span class={styles.legendItem}>
              <span class={`${styles.statusRing} ${styles.statusFalse}`} />
              recovered false
            </span>
            <span class={styles.legendItem}>
              <span class={`${styles.statusRing} ${styles.statusConflict}`} />
              conflict
            </span>
            <span class={styles.legendItem}>
              <span class={styles.anchorSwatch} />
              anchor
            </span>
            <span class={styles.legendItem}>
              <span class={styles.predictedSwatch} />
              predicted missing
            </span>
            <span class={styles.legendDivider} />
            <span class={styles.legendItem}>
              <span class={styles.latestEdgeSwatch} />
              latest added edge
            </span>
          </div>

          <div class={styles.scrubber}>
            <label class={styles.scrubberLabel} for="lab-edge-scrub">
              Edge build
            </label>
            <input
              id="lab-edge-scrub"
              class={`${styles.range} ${styles.scrubberRange}`}
              type="range"
              min={0}
              max={edgeCount()}
              step={1}
              value={visibleEdgeCount()}
              onInput={(event) => scrubEdges(Number(event.currentTarget.value))}
            />
            <span class={styles.scrubberValue}>
              {visibleEdgeCount()} / {edgeCount()}
            </span>
            <span class={styles.scrubberLatest}>{latestEdgeLabel()}</span>
            <button
              type="button"
              class={styles.scrubberAll}
              onClick={() => setEdgeReveal(undefined)}
              disabled={!isScrubbingEdges()}
            >
              All
            </button>
          </div>

          <div class={styles.inspector}>{renderInspector()}</div>
        </section>

        <aside class={styles.sidebar} aria-label="Experiment controls">
          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>View</h2>
              <button type="button" class={styles.textButton} onClick={resetView}>
                Reset
              </button>
            </div>
            <ControlSlider
              id="lab-yaw"
              label="Yaw"
              value={yawDeg()}
              min={-80}
              max={80}
              step={1}
              format={degrees}
              onInput={setYawDeg}
            />
            <ControlSlider
              id="lab-pitch"
              label="Pitch"
              value={pitchDeg()}
              min={-80}
              max={80}
              step={1}
              format={degrees}
              onInput={setPitchDeg}
            />
            <ControlSlider
              id="lab-distance"
              label="Camera distance"
              value={cameraDistance()}
              min={1400}
              max={4200}
              step={50}
              format={(value) => value.toFixed(0)}
              onInput={setCameraDistance}
            />
            <ControlSlider
              id="lab-spherical"
              label="Spherical distortion"
              value={sphericalDistortion()}
              min={0}
              max={1}
              step={0.01}
              format={(value) => `${(value * 100).toFixed(0)}%`}
              onInput={setSphericalDistortion}
            />
            <p class={styles.panelNote}>
              Out-of-plane tilt {tilt().toFixed(1)}°. Distance is in design units; the canvas auto-fits, so only
              perspective foreshortening changes. Distortion compresses the outer field like a fisheye; 0% is a perfect
              pinhole.
            </p>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Observation</h2>
              <button type="button" class={styles.textButton} onClick={resampleDropouts}>
                Resample
              </button>
            </div>
            <ControlSlider
              id="lab-dropout"
              label="Dropout"
              value={dropoutPercent()}
              min={0}
              max={100}
              step={1}
              format={(value) => `${value.toFixed(0)}%`}
              onInput={setDropoutPercent}
            />
            <ControlSlider
              id="lab-jitter"
              label="Position jitter"
              value={jitterPercent()}
              min={0}
              max={5}
              step={0.25}
              format={(value) => `${value.toFixed(2)}%`}
              onInput={setJitterPercent}
            />
            <ControlSlider
              id="lab-crop-left"
              label="Crop left"
              value={cropLeft()}
              min={0}
              max={50}
              step={1}
              format={(value) => `${value.toFixed(0)}%`}
              onInput={setCropLeft}
            />
            <ControlSlider
              id="lab-crop-right"
              label="Crop right"
              value={cropRight()}
              min={0}
              max={50}
              step={1}
              format={(value) => `${value.toFixed(0)}%`}
              onInput={setCropRight}
            />
            <ControlSlider
              id="lab-crop-top"
              label="Crop top"
              value={cropTop()}
              min={0}
              max={50}
              step={1}
              format={(value) => `${value.toFixed(0)}%`}
              onInput={setCropTop}
            />
            <ControlSlider
              id="lab-crop-bottom"
              label="Crop bottom"
              value={cropBottom()}
              min={0}
              max={50}
              step={1}
              format={(value) => `${value.toFixed(0)}%`}
              onInput={setCropBottom}
            />
            <dl class={styles.inlineStats}>
              <div>
                <dt>detected</dt>
                <dd>
                  {observedPoints().length} / {pattern().points.length}
                </dd>
              </div>
              <div>
                <dt>dropout seed</dt>
                <dd>{dropoutSeed()}</dd>
              </div>
              <div>
                <dt>jitter</dt>
                <dd>{jitterPercent().toFixed(2)}%</dd>
              </div>
            </dl>
            <p class={styles.panelNote}>
              Crop margins are relative to the projected point bounds. Points are removed before graph construction;
              jitter is a fraction of median spacing, applied after masking. Design identity and colors remain
              unchanged.
            </p>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Lattice</h2>
              <span class={styles.seedLabel}>{pattern().stats.cells} cells</span>
            </div>
            <ControlSlider
              id="lab-columns"
              label="Columns"
              value={columns()}
              min={3}
              max={14}
              step={1}
              format={(value) => value.toFixed(0)}
              onInput={changePatternNumber(setColumns)}
            />
            <ControlSlider
              id="lab-points-per-edge"
              label="Points per edge"
              value={pointsPerEdge()}
              min={4}
              max={6}
              step={1}
              format={(value) => value.toFixed(0)}
              onInput={changePatternNumber(setPointsPerEdge)}
            />
            <dl class={styles.inlineStats}>
              <div>
                <dt>cells</dt>
                <dd>{pattern().stats.cells}</dd>
              </div>
              <div>
                <dt>edges</dt>
                <dd>{pattern().stats.sampledEdges}</dd>
              </div>
              <div>
                <dt>points</dt>
                <dd>{pattern().stats.points}</dd>
              </div>
              <div>
                <dt>per edge</dt>
                <dd>{pointsPerEdge()}</dd>
              </div>
            </dl>
            <p class={styles.panelNote}>
              Direct flat-top honeycomb. Shared cell edges are sampled once at half-step offsets; no triangulation,
              Voronoi construction or jitter is involved.
            </p>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Topology</h2>
              <span class={styles.seedLabel}>chain-first</span>
            </div>
            <p class={styles.panelNote}>
              Exact N-point chains; three-chain corners close as triangles, two-chain boundary corners as single edges.
            </p>
            <dl class={styles.inlineStats}>
              <div>
                <dt>edges</dt>
                <dd>{observedGraph().stats.edges}</dd>
              </div>
              <div>
                <dt>mean deg</dt>
                <dd>{observedGraph().stats.meanDegree.toFixed(2)}</dd>
              </div>
              <div>
                <dt>chain</dt>
                <dd>{observedGraph().stats.chainEdges}</dd>
              </div>
              <div>
                <dt>junction</dt>
                <dd>{observedGraph().stats.junctionEdges}</dd>
              </div>
              <div>
                <dt>isolated</dt>
                <dd>{observedGraph().stats.isolated}</dd>
              </div>
              <div>
                <dt>components</dt>
                <dd>{observedGraph().stats.components}</dd>
              </div>
            </dl>
            <p class={styles.panelNote}>
              Chain edges are admitted first; compact triangles then close the cell corners.
            </p>
            <p class={styles.panelNote}>Rejected: {graphRejections()}</p>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Chains</h2>
              <span class={styles.seedLabel}>N={pointsPerEdge()}</span>
            </div>
            <dl class={styles.inlineStats}>
              <div>
                <dt>chains</dt>
                <dd>
                  {chainDiagnostics().observedChains} / {chainDiagnostics().expectedChains}
                </dd>
              </div>
              <div>
                <dt>chain edges</dt>
                <dd>
                  {chainDiagnostics().observedChainEdges} / {chainDiagnostics().expectedChainEdges}
                </dd>
              </div>
              <div>
                <dt>short</dt>
                <dd>{chainDiagnostics().short}</dd>
              </div>
              <div>
                <dt>exact</dt>
                <dd>{chainDiagnostics().exact}</dd>
              </div>
              <div>
                <dt>long</dt>
                <dd>{chainDiagnostics().long}</dd>
              </div>
              <div>
                <dt>mean length</dt>
                <dd>{chainDiagnostics().meanLength.toFixed(1)}</dd>
              </div>
              <div>
                <dt>covered</dt>
                <dd>
                  {chainDiagnostics().coveredPoints} / {pattern().points.length}
                </dd>
              </div>
              <div>
                <dt>unchained</dt>
                <dd>{Math.max(0, pattern().points.length - chainDiagnostics().coveredPoints)}</dd>
              </div>
            </dl>
            <div class={styles.histogram}>
              {chainDiagnostics().histogram.map((bin) => (
                <div class={styles.histRow}>
                  <span class={styles.histLabel}>{bin.length}</span>
                  <span class={styles.histTrack}>
                    <span
                      class={`${styles.histBar} ${
                        bin.length < pointsPerEdge()
                          ? styles.histBarShort
                          : bin.length > pointsPerEdge()
                            ? styles.histBarLong
                            : styles.histBarExact
                      }`}
                      style={{ width: `${(bin.count / histogramMax()) * 100}%` }}
                    />
                  </span>
                  <span class={styles.histCount}>{bin.count}</span>
                </div>
              ))}
            </div>
            <p class={styles.panelNote}>
              Maximal collinear runs from admitted edges. Expected {expectedChains()} chains of {pointsPerEdge()} points
              ({chainDiagnostics().expectedChainEdges} edges). Design check: {designChainDiagnostics().observedChains}{' '}
              chains, {designChainDiagnostics().short} short, {designChainDiagnostics().long} long.
            </p>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Identity</h2>
              <span class={styles.seedLabel}>random colors</span>
            </div>
            <p class={styles.panelNote}>
              Fingerprint depth is fixed at the maximum {MAX_FINGERPRINT_DEPTH} hops. A key is the point's own bit plus
              angular color rings at each hop. Ring sizes are included, so topology drift cannot silently pass as a
              color match.
            </p>
            <dl class={styles.inlineStats}>
              <div>
                <dt>distinct keys</dt>
                <dd>{designIndex().keyCount}</dd>
              </div>
              <div>
                <dt>duplicate groups</dt>
                <dd>{designIndex().duplicateKeyCount}</dd>
              </div>
              <div>
                <dt>edge overlap</dt>
                <dd>{percent(summary().edgeJaccard)}</dd>
              </div>
              <div>
                <dt>degree drift</dt>
                <dd>{summary().degreeChanged}</dd>
              </div>
            </dl>
          </section>

          <section class={styles.panel}>
            <div class={styles.panelHead}>
              <h2 class={styles.panelTitle}>Geometric recovery</h2>
              <span class={styles.seedLabel}>
                {recovery().accepted ? 'accepted' : recovery().rejectionReason ?? 'idle'}
              </span>
            </div>
            <label class={styles.toggle}>
              <input
                type="checkbox"
                checked={recoveryEnabled()}
                onChange={(event) => setRecoveryEnabled(event.currentTarget.checked)}
              />
              Enable local color-gated ICP
            </label>
            <ControlSlider
              id="lab-anchor-proof"
              label="Anchor proof power"
              value={minProofPower()}
              min={2}
              max={MAX_FINGERPRINT_DEPTH}
              step={1}
              format={(value) => `${value.toFixed(0)} hops`}
              onInput={setMinProofPower}
            />
            <p class={styles.panelNote}>
              Only matches with at least this many stable unique hops become anchors. Identity depth itself is fixed at
              the maximum {MAX_FINGERPRINT_DEPTH} hops.
            </p>
            <ControlSlider
              id="lab-hop-radius"
              label="Support radius"
              value={hopRadius()}
              min={1}
              max={10}
              step={1}
              format={(value) => `${value.toFixed(0)} hops`}
              onInput={setHopRadius}
            />
            <div class={styles.recoveryToggles}>
              <label class={styles.toggle}>
                <input
                  type="checkbox"
                  checked={showAnchors()}
                  onChange={(event) => setShowAnchors(event.currentTarget.checked)}
                />
                Anchors
              </label>
              <label class={styles.toggle}>
                <input
                  type="checkbox"
                  checked={showPredictedLattice()}
                  onChange={(event) => setShowPredictedLattice(event.currentTarget.checked)}
                />
                Warped lattice
              </label>
              <label class={styles.toggle}>
                <input
                  type="checkbox"
                  checked={showPredictedMissing()}
                  onChange={(event) => setShowPredictedMissing(event.currentTarget.checked)}
                />
                Predicted missing
              </label>
            </div>
            <dl class={styles.inlineStats}>
              <div>
                <dt>anchors</dt>
                <dd>{recovery().metrics.anchors}</dd>
              </div>
              <div>
                <dt>registrations</dt>
                <dd>{recovery().metrics.neighborhoods ?? recovery().metrics.initialInliers}</dd>
              </div>
              <div>
                <dt>iterations</dt>
                <dd>{recovery().metrics.iterations}</dd>
              </div>
              <div>
                <dt>ICP correspondences</dt>
                <dd>{recovery().metrics.pairs}</dd>
              </div>
              <div>
                <dt>coverage</dt>
                <dd>{percent(recovery().metrics.pairCoverage)}</dd>
              </div>
              <div>
                <dt>design RMS</dt>
                <dd>
                  {recovery().accepted && recovery().metrics.designSpacing > 0
                    ? `${(recovery().metrics.designRms / recovery().metrics.designSpacing).toFixed(2)} sp`
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>recovered</dt>
                <dd>{recovery().metrics.recovered}</dd>
              </div>
              <div>
                <dt>recovered false</dt>
                <dd>{recoverySummary().recoveredFalse}</dd>
              </div>
              <div>
                <dt>ambiguous in</dt>
                <dd>{recoverySummary().baselineAmbiguous}</dd>
              </div>
              <div>
                <dt>unresolved in</dt>
                <dd>{recoverySummary().baselineUnresolved}</dd>
              </div>
              <div>
                <dt>ambiguous left</dt>
                <dd>{recoverySummary().remainingAmbiguous}</dd>
              </div>
              <div>
                <dt>unresolved left</dt>
                <dd>{recoverySummary().remainingUnresolved}</dd>
              </div>
              <div>
                <dt>predicted</dt>
                <dd>{recovery().predictedMissing.length}</dd>
              </div>
              <div>
                <dt>predicted false</dt>
                <dd>{recoverySummary().predictedMissingFalse}</dd>
              </div>
              <div>
                <dt>conflicts</dt>
                <dd>{recovery().metrics.geometryConflicts}</dd>
              </div>
              <div>
                <dt>proof used</dt>
                <dd>{recovery().usedProofPower}</dd>
              </div>
              <div>
                <dt>color rejects</dt>
                <dd>{recovery().metrics.colorRejects}</dd>
              </div>
            </dl>
            <p class={styles.panelNote}>
              For each ambiguous point, the observed-graph support region supplies its anchors. Those anchors fit a
              local homography; the local observed points are then matched partially against the global design by color
              and predicted position, and the homography is refit from every matched pair. Fewer than four support
              anchors, or no match, leaves the point ambiguous. Red rings mark recovered points whose design id
              disagrees with the synthetic truth so local failures stay visible.
            </p>
          </section>
        </aside>
      </div>

      <p class={styles.footnote}>
        Values are evaluated on the synthetic view only. Color detection is treated as perfect; failures come from
        perspective changing the proximity graph. The ICP refiner is truth-blind — design ids are scored against the
        synthetic truth only after the fact.
      </p>
    </div>
  )
}

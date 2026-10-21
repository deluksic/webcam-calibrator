import { createMemo, createSignal } from 'solid-js'

import { ControlSlider } from '@/components/pointLab/ControlSlider'
import { PointLabCanvas } from '@/components/pointLab/PointLabCanvas'
import { extractChainRuns, summarizeChainRuns } from '@/lib/pointIdentity/chainRuns'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndex, evaluateIdentity } from '@/lib/pointIdentity/identity'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, outOfPlaneTiltDeg, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { GraphTuning, MatchOutcome, ProximityGraph, ViewPose } from '@/lib/pointIdentity/types'

import styles from '@/components/pointLab/PointLabView.module.css'

const DEFAULTS = {
  columns: 8,
  pointsPerEdge: 4,
  depth: 5,
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

  const [depth, setDepth] = createSignal(DEFAULTS.depth)
  const [dropoutPercent, setDropoutPercent] = createSignal(DEFAULT_OBSERVATION_MASK.dropoutPercent)
  const [dropoutSeed, setDropoutSeed] = createSignal(DEFAULT_OBSERVATION_MASK.dropoutSeed)
  const [cropLeft, setCropLeft] = createSignal(DEFAULT_OBSERVATION_MASK.cropLeft)
  const [cropRight, setCropRight] = createSignal(DEFAULT_OBSERVATION_MASK.cropRight)
  const [cropTop, setCropTop] = createSignal(DEFAULT_OBSERVATION_MASK.cropTop)
  const [cropBottom, setCropBottom] = createSignal(DEFAULT_OBSERVATION_MASK.cropBottom)

  const [showGraph, setShowGraph] = createSignal(true)
  const [zoomToCursor, setZoomToCursor] = createSignal(true)
  const [selectedIndex, setSelectedIndex] = createSignal<number | undefined>(undefined)
  const [edgeReveal, setEdgeReveal] = createSignal<number | undefined>(undefined)

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
  const observedPoints = createMemo(() =>
    applyObservationMask(projected(), {
      cropLeft: cropLeft(),
      cropRight: cropRight(),
      cropTop: cropTop(),
      cropBottom: cropBottom(),
      dropoutPercent: dropoutPercent(),
      dropoutSeed: dropoutSeed(),
    }),
  )

  const tuning = createMemo<GraphTuning>(() => ({ pointsPerEdge: pointsPerEdge() }))

  const designGraph = createMemo(() => buildProximityGraph(pattern().points, tuning()))
  const designIndex = createMemo(() => createIdentityIndex(designGraph(), colors(), depth()))
  const observedGraph = createMemo(() => buildProximityGraph(observedPoints(), tuning()))
  const evaluation = createMemo(() =>
    evaluateIdentity(designGraph(), observedGraph(), designIndex(), colors(), depth()),
  )

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
    return {
      item,
      color: colors()[index] ?? 0,
      designFingerprint: fingerprint,
      designDegree: degreeById(designGraph(), index),
      observedDegree: degreeById(observedGraph(), index),
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

  const accuracyTone = (): string =>
    summary().falseMatch > 0
      ? (styles.metricDanger ?? '')
      : summary().accuracy >= 0.9
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
          <strong class={styles.metricValue}>{percent(summary().accuracy)}</strong>
          <span class={styles.metricDetail}>
            {summary().correct} of {summary().total} projected points
          </span>
        </article>
        <article class={`${styles.metric} ${summary().falseMatch > 0 ? styles.metricDanger : styles.metricNeutral}`}>
          <span class={styles.metricLabel}>False matches</span>
          <strong class={styles.metricValue}>{summary().falseMatch}</strong>
          <span class={styles.metricDetail}>{percent(summary().falseMatchRate)} of the field · dangerous</span>
        </article>
        <article class={`${styles.metric} ${styles.metricWarning}`}>
          <span class={styles.metricLabel}>Safe failures</span>
          <strong class={styles.metricValue}>{summary().safeFail}</strong>
          <span class={styles.metricDetail}>
            {summary().unresolved} unmatched · {summary().ambiguous} ambiguous
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
              showGraph={showGraph()}
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
            </dl>
            <p class={styles.panelNote}>
              Crop margins are relative to the projected point bounds. Points are removed before graph construction;
              design identity and colors remain unchanged.
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
                <dt>repair</dt>
                <dd>{observedGraph().stats.repairEdges}</dd>
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
            <ControlSlider
              id="lab-depth"
              label="Fingerprint depth"
              value={depth()}
              min={1}
              max={6}
              step={1}
              format={(value) => `${value.toFixed(0)} hops`}
              onInput={setDepth}
            />
            <p class={styles.panelNote}>
              A key is the point's own bit plus angular color rings at each hop. Ring sizes are included, so topology
              drift cannot silently pass as a color match.
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
        </aside>
      </div>

      <p class={styles.footnote}>
        Values are evaluated on the synthetic view only. Color detection is treated as perfect; failures come from
        perspective changing the proximity graph.
      </p>
    </div>
  )
}

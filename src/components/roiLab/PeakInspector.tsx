import { For, Show, createEffect } from 'solid-js'

import { ROI_ACCEPT_SIGMA, type RoiPeakFeatureInspection, type RoiPeakInspection } from '@/lib/roiNeighbourFilter'

import styles from '@/components/roiLab/RoiLabView.module.css'

export type PeakInspectorProps = {
  detail: RoiPeakInspection | undefined
  acceptSigma?: number
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return '—'
  }
  const magnitude = Math.abs(value)
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return value.toExponential(2)
  }
  return value.toFixed(4)
}

function drawFeature(canvas: HTMLCanvasElement, feature: RoiPeakFeatureInspection, acceptSigma: number): void {
  const context = canvas.getContext('2d')
  if (!context) {
    return
  }
  const width = canvas.width
  const height = canvas.height
  const plotTop = 4
  const plotBottom = height - 14
  const plotHeight = Math.max(1, plotBottom - plotTop)
  context.clearRect(0, 0, width, height)
  context.fillStyle = '#050506'
  context.fillRect(0, 0, width, height)

  const allValues = [...feature.neighbourValues, feature.value]
  let minimum = Math.min(...allValues)
  let maximum = Math.max(...allValues)
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
    return
  }
  if (maximum - minimum < 1e-9) {
    const padding = Math.max(Math.abs(maximum) * 0.1, 1e-6)
    minimum -= padding
    maximum += padding
  }
  const span = maximum - minimum
  const xFor = (value: number): number => Math.max(0, Math.min(width, ((value - minimum) / span) * width))

  const binCount = 10
  const counts = Array.from({ length: binCount }, () => 0)
  for (const value of feature.neighbourValues) {
    const bin = Math.min(binCount - 1, Math.max(0, Math.floor(((value - minimum) / span) * binCount)))
    counts[bin] = (counts[bin] ?? 0) + 1
  }
  const maxCount = Math.max(1, ...counts)
  const barWidth = width / binCount
  context.fillStyle = '#8b93a1'
  for (let bin = 0; bin < binCount; bin += 1) {
    const barHeight = ((counts[bin] ?? 0) / maxCount) * plotHeight
    context.fillRect(bin * barWidth + 1, plotBottom - barHeight, Math.max(1, barWidth - 2), barHeight)
  }

  const bandLow = xFor(feature.mean - acceptSigma * feature.std)
  const bandHigh = xFor(feature.mean + acceptSigma * feature.std)
  context.fillStyle = 'rgba(44, 160, 44, 0.16)'
  context.fillRect(Math.min(bandLow, bandHigh), plotTop, Math.abs(bandHigh - bandLow), plotHeight)

  context.strokeStyle = '#9aa0a6'
  context.lineWidth = 1
  for (const value of feature.trimmed) {
    const x = xFor(value)
    context.beginPath()
    context.moveTo(x + 0.5, plotBottom + 2)
    context.lineTo(x + 0.5, plotBottom + 7)
    context.stroke()
  }

  context.strokeStyle = '#60a5fa'
  context.lineWidth = 1.25
  const meanX = xFor(feature.mean)
  context.beginPath()
  context.moveTo(meanX + 0.5, plotTop)
  context.lineTo(meanX + 0.5, plotBottom)
  context.stroke()

  const inside = Number.isFinite(feature.zScore) && Math.abs(feature.zScore) <= acceptSigma
  context.strokeStyle = inside ? '#4ade80' : '#f87171'
  context.lineWidth = 1.75
  const pointX = xFor(feature.value)
  context.beginPath()
  context.moveTo(pointX + 0.5, plotTop)
  context.lineTo(pointX + 0.5, plotBottom)
  context.stroke()

  context.fillStyle = '#8b93a1'
  context.font = '8px monospace'
  context.fillText(formatNumber(minimum), 2, height - 2)
  const maximumLabel = formatNumber(maximum)
  context.fillText(maximumLabel, width - context.measureText(maximumLabel).width - 2, height - 2)
}

export function PeakInspector(props: PeakInspectorProps) {
  const canvases: Array<HTMLCanvasElement | undefined> = []
  const acceptSigma = () => props.acceptSigma ?? ROI_ACCEPT_SIGMA

  createEffect(
    () => ({ detail: props.detail, acceptSigma: acceptSigma() }),
    ({ detail, acceptSigma: sigma }) => {
      if (!detail?.tested) {
        return
      }
      detail.features.forEach((feature, index) => {
        const canvas = canvases[index]
        if (canvas) {
          drawFeature(canvas, feature, sigma)
        }
      })
    },
  )

  const decisionLabel = (detail: RoiPeakInspection): string => {
    if (!detail.tested) {
      return detail.note === 'fewer-than-four-same-polarity'
        ? 'kept: fewer than 4 same-polarity points'
        : 'kept: fewer than 3 neighbours'
    }
    const peak = detail.features.find((feature) => feature.key === 'peak')
    const peakPass = peak !== undefined && Number.isFinite(peak.zScore) && Math.abs(peak.zScore) <= acceptSigma()
    if (!peakPass) {
      return 'dropped: peak outside 4σ'
    }
    const hessian = detail.features.filter((feature) => feature.key !== 'peak')
    const hessianFailed = hessian
      .filter((feature) => !Number.isFinite(feature.zScore) || Math.abs(feature.zScore) > acceptSigma())
      .map((feature) => feature.label)
    const hessianPassing = hessian.length - hessianFailed.length
    if (!detail.kept) {
      return `dropped: ${hessianPassing}/3 Hessian (${hessianFailed.join(', ')} outside 4σ)`
    }
    return hessianFailed.length > 0
      ? `kept (${hessianPassing}/3 Hessian): ${hessianFailed.join(', ')} outside 4σ`
      : 'kept'
  }

  return (
    <div class={styles.peakPanel}>
      <div class={styles.peakHeader}>
        <span>Peak inspector</span>
        <Show when={props.detail}>
          {(detail) => (
            <span>
              #{detail().index} · {detail().point.polarity === 0 ? 'black' : 'white'}
            </span>
          )}
        </Show>
      </div>
      <Show
        when={props.detail}
        fallback={<p class={styles.peakHint}>Click a peak in the Peaks view to inspect its neighbour distributions.</p>}
      >
        {(detail) => (
          <>
            <p class={[styles.peakStatus, !detail().kept && styles.peakStatusFail]}>
              {`(${detail().point.x.toFixed(2)}, ${detail().point.y.toFixed(2)}) · ${
                detail().neighbourCount
              } neighbours · `}
              {decisionLabel(detail())}
            </p>
            <Show when={detail().tested}>
              <For each={detail().features}>
                {(feature, index) => (
                  <div class={styles.peakFeature}>
                    <div class={styles.peakFeatureHeader}>
                      <span class={styles.peakFeatureTitle}>{feature().label}</span>
                      <span class={styles.peakFeatureStats}>
                        {`val ${formatNumber(feature().value)} · μ ${formatNumber(feature().mean)} · σ ${formatNumber(
                          feature().std,
                        )} · z ${Number.isFinite(feature().zScore) ? feature().zScore.toFixed(2) : '—'} · trim ${
                          feature().neighbourValues.length - feature().inlierCount
                        }`}
                      </span>
                    </div>
                    <canvas
                      ref={(element) => {
                        canvases[index()] = element
                      }}
                      class={styles.peakCanvas}
                      width={300}
                      height={86}
                    />
                  </div>
                )}
              </For>
              <div class={styles.peakLegend}>
                <span>
                  <i class={styles.peakSwatch} style={{ background: '#60a5fa' }} /> neighbour mean
                </span>
                <span>
                  <i class={styles.peakSwatch} style={{ background: 'rgba(44, 160, 44, 0.5)' }} /> ±{acceptSigma()}σ
                  band
                </span>
                <span>
                  <i class={styles.peakSwatch} style={{ background: '#4ade80' }} /> point value
                </span>
                <span>
                  <i class={styles.peakSwatch} style={{ background: '#9aa0a6' }} /> trimmed
                </span>
              </div>
            </Show>
          </>
        )}
      </Show>
    </div>
  )
}

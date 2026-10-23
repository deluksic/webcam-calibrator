import { Show } from 'solid-js'

import type { CalibRun } from '@/components/calibration/CalibrationRunContext'
import { DEFAULT_CALIBRATION_TOP_K } from '@/lib/calibrationTopK'
import { formatFixed } from '@/lib/formatFixed'
import type { RoiStepTimings } from '@/gpu/roiLab/roiLabPipeline'
import type { HexReprojectionResult } from '@/lib/hexReprojection'

import styles from '@/components/CalibrationView.module.css'

export type CalibMetricsRow =
  | { solverSummary: string }
  | {
      solverSummary: string
      fxfy: string
      cxyc: string
      fov: string
      ratio: string
      off: string
    }

export type CalibrationAdvancedMetricsProps = {
  run: CalibRun
  validSolveCount: number
  uniqueTargetPointCount: number
  matchedPointCount: number
  calibBlock: CalibMetricsRow
  reproj: HexReprojectionResult | undefined
  timings?: RoiStepTimings
}

export function CalibrationAdvancedMetrics(props: CalibrationAdvancedMetricsProps) {
  return (
    <details class={styles.advanced}>
      <summary>Advanced metrics</summary>
      <div class={styles.stats}>
        <div>
          Solver-ready (Results): {props.validSolveCount} / 4 · Pool {props.run.framePool.length} /{' '}
          {DEFAULT_CALIBRATION_TOP_K} · {props.uniqueTargetPointCount} target points in pool ·{' '}
          {props.matchedPointCount} identified now
        </div>
        <div>
          Stream: {props.run.stats.framesProcessed} frames · {props.run.stats.framesAccepted} snapshots ·{' '}
          {props.run.stats.evictions} Top-K evictions
        </div>
        <div class={styles.statsSection}>Worker solve</div>
        <div>{props.calibBlock.solverSummary}</div>
        {'fxfy' in props.calibBlock ? (
          <>
            <div>fx / fy: {props.calibBlock.fxfy} px</div>
            <div>cx, cy: {props.calibBlock.cxyc}</div>
            <div>H FOV (x): {props.calibBlock.fov}°</div>
            <div>fy/fx: {props.calibBlock.ratio}</div>
            <div>Principal point offset (cx−W/2, cy−H/2): {props.calibBlock.off}</div>
          </>
        ) : null}
        <div class={styles.statsSection}>Live reprojection</div>
        <div>
          RMS {props.reproj ? formatFixed(props.reproj.rms, 3) : '—'} px · points{' '}
          {props.reproj ? props.reproj.pointCount : '—'} · ‖t‖ {props.reproj ? formatFixed(props.reproj.dist, 3) : '—'} ·
          tilt {props.reproj ? formatFixed(props.reproj.tiltDeg, 1) : '—'}°
        </div>
        <Show when={props.timings}>
          {(timings) => (
            <>
              <div class={styles.statsSection}>Pipeline timings</div>
              <div>
                stats {timings().imageStatsMs.toFixed(2)} (age {timings().statsAgeFrames}
                {timings().statsRefreshed ? ', refreshed' : ''}) · pre {timings().preprocessMs.toFixed(2)} · queue{' '}
                {timings().queueWaitMs.toFixed(2)} · inf-run {timings().inferenceRunMs.toFixed(2)} · inf-drain{' '}
                {timings().inferenceDrainMs.toFixed(2)} · copy {timings().cameraCopyMs.toFixed(2)} · mask{' '}
                {timings().maskMs.toFixed(2)} · bg {timings().backgroundMs.toFixed(2)} · comp{' '}
                {timings().componentsEncodeMs.toFixed(2)} · readback {timings().readbackMs.toFixed(2)} ({timings().readbackComponents}{' '}
                comps, {(timings().readbackBytes / 1024).toFixed(1)} KiB{timings().readbackTruncated ? ', truncated' : ''}) · identity{' '}
                {timings().identityMs.toFixed(2)} · frame-drain {timings().frameDrainMs.toFixed(2)} · hist{' '}
                {timings().histogramMs.toFixed(2)} · account {timings().accountedMs.toFixed(2)} / total{' '}
                {timings().totalMs.toFixed(2)} ms
              </div>
              <div>
                GPU pre {timings().gpuMs.preprocess?.toFixed(2) ?? '—'} · mask{' '}
                {timings().gpuMs.mask?.toFixed(2) ?? '—'} · copy {timings().gpuMs.cameraCopy?.toFixed(2) ?? '—'} · bg{' '}
                {(
                  (timings().gpuMs.backgroundWeight ?? 0) +
                  (timings().gpuMs.backgroundHorizontal ?? 0) +
                  (timings().gpuMs.backgroundVertical ?? 0) +
                  (timings().gpuMs.backgroundTexture ?? 0)
                ).toFixed(2)}{' '}
                ms
              </div>
            </>
          )}
        </Show>
      </div>
    </details>
  )
}

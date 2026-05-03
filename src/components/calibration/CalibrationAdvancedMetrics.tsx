import type { CalibRun } from '@/components/calibration/CalibrationRunContext'
import { DEFAULT_CALIBRATION_TOP_K } from '@/lib/calibrationTopK'
import { formatFixed } from '@/lib/formatFixed'

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
  uniqueTagCount: number
  layoutSize: number | undefined
  calibBlock: CalibMetricsRow
  reproj:
    | {
        rms: number
        tagCount: number
        tiltDeg: number
        dist: number
      }
    | undefined
}

export function CalibrationAdvancedMetrics(props: CalibrationAdvancedMetricsProps) {
  const b = props.calibBlock
  const reproj = props.reproj

  return (
    <details class={styles.advanced}>
      <summary>Advanced metrics</summary>
      <div class={styles.stats}>
        <div>
          Solver-ready (Results): {props.validSolveCount} / 4 · Pool {props.run.framePool.length} /{' '}
          {DEFAULT_CALIBRATION_TOP_K} · Layout {props.layoutSize ?? '—'} tags · {props.uniqueTagCount} IDs in pool
        </div>
        <div>
          Stream: {props.run.stats.framesProcessed} frames · {props.run.stats.framesAccepted} snapshots ·{' '}
          {props.run.stats.evictions} Top-K evictions
        </div>
        <div class={styles.statsSection}>Worker solve</div>
        <div>{b.solverSummary}</div>
        {'fxfy' in b ? (
          <>
            <div>
              fx / fy: {b.fxfy} px
            </div>
            <div>cx, cy: {b.cxyc}</div>
            <div>H FOV (x): {b.fov}°</div>
            <div>fy/fx: {b.ratio}</div>
            <div>Principal point offset (cx−W/2, cy−H/2): {b.off}</div>
          </>
        ) : null}
        <div class={styles.statsSection}>Live reprojection</div>
        <div>
          RMS {reproj ? formatFixed(reproj.rms, 3) : '—'} px · tags {reproj ? reproj.tagCount : '—'} · ‖t‖{' '}
          {reproj ? formatFixed(reproj.dist, 3) : '—'} · tilt {reproj ? formatFixed(reproj.tiltDeg, 1) : '—'}°
        </div>
      </div>
    </details>
  )
}

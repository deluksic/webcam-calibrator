import type { CalibRun } from '@/components/calibration/CalibrationRunContext'
import { DEFAULT_CALIBRATION_TOP_K } from '@/lib/calibrationTopK'
import { formatFixed } from '@/lib/formatFixed'
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
      </div>
    </details>
  )
}

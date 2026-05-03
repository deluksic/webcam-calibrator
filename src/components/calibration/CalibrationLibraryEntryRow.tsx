import { Show, createMemo } from 'solid-js'

import {
  CALIBRATION_COMPARE_MAX,
  useCalibrationLibrary,
} from '@/components/calibration/CalibrationLibraryContext'
import type { CalibrationLibraryEntry } from '@/lib/calibrationLibraryTypes'
import { letterForSlot } from '@/lib/calibrationLibraryLetters'

import styles from '@/components/calibration/CalibrationLibraryPanel.module.css'

export type CalibrationLibraryEntryRowProps = {
  entry: CalibrationLibraryEntry
}

export function CalibrationLibraryEntryRow(props: CalibrationLibraryEntryRowProps) {
  const lib = useCalibrationLibrary()

  const slot = createMemo(() => lib.compareIds().indexOf(props.entry.id))
  const isCompared = createMemo(() => slot() >= 0)
  const canCompare = createMemo(() => props.entry.result.kind === 'ok')

  const compareDisabled = createMemo(() => {
    if (!canCompare()) {
      return true
    }
    if (!isCompared() && lib.compareIds().length >= CALIBRATION_COMPARE_MAX) {
      return true
    }
    return false
  })

  const compareTitle = createMemo(() => {
    if (!canCompare()) {
      return 'Only successful calibrations can be compared'
    }
    if (!isCompared() && lib.compareIds().length >= CALIBRATION_COMPARE_MAX) {
      return `At most ${CALIBRATION_COMPARE_MAX} columns (${letterForSlot(0)}–${letterForSlot(CALIBRATION_COMPARE_MAX - 1)})`
    }
    return isCompared() ? 'Remove from comparison' : 'Add to comparison'
  })

  return (
    <li class={[styles.row, props.entry.id === lib.selectedId() && styles.rowSelected]}>
      <div class={styles.rowMain}>
        <button
          type="button"
          class={styles.useBtn}
          title="Show this run in the 3D view"
          onClick={() => lib.setSelectedId(props.entry.id)}
        >
          {props.entry.id === lib.selectedId() ? '● Showing' : 'Show'}
        </button>
        <button
          type="button"
          class={[styles.compareBtn, isCompared() && styles.compareBtnActive]}
          disabled={compareDisabled()}
          title={compareTitle()}
          onClick={() => lib.toggleCompareId(props.entry.id)}
        >
          Compare
        </button>
        <Show when={isCompared()}>
          <span
            class={styles.compareLetterTag}
            data-slot={String(slot())}
            aria-label={`Compare column ${letterForSlot(slot())}`}
          >
            {letterForSlot(slot())}
          </span>
        </Show>
        <span class={styles.time}>{new Date(props.entry.createdAt).toLocaleString()}</span>
        <span class={styles.badge}>{props.entry.result.kind === 'ok' ? 'ok' : props.entry.result.kind}</span>
        <span class={styles.metaHint}>{props.entry.meta.validSolveFrameCount} views</span>
      </div>
      <input
        class={styles.labelInput}
        type="text"
        placeholder="Label (optional)"
        value={props.entry.label}
        aria-label={`Label for calibration ${props.entry.id.slice(0, 8)}`}
        onInput={(e) => lib.rename(props.entry.id, e.currentTarget.value)}
      />
      <button type="button" class={styles.dangerBtn} title="Remove from library" onClick={() => lib.remove(props.entry.id)}>
        Delete
      </button>
    </li>
  )
}

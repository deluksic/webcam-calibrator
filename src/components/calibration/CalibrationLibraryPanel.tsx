import { For, Show, createMemo } from 'solid-js'

import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'
import { CalibrationLibraryEntryRow } from '@/components/calibration/CalibrationLibraryEntryRow'
import { letterForSlot } from '@/lib/calibrationLibraryLetters'
import type { CalibrationOk } from '@/workers/calibration.worker'

import styles from '@/components/calibration/CalibrationLibraryPanel.module.css'

function CompareRows(props: { okList: CalibrationOk[] }) {
  return (
    <table class={styles.compareTable}>
      <thead>
        <tr>
          <th class={styles.th} />
          <For each={props.okList}>
            {(_, i) => (
              <th class={[styles.th, styles.compareTh]}>
                <span class={styles.compareLetterTag} data-slot={String(i())}>
                  {letterForSlot(i())}
                </span>
              </th>
            )}
          </For>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class={styles.tdLabel}>RMS (px)</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().rmsPx.toFixed(3)}</td>}</For>
        </tr>
        <tr>
          <td class={styles.tdLabel}>fx</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().K.fx.toFixed(2)}</td>}</For>
        </tr>
        <tr>
          <td class={styles.tdLabel}>fy</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().K.fy.toFixed(2)}</td>}</For>
        </tr>
        <tr>
          <td class={styles.tdLabel}>cx</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().K.cx.toFixed(2)}</td>}</For>
        </tr>
        <tr>
          <td class={styles.tdLabel}>cy</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().K.cy.toFixed(2)}</td>}</For>
        </tr>
        <tr>
          <td class={styles.tdLabel}>Views</td>
          <For each={props.okList}>{(r) => <td class={styles.tdNum}>{r().extrinsics.length}</td>}</For>
        </tr>
      </tbody>
    </table>
  )
}

export function CalibrationLibraryPanel() {
  const lib = useCalibrationLibrary()

  const compareOkList = createMemo((): CalibrationOk[] => {
    const ids = lib.compareIds()
    const entries = lib.entries()
    const out: CalibrationOk[] = []
    for (const id of ids) {
      const e = entries.find((x) => x.id === id)
      if (e?.result.kind === 'ok') {
        out.push(e.result)
      }
    }
    return out
  })

  return (
    <section class={styles.section} aria-label="Saved calibrations">
      <h3 class={styles.heading}>Saved calibrations</h3>
      <p class={styles.help}>
        Saved runs stay here after <strong>Reset</strong> on Calibrate. <strong>Show</strong> drives the 3D view.{' '}
        <strong>Compare</strong> adds or removes a row from the table below (letters match columns). Use{' '}
        <strong>Save to library</strong> on this page after you finish a calibration run on <strong>Calibrate</strong>.
      </p>
      <Show
        when={lib.entries().length > 0}
        fallback={
          <p class={styles.empty}>
            No saved runs yet. Use <strong>Save to library</strong> above once you have a calibration ready from{' '}
            <strong>Calibrate</strong>.
          </p>
        }
      >
        <ul class={styles.list}>
          <For each={lib.entries()}>{(entry) => <CalibrationLibraryEntryRow entry={entry()} />}</For>
        </ul>
      </Show>

      <Show when={compareOkList().length > 0}>
        <div class={styles.compareBoard}>
          <CompareRows okList={compareOkList()} />
          <div class={styles.compareFooter}>
            <button type="button" class={styles.clearCompareBtn} onClick={() => lib.clearCompare()}>
              Clear comparison
            </button>
          </div>
        </div>
      </Show>
    </section>
  )
}

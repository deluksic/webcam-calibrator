import { For, Show, createMemo } from 'solid-js'

import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'

import styles from '@/components/calibration/CalibrationLibraryPanel.module.css'

import type { CalibrationOk } from '@/workers/calibration.worker'

function compareRows(a: CalibrationOk, b: CalibrationOk) {
  return (
    <table class={styles.compareTable}>
      <thead>
        <tr>
          <th class={styles.th} />
          <th class={styles.th}>A</th>
          <th class={styles.th}>B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class={styles.tdLabel}>RMS (px)</td>
          <td class={styles.tdNum}>{a.rmsPx.toFixed(3)}</td>
          <td class={styles.tdNum}>{b.rmsPx.toFixed(3)}</td>
        </tr>
        <tr>
          <td class={styles.tdLabel}>fx</td>
          <td class={styles.tdNum}>{a.K.fx.toFixed(2)}</td>
          <td class={styles.tdNum}>{b.K.fx.toFixed(2)}</td>
        </tr>
        <tr>
          <td class={styles.tdLabel}>fy</td>
          <td class={styles.tdNum}>{a.K.fy.toFixed(2)}</td>
          <td class={styles.tdNum}>{b.K.fy.toFixed(2)}</td>
        </tr>
        <tr>
          <td class={styles.tdLabel}>cx</td>
          <td class={styles.tdNum}>{a.K.cx.toFixed(2)}</td>
          <td class={styles.tdNum}>{b.K.cx.toFixed(2)}</td>
        </tr>
        <tr>
          <td class={styles.tdLabel}>cy</td>
          <td class={styles.tdNum}>{a.K.cy.toFixed(2)}</td>
          <td class={styles.tdNum}>{b.K.cy.toFixed(2)}</td>
        </tr>
        <tr>
          <td class={styles.tdLabel}>Views</td>
          <td class={styles.tdNum}>{a.extrinsics.length}</td>
          <td class={styles.tdNum}>{b.extrinsics.length}</td>
        </tr>
      </tbody>
    </table>
  )
}

export function CalibrationLibraryPanel() {
  const lib = useCalibrationLibrary()

  const comparePair = createMemo((): { a: CalibrationOk; b: CalibrationOk } | undefined => {
    const aId = lib.selectedId()
    const bId = lib.compareId()
    if (!aId || !bId || aId === bId) {
      return undefined
    }
    const entries = lib.entries()
    const ea = entries.find((e) => e.id === aId)
    const eb = entries.find((e) => e.id === bId)
    if (!ea || !eb || ea.result.kind !== 'ok' || eb.result.kind !== 'ok') {
      return undefined
    }
    return { a: ea.result, b: eb.result }
  })

  const compareTable = createMemo(() => {
    const p = comparePair()
    if (!p) {
      return null
    }
    return compareRows(p.a, p.b)
  })

  return (
    <section class={styles.section} aria-label="Saved calibrations">
      <h3 class={styles.heading}>Saved calibrations</h3>
      <p class={styles.help}>
        Saved runs stay here after <strong>Reset</strong> on Calibrate. Pick one to drive the 3D view, or compare two
        entries below. Use <strong>Save to library</strong> on this page after you finish a calibration run on{' '}
        <strong>Calibrate</strong>.
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
          <For each={lib.entries()}>
            {(entry) => (
              <li
                class={[
                  styles.row,
                  entry().id === lib.selectedId() && styles.rowSelected,
                ]}
              >
                <div class={styles.rowMain}>
                  <button
                    type="button"
                    class={styles.useBtn}
                    title="Show this run in the 3D view"
                    onClick={() => lib.setSelectedId(entry().id)}
                  >
                    {entry().id === lib.selectedId() ? '● Showing' : 'Show'}
                  </button>
                  <span class={styles.time}>{new Date(entry().createdAt).toLocaleString()}</span>
                  <span class={styles.badge}>{entry().result.kind === 'ok' ? 'ok' : entry().result.kind}</span>
                  <span class={styles.metaHint}>{entry().meta.validSolveFrameCount} views</span>
                </div>
                <input
                  class={styles.labelInput}
                  type="text"
                  placeholder="Label (optional)"
                  value={entry().label}
                  aria-label={`Label for calibration ${entry().id.slice(0, 8)}`}
                  onInput={(e) => lib.rename(entry().id, e.currentTarget.value)}
                />
                <button
                  type="button"
                  class={styles.dangerBtn}
                  title="Remove from library"
                  onClick={() => lib.remove(entry().id)}
                >
                  Delete
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>

      <Show when={lib.entries().length > 0}>
        <div class={styles.compareBar}>
          <label class={styles.compareLabel}>
            Compare with
            <select
              class={styles.compareSelect}
              value={lib.compareId() ?? ''}
              onChange={(e) => lib.setCompareId(e.currentTarget.value || undefined)}
            >
              <option value="">—</option>
              <For each={lib.entries()}>
                {(entry) => (
                  <option value={entry().id}>
                    {(entry().label || new Date(entry().createdAt).toLocaleString()) + ` (${entry().id.slice(0, 8)})`}
                  </option>
                )}
              </For>
            </select>
          </label>
        </div>
        {compareTable()}
      </Show>
    </section>
  )
}

import { For } from 'solid-js'

import styles from '@/components/calibration/CalibrateGuidancePanel.module.css'

export type GuidanceBand = 'progress' | 'needs-attention'

type Props = {
  band: () => GuidanceBand
  lines: () => string[]
}

export function CalibrateGuidancePanel(props: Props) {
  return (
    <section
      class={[styles.wrap, props.band() === 'needs-attention' && styles.wrapAttention]}
      aria-live="polite"
    >
      <For each={props.lines()}>
        {(line) => <p class={styles.line}>{line()}</p>}
      </For>
    </section>
  )
}

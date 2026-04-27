import { For } from 'solid-js'

import styles from './Home.module.css'

const SUB_PAGES = [
  {
    id: 'calibration',
    label: 'Calibration',
    description: 'Set up your target and begin taking snapshots.',
  },
  {
    id: 'results',
    label: 'Results',
    description: 'Review calibration results and test your camera.',
  },
  {
    id: 'about',
    label: 'About',
    description: 'Learn how to use AprilTag calibration for AR/robotics.',
  },
]

export function Home() {
  return (
    <div class={styles.root}>
      <h1 class={styles.title}>AprilTag Camera Calibration</h1>
      <p class={styles.description}>
        Calibrate cameras for AR/robotics using AprilTag fiducials. Optimized for solid, static targets.
      </p>

      <div class={styles.setup}>
        <h2 class={styles.setupTitle}>What you need</h2>
        <p class={styles.setupText}>
          Two devices: a camera (phone or webcam) and a target display (phone, tablet, or computer).
          They can be the same device or different ones.
        </p>
      </div>

      <div class={styles.flow}>
        <h2 class={styles.flowTitle}>How to calibrate</h2>
        <ol class={styles.flowSteps}>
          <li>Open this site on both devices</li>
          <li>Point your camera at the target display</li>
          <li>Take snapshots until the results page appears</li>
          <li>Test and export your calibrated camera</li>
        </ol>
      </div>

      <div class={styles.links}>
        <div class={styles.linksTitle}>Get started</div>
        <For each={SUB_PAGES}>
          {(item) => (
            <a href={`#${item().id}`} class={styles.link}>
              <div class={styles.linkLabel}>{item().label}</div>
              <div class={styles.linkDesc}>{item().description}</div>
            </a>
          )}
        </For>
      </div>
    </div>
  )
}

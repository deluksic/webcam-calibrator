import styles from './Home.module.css'

export function Home() {
  return (
    <div class={styles.root}>
      <h1 class={styles.title}>Camera Calibration</h1>
      <p class={styles.description}>
        Calibrate cameras for AR/robotics. Optimized for solid, static targets.
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

      <div class={styles.buttons}>
        <a href="/target" class={styles.buttonPrimary}>
          Show Target
        </a>
        <a href="/calibrate" class={styles.buttonPrimary}>
          Calibrate
        </a>
      </div>
    </div>
  )
}

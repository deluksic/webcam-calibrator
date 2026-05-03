import { A } from '@solidjs/router'
import { Show, createSignal } from 'solid-js'

import styles from './Home.module.css'

import { loadUserGuidancePrefs, patchUserGuidancePrefs } from '@/lib/userGuidancePrefs'

export function Home() {
  const [hideIntro, setHideIntro] = createSignal(loadUserGuidancePrefs().hideHomeIntro === true)

  return (
    <div class={styles.root}>
      <h1 class={styles.title}>Camera Calibration</h1>
      <p class={styles.description}>
        Calibrate cameras for AR and robotics using a static AprilTag board. Use one device for the target and another
        for the camera, or the same machine for both.
      </p>

      <Show when={hideIntro()}>
        <div class={styles.reopenRow}>
          <button
            type="button"
            class={styles.dismissBtn}
            onClick={() => {
              patchUserGuidancePrefs({ hideHomeIntro: false })
              setHideIntro(false)
            }}
          >
            Show intro again
          </button>
        </div>
      </Show>

      <Show when={!hideIntro()}>
        <div class={styles.callout}>
          <h2 class={styles.calloutTitle}>Two roles</h2>
          <ul class={styles.calloutList}>
            <li>
              <strong>Display</strong> — open <A href="/target">Target</A> on a separate screen or print it out.
            </li>
            <li>
              <strong>Camera</strong> — open <A href="/calibrate">Calibrate</A> on the capturing device.
            </li>
          </ul>
          <button
            type="button"
            class={styles.dismissBtn}
            onClick={() => {
              patchUserGuidancePrefs({ hideHomeIntro: true })
              setHideIntro(true)
            }}
          >
            Got it, hide this
          </button>
        </div>
      </Show>

      <div class={styles.bigActions}>
        <A href="/target" class={styles.bigAction}>
          <span class={styles.bigActionTitle}>Show the target</span>
          <span class={styles.bigActionDesc}>Separate screen or projector—the tag grid the camera will see.</span>
        </A>
        <A href="/calibrate" class={styles.bigAction}>
          <span class={styles.bigActionTitle}>Use the camera</span>
          <span class={styles.bigActionDesc}>Capture from several viewpoints; open Results when calibration is ready.</span>
        </A>
      </div>
    </div>
  )
}

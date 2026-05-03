import styles from '@/components/calibration/CalibrateFocusOverlay.module.css'

/** Advisory 75%×75% framing guide; does not gate capture (plan §4D). */
export function CalibrateFocusOverlay() {
  return <div class={styles.overlay} aria-hidden="true" />
}

/** Live Calibrate grid overlay: session indices for negative (custom) tag ids. */
export type CustomTagOverlaySession = {
  collectionRunning: boolean
  sessionIndexByCustomTagId: Map<number, number>
}

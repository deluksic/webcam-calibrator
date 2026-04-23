/**
 * Manages a requestVideoFrameCallback loop for a video element.
 *
 * Fires `onPrime` once when the first video frame is presented, then calls
 * `onFrame` on every subsequent frame. Errors thrown by `onFrame` are caught
 * and logged so they don't break the loop.
 *
 * Returns a `dispose()` function; call it (e.g. from a SolidJS `onCleanup`)
 * to cancel any pending callbacks and stop the loop.
 */
export function createFrameLoop(options: { video: HTMLVideoElement; onFrame: () => void }): { dispose: () => void } {
  const { video, onFrame } = options
  let rafHandle = 0
  let disposed = false

  const loop = () => {
    if (disposed) {
      return
    }
    try {
      onFrame()
    } catch (e) {
      console.error('[createFrameLoop] onFrame error:', e)
    }
    rafHandle = video.requestVideoFrameCallback(loop)
  }

  rafHandle = video.requestVideoFrameCallback(loop)

  return {
    dispose() {
      disposed = true
      if (rafHandle) {
        video.cancelVideoFrameCallback(rafHandle)
        rafHandle = 0
      }
    },
  }
}

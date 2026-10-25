export type CanvasRectLike = {
  left: number
  top: number
  width: number
  height: number
}

export type PointerFramePoint = {
  /** Pointer position in frame-pixel units, before zoom/pan. */
  canvasX: number
  canvasY: number
  /** Zoom/pan-inverted frame coordinate used for peak hit-testing. */
  frameX: number
  frameY: number
}

/**
 * Convert a pointer to the frame coordinate drawn by the ROI canvas.
 *
 * The canvas uses `object-fit: contain`, so its CSS box can be wider or taller
 * than the frame aspect. The unused bars must be removed before normalizing;
 * otherwise clicks near an edge select a peak offset toward the center.
 */
export function pointerToFrame(
  clientX: number,
  clientY: number,
  rect: CanvasRectLike,
  frameWidth: number,
  frameHeight: number,
  zoom: number,
  panX: number,
  panY: number,
): PointerFramePoint | undefined {
  if (rect.width <= 0 || rect.height <= 0 || frameWidth <= 0 || frameHeight <= 0) {
    return undefined
  }
  const frameAspect = frameWidth / frameHeight
  const boxAspect = rect.width / rect.height
  const contentWidth = boxAspect > frameAspect ? rect.height * frameAspect : rect.width
  const contentHeight = boxAspect > frameAspect ? rect.height : rect.width / frameAspect
  const offsetX = (rect.width - contentWidth) / 2
  const offsetY = (rect.height - contentHeight) / 2
  const u = (clientX - rect.left - offsetX) / contentWidth
  const v = (clientY - rect.top - offsetY) / contentHeight
  if (u < 0 || u > 1 || v < 0 || v > 1) {
    return undefined
  }
  const canvasX = u * frameWidth
  const canvasY = v * frameHeight
  return {
    canvasX,
    canvasY,
    frameX: (canvasX - panX) / zoom,
    frameY: (canvasY - panY) / zoom,
  }
}

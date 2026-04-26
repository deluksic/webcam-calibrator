/**
 * Color attachment passed to TypeGPU `.withColorAttachment(...)`.
 * Typed loosely so `configureContext` targets and texture views both work.
 */
export type RenderColorAttachment = {
  view: unknown
  clearValue?: GPUColor
  loadOp?: GPULoadOp
  storeOp?: GPUStoreOp
}

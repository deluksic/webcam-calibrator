import type { TgpuRoot } from 'typegpu'

/** Clear color + depth when there is nothing to visualize (MSAA swap like lines-combinations). */
export function clearResultsAttachments(
  root: TgpuRoot,
  gpuContext: GPUCanvasContext,
  msaaColorView: GPUTextureView,
  depthView: GPUTextureView,
): void {
  const enc = root.device.createCommandEncoder({ label: 'results clear' })
  const resolveTarget = gpuContext.getCurrentTexture().createView()
  const rp = enc.beginRenderPass({
    label: 'results clear pass',
    colorAttachments: [
      {
        view: msaaColorView,
        resolveTarget,
        clearValue: [0.1, 0.1, 0.2, 1],
        loadOp: 'clear',
        storeOp: 'discard',
      },
    ],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'discard',
    },
  })
  rp.end()
  root.device.queue.submit([enc.finish()])
}

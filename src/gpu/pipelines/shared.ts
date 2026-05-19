/** Premultiplied alpha blend state — shared by all overlay/compositing pipelines. */
export const PREMULTIPLIED_ALPHA_BLEND: GPUBlendState = {
  color: {
    operation: 'add',
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
  },
  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
}

/**
 * The ROI segmentation net (blob_lab TinyRoiNet, output_stride=2) rebuilt on
 * runntime core, for the spike. Mirrors the PyTorch module tree field-for-field
 * so runntime's loader can bind the checkpoint and fold the BatchNorms into
 * their preceding convs.
 *
 * Shape notes:
 *  - channel skips are plain `cat`, the natural form of the decoder;
 *  - `halfPixel` bilinear resize matches torch's align_corners=False;
 *  - runs f16 end to end (conv2d is f16-only in runntime core).
 */
import {
  bufferSource,
  cat,
  evalValues,
  fromSafetensors,
  maxPool2d,
  nn,
  resizeBilinear2d,
  sigmoid,
  toChw,
  toHwc4,
  type Value,
} from '@runntime/core'
import type { TgpuRoot } from 'typegpu'

const { BatchNorm2d, Conv2d, Module } = nn
const { performance } = globalThis

/** conv → bn → relu → conv → bn → relu, declared in order so BatchNorm2d can
 *  infer its conv from the sibling registered immediately before it. The relu
 *  is fused into the conv epilogue and runs after the folded affine. */
class ConvBlock extends Module {
  readonly conv1: InstanceType<typeof Conv2d>
  readonly bn1: InstanceType<typeof BatchNorm2d>
  readonly conv2: InstanceType<typeof Conv2d>
  readonly bn2: InstanceType<typeof BatchNorm2d>

  constructor(inChannels: number, outChannels: number) {
    super()
    this.conv1 = new Conv2d(inChannels, outChannels, {
      kernelSize: 3,
      padding: 1,
      bias: true,
      activation: 'relu',
    })
    this.bn1 = new BatchNorm2d(outChannels)
    this.conv2 = new Conv2d(outChannels, outChannels, {
      kernelSize: 3,
      padding: 1,
      bias: true,
      activation: 'relu',
    })
    this.bn2 = new BatchNorm2d(outChannels)
  }

  override forward(x: Value): Value {
    const y = this.conv1.forward(x)
    const z = this.bn1.forward(y)
    const w = this.conv2.forward(z)
    return this.bn2.forward(w)
  }
}

const pool2 = (x: Value): Value => maxPool2d(x, { kernelSize: 2, stride: 2, padding: 0 })

export class RoiNet extends Module {
  readonly enc1: ConvBlock
  readonly enc2: ConvBlock
  readonly enc3: ConvBlock
  readonly enc4: ConvBlock
  readonly bottleneck: ConvBlock
  readonly dec3: ConvBlock
  readonly dec2: ConvBlock
  readonly mask_head: InstanceType<typeof Conv2d>

  constructor() {
    super()
    this.enc1 = new ConvBlock(1, 16)
    this.enc2 = new ConvBlock(16, 24)
    this.enc3 = new ConvBlock(24, 24)
    this.enc4 = new ConvBlock(24, 24)
    this.bottleneck = new ConvBlock(24, 24)
    this.dec3 = new ConvBlock(48, 24)
    this.dec2 = new ConvBlock(48, 24)
    this.mask_head = new Conv2d(24, 1, { kernelSize: 1 })
  }

  /** x: [1, H, W] f32 CHW → mask probabilities [H/2, W/2] f32. */
  override forward(x: Value): Value {
    // toHwc4 once at the input; the whole trunk stays in hwc4 f16.
    const e1 = this.enc1.forward(toHwc4(x))
    const e2 = this.enc2.forward(pool2(e1))
    const e3 = this.enc3.forward(pool2(e2))
    const e4 = this.enc4.forward(pool2(e3))
    const b = this.bottleneck.forward(pool2(e4))

    const [, h3, w3] = e3.shape.dims as [number, number, number]
    const d3 = this.dec3.forward(
      cat([resizeBilinear2d(b, { outH: h3, outW: w3, mode: 'halfPixel' }), e3], 0),
    )

    const [, h2, w2] = e2.shape.dims as [number, number, number]
    const d2 = this.dec2.forward(
      cat([resizeBilinear2d(d3, { outH: h2, outW: w2, mode: 'halfPixel' }), e2], 0),
    )

    // sigmoid refuses hwc4 (f(0) !== 0 would corrupt the padded lanes), so the
    // activation runs after leaving hwc4. Only 1 channel, so the hop is cheap.
    return sigmoid(toChw(this.mask_head.forward(d2), 'f32'))
  }
}

export type LoadedRoiNet = {
  model: RoiNet
  /** Fresh forward pass, fully encoded on the GPU. */
  evalMask: (x: Value) => Value
  loadMs: number
  tensorCount: number
  /** Trainable scalar count, summed from the module tree's declared shapes. */
  paramCount: number
}

/** Build the net, convert to f16 and bind the exported safetensors. Accepts a
 *  URL or already-fetched bytes, so the caller can report download progress. */
export async function loadRoiNet(
  root: TgpuRoot,
  source: string | ArrayBuffer,
): Promise<LoadedRoiNet> {
  const model = new RoiNet()
  // Declared shapes are the raw checkpoint sizes (hwc4 padding is applied to the
  // uploaded blob, not to `elems`), so this matches PyTorch's parameter count.
  let paramCount = 0
  for (const [, parameter] of model.namedParameters()) {
    paramCount += parameter.shape.elems
  }
  model.half(root)
  const started = performance.now()
  const stateDict = await fromSafetensors(
    typeof source === 'string' ? source : bufferSource(source),
  )
  await model.loadStateDict(stateDict, { root })
  const loadMs = performance.now() - started

  return {
    model,
    evalMask: (x: Value) => {
      const mask = model.forward(x)
      evalValues([mask])
      return mask
    },
    loadMs,
    tensorCount: [...stateDict.tensors.keys()].length,
    paramCount,
  }
}

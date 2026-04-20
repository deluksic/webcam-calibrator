import { refineSubpixelHomographyV1, type SubpixelRefineInput } from '@/lib/subpixelRefinement'
import type { SubpixelRefineFn } from '@/tests/subpixel/types'

export const refineIdentity: SubpixelRefineFn = (input: SubpixelRefineInput) => new Float32Array(input.homography8)

export const refineHomographyV1: SubpixelRefineFn = refineSubpixelHomographyV1

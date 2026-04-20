import type { SubpixelRefineInput } from '@/lib/subpixelRefinement'

export type { SubpixelRefineInput }

export type SubpixelRefineFn = (input: SubpixelRefineInput) => Float32Array

import type { Mat3 } from '@/lib/geometry'
import type { SubpixelRefineInput } from '@/lib/subpixelRefinement'

export type { SubpixelRefineInput }

export type SubpixelRefineFn = (input: SubpixelRefineInput) => Mat3

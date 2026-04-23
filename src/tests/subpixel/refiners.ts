import { refineSubpixelHomographyV1, type SubpixelRefineInput } from '@/lib/subpixelRefinement'
import type { SubpixelRefineFn } from '@/tests/subpixel/types'

export const refineIdentity: SubpixelRefineFn = ({ homography }) => homography

export const refineHomographyV1: SubpixelRefineFn = refineSubpixelHomographyV1

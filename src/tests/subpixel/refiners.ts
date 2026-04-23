import { refineSubpixelHomographyV1 } from '@/lib/subpixelRefinement'
import type { SubpixelRefineFn } from '@/tests/subpixel/types'

const refineIdentity: SubpixelRefineFn = ({ homography }) => homography

export { refineSubpixelHomographyV1, refineIdentity }

/** Interior module classification from Sobel vote accumulation (6×6 AprilTag data cells). */

export const TAG_MODULE_CELL = {
  black: 0,
  white: 1,
  weak: -1,
  tie: -2,
} as const

export type TagModuleCell = (typeof TAG_MODULE_CELL)[keyof typeof TAG_MODULE_CELL]

export type TagPattern = TagModuleCell[]

export function isTagModuleWeak(v: TagModuleCell): boolean {
  return v === TAG_MODULE_CELL.weak
}

export function isTagModuleTie(v: TagModuleCell): boolean {
  return v === TAG_MODULE_CELL.tie
}

/** True if any cell is an ambiguous tie — quad must be discarded without decode. */
export function patternHasAnyTie(pattern: TagPattern): boolean {
  return pattern.some(isTagModuleTie)
}

/** Weak (-1) or tie (-2) cells — live id overlay is hidden (no red “?”). */
export function patternHasWeakOrTie(pattern: TagPattern | undefined): boolean {
  if (!pattern || pattern.length !== 36) {
    return false
  }
  for (let i = 0; i < pattern.length; i++) {
    const v = pattern[i]!
    if (v === TAG_MODULE_CELL.weak || v === TAG_MODULE_CELL.tie) {
      return true
    }
  }
  return false
}

/** True if every interior cell is confident black or white (no weak, no tie). */
export function patternIsFullyBinary(pattern: TagPattern): boolean {
  for (let i = 0; i < pattern.length; i++) {
    const v = pattern[i]!
    if (v !== TAG_MODULE_CELL.black && v !== TAG_MODULE_CELL.white) {
      return false
    }
  }
  return pattern.length === 36
}

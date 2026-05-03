const STORAGE_KEY = 'webcam-calibrator.workflow.v1'

export type UserGuidancePrefs = {
  /** Bump in code to invalidate stored dismiss flags when copy changes materially. */
  prefsVersion?: number
  hideHomeIntro?: boolean
  hideTargetTips?: boolean
}

export const USER_GUIDANCE_PREFS_VERSION = 1

function readRaw(): unknown {
  if (typeof globalThis.localStorage === 'undefined') {
    return undefined
  }
  try {
    const s = globalThis.localStorage.getItem(STORAGE_KEY)
    if (!s) {
      return undefined
    }
    return JSON.parse(s) as unknown
  } catch {
    return undefined
  }
}

export function loadUserGuidancePrefs(): UserGuidancePrefs {
  const raw = readRaw()
  if (!raw || typeof raw !== 'object' || raw === null) {
    return {}
  }
  const o = raw as Record<string, unknown>
  const storedVersion = typeof o.prefsVersion === 'number' ? o.prefsVersion : 0
  if (storedVersion !== USER_GUIDANCE_PREFS_VERSION) {
    return {}
  }
  return {
    prefsVersion: USER_GUIDANCE_PREFS_VERSION,
    hideHomeIntro: o.hideHomeIntro === true,
    hideTargetTips: o.hideTargetTips === true,
  }
}

export function saveUserGuidancePrefs(next: UserGuidancePrefs): void {
  if (typeof globalThis.localStorage === 'undefined') {
    return
  }
  try {
    globalThis.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...next, prefsVersion: USER_GUIDANCE_PREFS_VERSION }),
    )
  } catch {
    // ignore quota / private mode
  }
}

export function patchUserGuidancePrefs(patch: Partial<UserGuidancePrefs>): UserGuidancePrefs {
  const merged = { ...loadUserGuidancePrefs(), ...patch, prefsVersion: USER_GUIDANCE_PREFS_VERSION }
  saveUserGuidancePrefs(merged)
  return merged
}

/** Editable hex-target configuration shared by the Target and Calibrate pages. */
export type HexTargetConfig = {
  columns: number
  rows: number
  pointsPerEdge: number
  markerScale: number
  colorSeed: number
  identityDepth: number
  widthPx?: number
  heightPx?: number
}

export const DEFAULT_TARGET_CONFIG: HexTargetConfig = {
  columns: 8,
  rows: 4,
  pointsPerEdge: 4,
  markerScale: 0.4,
  colorSeed: 1,
  identityDepth: 12,
}

/** Minimum identified markers before Start/snapshot are accepted. */
export const TARGET_MIN_CAPTURE_POINTS = 12

function clampInteger(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback
  }
  return Math.min(max, Math.max(min, Math.round(value)))
}

function clampNumber(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback
  }
  return Math.min(max, Math.max(min, value))
}

function optionalPositiveInteger(value: number | undefined): number | undefined {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return undefined
  }
  return Math.round(value)
}

export function normalizeHexTargetConfig(input: Partial<HexTargetConfig>): HexTargetConfig {
  return {
    columns: clampInteger(input.columns ?? DEFAULT_TARGET_CONFIG.columns, 2, 24, DEFAULT_TARGET_CONFIG.columns),
    rows: clampInteger(input.rows ?? DEFAULT_TARGET_CONFIG.rows, 2, 24, DEFAULT_TARGET_CONFIG.rows),
    pointsPerEdge: clampInteger(
      input.pointsPerEdge ?? DEFAULT_TARGET_CONFIG.pointsPerEdge,
      1,
      8,
      DEFAULT_TARGET_CONFIG.pointsPerEdge,
    ),
    markerScale: clampNumber(
      input.markerScale ?? DEFAULT_TARGET_CONFIG.markerScale,
      0.2,
      0.7,
      DEFAULT_TARGET_CONFIG.markerScale,
    ),
    colorSeed: clampInteger(input.colorSeed ?? DEFAULT_TARGET_CONFIG.colorSeed, 0, 999_999, DEFAULT_TARGET_CONFIG.colorSeed),
    identityDepth: clampInteger(
      input.identityDepth ?? DEFAULT_TARGET_CONFIG.identityDepth,
      1,
      12,
      DEFAULT_TARGET_CONFIG.identityDepth,
    ),
    widthPx: optionalPositiveInteger(input.widthPx),
    heightPx: optionalPositiveInteger(input.heightPx),
  }
}

function numberParam(params: URLSearchParams, key: string): number | undefined {
  const raw = params.get(key)
  if (raw === null || raw.trim() === '') {
    return undefined
  }
  const value = Number(raw)
  return Number.isFinite(value) ? value : undefined
}

export function hexTargetConfigFromSearchParams(params: URLSearchParams): HexTargetConfig {
  return normalizeHexTargetConfig({
    columns: numberParam(params, 'columns'),
    rows: numberParam(params, 'rows'),
    pointsPerEdge: numberParam(params, 'points'),
    markerScale: numberParam(params, 'scale'),
    colorSeed: numberParam(params, 'seed'),
    identityDepth: numberParam(params, 'depth'),
    widthPx: numberParam(params, 'w'),
    heightPx: numberParam(params, 'h'),
  })
}

export function hexTargetConfigToSearchParams(
  config: HexTargetConfig,
  resolution?: { widthPx: number; heightPx: number },
): URLSearchParams {
  const normalized = normalizeHexTargetConfig({
    ...config,
    widthPx: resolution?.widthPx ?? config.widthPx,
    heightPx: resolution?.heightPx ?? config.heightPx,
  })
  const params = new URLSearchParams({
    columns: String(normalized.columns),
    rows: String(normalized.rows),
    points: String(normalized.pointsPerEdge),
    seed: String(normalized.colorSeed),
    scale: String(normalized.markerScale),
    depth: String(normalized.identityDepth),
  })
  if (normalized.widthPx) {
    params.set('w', String(normalized.widthPx))
  }
  if (normalized.heightPx) {
    params.set('h', String(normalized.heightPx))
  }
  return params
}

/** Read config from either the hash-router query (`#/calibrate?...`) or the normal query. */
export function readHexTargetConfigFromLocation(): HexTargetConfig {
  if (typeof globalThis.location === 'undefined') {
    return DEFAULT_TARGET_CONFIG
  }
  const hash = globalThis.location.hash
  const queryIndex = hash.indexOf('?')
  const query = queryIndex >= 0 ? hash.slice(queryIndex + 1) : globalThis.location.search
  return hexTargetConfigFromSearchParams(new URLSearchParams(query))
}

/** Absolute hash-router URL for the Calibrate page, suitable for a QR code. */
export function buildCalibrationUrl(
  config: HexTargetConfig,
  resolution?: { widthPx: number; heightPx: number },
): string {
  const params = hexTargetConfigToSearchParams(config, resolution)
  if (typeof globalThis.location === 'undefined') {
    return `#/calibrate?${params.toString()}`
  }
  return `${globalThis.location.origin}${globalThis.location.pathname}#/calibrate?${params.toString()}`
}

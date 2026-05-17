/** Shared thresholds for oriented histogram peaks, assignment, and line fit (gradient-profile path). */

export const ORIENT_HIST_BINS = 64
export const MAX_EDGES_PER_LABEL = 4

export const ORIENT_PEAK_MIN_COUNT = 6
/** Min circular bin distance between accepted peaks (~45° at 64 bins). */
export const MIN_PEAK_BIN_SEPARATION = 8
/** Prefer peaks within this bin distance when tie-breaking (see assignPeakEdgeId). */
export const ORIENT_ASSIGN_MAX_BIN_DIST = 6
/** Min dot(ĝ, peakDir) to assign; bin distance alone is not used as a hard gate. */
export const ORIENT_ASSIGN_MIN_ALIGN = 0.5

export const LINE_MIN_PEAK_HIST_COUNT = 12
export const LINE_MIN_SLOT_COUNT = 12
/** Perpendicular distance (px) for inlier gate and refine pass. */
export const LINE_INLIER_DIST_PX = 4.0
/** Min fraction of assigned slot pixels that pass the inlier gate after refit. */
export const LINE_MIN_INLIER_RATIO = 0.8
/** Min inlier pixels required before TLS refit (after coarse fit + gate). */
export const LINE_MIN_REFINE_INLIERS = 12
/** Fraction trimmed from each end of along-edge extent (corner pull). */
export const LINE_EXTENT_TRIM_FRAC = 0.08

export const TLS_REF_COS_MAX_ANGLE = 0.85

/** Min valid lines to register a label as a quad for visualization. */
export const MIN_QUAD_VALID_EDGES = 2
/** Min TLS inliers for a side to count toward quad registration. */
export const MIN_QUAD_EDGE_INLIERS = 12

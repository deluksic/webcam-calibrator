/** Shared thresholds for oriented histogram peaks, assignment, and line fit (gradient-profile path). */

export const ORIENT_HIST_BINS = 64
export const MAX_EDGES_PER_LABEL = 4

export const ORIENT_PEAK_MIN_COUNT = 6
/** Min circular bin distance between accepted peaks (~45° at 64 bins). */
export const MIN_PEAK_BIN_SEPARATION = 8
/** Prefer peaks within this bin distance when tie-breaking (see assignPeakEdgeId). */
export const ORIENT_ASSIGN_MAX_BIN_DIST = 8
/** Min dot(ĝ, peakDir) to assign; bin distance alone is not used as a hard gate. */
export const ORIENT_ASSIGN_MIN_ALIGN = 0.4

/** Must match `ORIENT_PEAK_MIN_COUNT` — peaks are accepted at 6 but accum used to require 12. */
export const LINE_MIN_PEAK_HIST_COUNT = ORIENT_PEAK_MIN_COUNT
/** Min edge pixels assigned to a peak slot before coarse/TLS fit. */
export const LINE_MIN_SLOT_COUNT = 6
/** Perpendicular distance (px) for inlier gate and refine pass. */
export const LINE_INLIER_DIST_PX = 2.5
/** Min fraction of assigned slot pixels that pass the inlier gate after refit. */
export const LINE_MIN_INLIER_RATIO = 0.7
/** Min inlier pixels required before TLS refit (after coarse fit + gate). */
export const LINE_MIN_REFINE_INLIERS = 12
/** Fraction trimmed from each end of along-edge extent (corner pull). */
export const LINE_EXTENT_TRIM_FRAC = 0.05

export const TLS_REF_COS_MAX_ANGLE = 0.85

/** Valid fitted lines required to register a label as a quad (AprilTag = 4). */
export const MIN_QUAD_VALID_EDGES = MAX_EDGES_PER_LABEL

/** Line intersection: |det| below this → parallel (geometric only, not a shape prior). */
export const LINE_INTERSECT_DET_EPS = 1e-10
/** Homography DLT pivot floor — looser than CPU `1e-10` so f32 Gaussian elimination does not spuriously fail. */
export const HOMOGRAPHY_PIVOT_EPS = 1e-5
/** Min edge length (px) for corner quad degeneracy check. */
export const QUAD_MIN_EDGE_PX = 2
/** Signed-area floor scale: reject if |area| < this × scale² (scale = max corner |coord|). */
export const QUAD_MIN_SIGNED_AREA_REL = 1e-4

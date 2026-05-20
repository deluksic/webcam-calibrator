/** Shared tag-decode constants for the GPU pipeline (`tagDecodePipeline.ts`). */

export const TAG_DECODE_HIST_BINS = 32
export const TAG_DECODE_PEAK_GAP_FRAC = 0.1
export const TAG_DECODE_MAX_DICT_ERROR = 3
/** Max weak (-1) cells for GPU `2^u` wildcard unroll per codeword thread. */
export const TAG_DECODE_MAX_WEAK_WILDCARD = 4

/** Linear bin separation for white peak. */
export const TAG_DECODE_MIN_PEAK_BIN_SEP = 8

/** Minimum peak luma span (in bins) to accept thresholds. */
export const TAG_DECODE_MIN_PEAK_LUMA_BINS = 3

export const DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE = 0.02

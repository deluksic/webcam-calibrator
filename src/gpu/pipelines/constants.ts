// Camera pipeline constants and shared utilities

export const WR = 0.2126;
export const WG = 0.7152;
export const WB = 0.0722;
export const HISTOGRAM_BINS = 256;
export const HIST_WIDTH = 512;
export const HIST_HEIGHT = 120;

/** Compute adaptive threshold from histogram */
export function computeThreshold(histogramData: number[], percentile: number = 0.85): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0);
  const targetCount = totalPixels * percentile;

  let cumulative = 0;
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i];
    if (cumulative >= targetCount) {
      return (i + 1) / 512.0;
    }
  }
  return 0.5;
}

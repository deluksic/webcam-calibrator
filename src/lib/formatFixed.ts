/** Fixed-decimal display; non-finite values render as an em dash. */
export function formatFixed(n: number | undefined, decimals: number): string {
  if (n === undefined || !Number.isFinite(n)) {
    return '—'
  }
  return n.toFixed(decimals)
}

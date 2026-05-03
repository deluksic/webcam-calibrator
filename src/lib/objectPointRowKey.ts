/** > max corner index (3). Used to pack `(tagId, cornerId)` for calibration object-point rows. */
export const OBJECT_POINT_ROW_KEY_STRIDE = 4

/**
 * Non-negative `tagId`: `tagId * stride + cornerId`.
 * Negative `tagId` (custom): `tagId * stride - cornerId` — avoids broken JS `%` on negative packed keys.
 */
export function objectPointRowKey(tagId: number, cornerId: number): number {
  return tagId * OBJECT_POINT_ROW_KEY_STRIDE + (tagId < 0 ? -cornerId : cornerId)
}

export function unpackObjectPointRowKey(rowKey: number): { tagId: number; cornerId: number } {
  if (rowKey >= 0) {
    const tagId = Math.floor(rowKey / OBJECT_POINT_ROW_KEY_STRIDE)
    const cornerId = rowKey - tagId * OBJECT_POINT_ROW_KEY_STRIDE
    return { tagId, cornerId }
  }
  // `rowKey === 4*tagId - cornerId` with `cornerId ∈ [0,3]` ⇒ `cornerId ≡ -rowKey (mod 4)`.
  // Euclidean residue of `rowKey` mod 4 (JS `%` is not Euclidean on negatives):
  const r =
    ((rowKey % OBJECT_POINT_ROW_KEY_STRIDE) + OBJECT_POINT_ROW_KEY_STRIDE) % OBJECT_POINT_ROW_KEY_STRIDE
  const cornerId = (OBJECT_POINT_ROW_KEY_STRIDE - r) % OBJECT_POINT_ROW_KEY_STRIDE
  const tagId = (rowKey + cornerId) / OBJECT_POINT_ROW_KEY_STRIDE
  return { tagId, cornerId }
}

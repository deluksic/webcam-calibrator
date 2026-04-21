export function cameraDeviceScore(d: MediaDeviceInfo): number {
  const label = d.label.toLowerCase()
  let score = 0
  if (label.includes('back') || label.includes('rear')) {
    score += 100
  }
  if (label.includes('wide')) {
    score += 50
  }
  if (label.includes('ultra')) {
    score += 30
  }
  if (label.includes('tele')) {
    score -= 20
  }
  if (label.includes('front') || label.includes('user')) {
    score -= 100
  }
  return score
}

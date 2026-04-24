const mlmatrix = require('ml-matrix');

// Python's true homography
const H_true = [[2, 0, 10], [0, 1.5, 5], [0, 0, 1]]
const plane_pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
const img_pts = plane_pts.map(([x, y]) => {
  const w = x * H_true[2][0] + y * H_true[2][1] + H_true[2][2]
  return {
    x: (x * H_true[0][0] + y * H_true[0][1] + H_true[0][2]) / w,
    y: (x * H_true[1][0] + y * H_true[1][1] + H_true[1][2]) / w
  }
})

// Hartley normalization
function hartley2(pts) {
  const n = pts.length
  console.log('n =', n)
  const cx = pts.reduce((sum, p) => sum + p.x, 0) / n
  const cy = pts.reduce((sum, p) => sum + p.y, 0) / n
  console.log('center:', cx, cy)
  let r2 = 0
  for (let i = 0; i < n; i++) {
    const dx = pts[i].x - cx
    const dy = pts[i].y - cy
    r2 += dx * dx + dy * dy
  }
  const r = Math.sqrt((r2 / n) + 1e-18)
  const s = Math.sqrt(2) / r
  console.log('r2=', r2, 'r=', r, 's=', s)
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1]
}

console.log('Plane pts:', plane_pts)
const tP = hartley2(plane_pts)
console.log('tP:', tP)

console.log('\nImage pts:', img_pts)
const tI = hartley2(img_pts)
console.log('tI:', tI)

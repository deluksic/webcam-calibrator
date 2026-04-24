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

console.log('True H:', H_true);
console.log('Image pts:', img_pts);

// Hartley normalization
function hartley2(pts) {
  const n = pts.length
  const cx = pts.reduce((sum, p) => sum + p.x, 0) / n
  const cy = pts.reduce((sum, p) => sum + p.y, 0) / n
  let r2 = 0
  for (let i = 0; i < n; i++) {
    const dx = pts[i].x - cx
    const dy = pts[i].y - cy
    r2 += dx * dx + dy * dy
  }
  const r = Math.sqrt((r2 / n) + 1e-18)
  const s = Math.sqrt(2) / r
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1]
}

const tP = hartley2(plane_pts)
const tI = hartley2(img_pts)

console.log('tP:', tP);
console.log('tI:', tI);

const rows = 8
const a = new Float32Array(rows * 9)

for (let r = 0; r < 4; r++) {
  const pl = plane_pts[r]
  const im = img_pts[r]

  const Xn0 = tP[0] * pl.x + tP[1] * pl.y + tP[2]
  const Xn1 = tP[3] * pl.x + tP[4] * pl.y + tP[5]
  const Un0 = tI[0] * im.x + tI[1] * im.y + tI[2]
  const Un1 = tI[3] * im.x + tI[4] * im.y + tI[5]

  const row0 = r * 18
  a[row0 + 0] = Xn0
  a[row0 + 1] = Xn1
  a[row0 + 2] = 1
  a[row0 + 6] = -Xn0 * Un0
  a[row0 + 7] = -Xn1 * Un0
  a[row0 + 8] = -Un0

  const row1 = r * 18 + 9
  a[row1 + 3] = Xn0
  a[row1 + 4] = Xn1
  a[row1 + 5] = 1
  a[row1 + 6] = -Xn0 * Un1
  a[row1 + 7] = -Xn1 * Un1
  a[row1 + 8] = -Un1

  console.log(`\nPoint ${r}:`)
  console.log(`  Xn=[${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}], Un=[${Un0.toFixed(6)}, ${Un1.toFixed(6)}]`)
  console.log(`  Row ${row0}: [${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}, 1, 0, 0, 0, ${(-Xn0*Un0).toFixed(6)}, ${(-Xn1*Un0).toFixed(6)}, ${(-Un0).toFixed(6)}]`)
  console.log(`  Row ${row1}: [0, 0, 0, ${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}, 1, ${(-Xn0*Un1).toFixed(6)}, ${(-Xn1*Un1).toFixed(6)}, ${(-Un1).toFixed(6)}]`)
}

console.log('\n\nMatrix a (8x9, row-major):')
console.log('Row 0:', Array.from({length: 9}, (_, j) => a[j].toFixed(10)))
console.log('Row 1:', Array.from({length: 9}, (_, j) => a[9 + j].toFixed(10)))

// SVD - autoTranspose to fix
const M = mlmatrix.Matrix.from1DArray(rows, 9, a)
const svd = new mlmatrix.SingularValueDecomposition(M, {
  autoTranspose: true,
  computeLeftSingularVectors: false,
  computeRightSingularVectors: true
})

const s = svd.diagonal
const V = svd.rightSingularVectors

console.log('\nSingular values:', s.map(v => v.toFixed(10)))
console.log('Last singular value:', s[8].toFixed(10))

// Last column of V (index 8)
console.log('\nLast column of V (index 8):')
for (let i = 0; i < 9; i++) {
  console.log(`  V[${i}, 8] = ${V.get(i, 8).toFixed(10)}`)
}

const h = new Float32Array(9)
for (let i = 0; i < 9; i++) {
  h[i] = V.get(i, 8)
}

console.log('\nh (last column):', Array.from(h).map(v => v.toFixed(10)))
const h22 = h[8]
console.log('h22 (h[8]):', h22)

const h_norm = h.map(v => v / h22)
console.log('h (normalized, h22=1):', Array.from(h_norm).map(v => v.toFixed(10)))

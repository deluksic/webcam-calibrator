const { Matrix, SingularValueDecomposition } = require('ml-matrix');

// Python's true homography
const H_true = [[2, 0, 10], [0, 1.5, 5], [0, 0, 1]]
const plane_pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
const img_pts = H_true.map((row, i) => {
  return { x: row[0] * 0 + row[1] * 0 + row[2], y: 0 }
})

// Hartley normalization (matches Python)
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

console.log('tP:', tP)
console.log('tI:', tI)
console.log('tInvI:', [tI[0]/tI[8], tI[1]/tI[8], tI[2]/tI[8], tI[3]/tI[8], tI[4]/tI[8], tI[5]/tI[8], tI[6]/tI[8], tI[7]/tI[8], 1])
console.log()

// Compute normalized coordinates like in Python
const Xn = []
const Un = []
for (let i = 0; i < 4; i++) {
  Xn.push(
    tP[0] * plane_pts[i][0] + tP[1] * plane_pts[i][1] + tP[2],
    tP[3] * plane_pts[i][0] + tP[4] * plane_pts[i][1] + tP[5],
    1
  )
  Un.push(
    tI[0] * img_pts[i].x + tI[1] * img_pts[i].y + tI[2],
    tI[3] * img_pts[i].x + tI[4] * img_pts[i].y + tI[5],
    1
  )
}

console.log('Xn (3x4):')
for (let i = 0; i < 4; i++) {
  console.log(`  [${Xn[i*3].toFixed(6)}, ${Xn[i*3+1].toFixed(6)}, 1]`)
}
console.log()
console.log('Un (3x4):')
for (let i = 0; i < 4; i++) {
  console.log(`  [${Un[i*3].toFixed(6)}, ${Un[i*3+1].toFixed(6)}, 1]`)
}
console.log()

// Build constraint matrix A (8x9) - rows
const rows = 8
const a = new Float64Array(rows * 9)

console.log('Building A (rows as Python):')
for (let r = 0; r < 4; r++) {
  const row0 = r * 2
  const row1 = r * 2 + 1
  const Xn0 = Xn[r * 3], Xn1 = Xn[r * 3 + 1]
  const Un0 = Un[r * 3], Un1 = Un[r * 3 + 1]

  console.log(`Correspondence ${r}: Xn=[${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}], Un=[${Un0.toFixed(6)}, ${Un1.toFixed(6)}]`)
  console.log(`  Row ${row0}: [${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}, 1, 0, 0, 0, ${(-Xn0*Un0).toFixed(6)}, ${(-Xn1*Un0).toFixed(6)}, ${(-Un0).toFixed(6)}]`)
  console.log(`  Row ${row1}: [0, 0, 0, ${Xn0.toFixed(6)}, ${Xn1.toFixed(6)}, 1, ${(-Xn0*Un1).toFixed(6)}, ${(-Xn1*Un1).toFixed(6)}, ${(-Un1).toFixed(6)}]`)

  // Row 0
  a[row0 * 9 + 0] = Xn0
  a[row0 * 9 + 1] = Xn1
  a[row0 * 9 + 2] = 1
  a[row0 * 9 + 6] = -Xn0 * Un0
  a[row0 * 9 + 7] = -Xn1 * Un0
  a[row0 * 9 + 8] = -Un0

  // Row 1
  a[row1 * 9 + 3] = Xn0
  a[row1 * 9 + 4] = Xn1
  a[row1 * 9 + 5] = 1
  a[row1 * 9 + 6] = -Xn0 * Un1
  a[row1 * 9 + 7] = -Xn1 * Un1
  a[row1 * 9 + 8] = -Un1
}

console.log()

// SVD
const M = Matrix.from1DArray(rows, 9, a)
const svd = new SingularValueDecomposition(M, {
  autoTranspose: true,
  computeLeftSingularVectors: false,
  computeRightSingularVectors: true,
})

const V = svd.rightSingularVectors
const singularValues = svd.diagonal

console.log('Singular values:', Array.from(singularValues).map(v => v.toFixed(10)))
console.log('Last singular value:', singularValues[8].toFixed(10))
console.log()

console.log('V matrix (3x9):')
for (let i = 0; i < 9; i++) {
  console.log(`  Row ${i}: [${Array.from({length: 9}, (_, j) => V.get(i, j).toFixed(6)).join(', ')}]`)
}
console.log()

// Get null vector as last column
const h = new Float64Array(9)
for (let i = 0; i < 9; i++) {
  h[i] = V.get(i, 8)
}

console.log('h (from ml-matrix):', Array.from(h).map(v => v.toFixed(6)))
const h_norm = h.map(v => v/h[8])
console.log('h (normalized, h8=1):', Array.from(h_norm).map(v => v.toFixed(6)))
console.log()

// Verify A * h = 0
console.log('Verifying A * h = 0:')
let maxResidual = 0
for (let i = 0; i < 8; i++) {
  let res = 0
  for (let j = 0; j < 9; j++) {
    res += a[i * 9 + j] * h[j]
  }
  console.log(`  Row ${i}: ${res.toExponential(6)}`)
  maxResidual = Math.max(maxResidual, Math.abs(res))
}
console.log(`  Max residual: ${maxResidual.toExponential(6)}`)
console.log()

// Convert h to hN (3x3)
const hN = [
  h[0]/h[8], h[1]/h[8], h[2]/h[8],
  h[3]/h[8], h[4]/h[8], h[5]/h[8],
  h[6]/h[8], h[7]/h[8], 1
]

console.log('hN (normalized coordinates homography):')
console.log(hN.map(v => v.toFixed(6)))
console.log()

// Denormalize: H = tInvI @ hN @ tP
function matMul3(a, b) {
  const result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j]
      }
    }
  }
  return result
}

const tInvI = [tI[0]/tI[8], tI[1]/tI[8], tI[2]/tI[8], tI[3]/tI[8], tI[4]/tI[8], tI[5]/tI[8], tI[6]/tI[8], tI[7]/tI[8], 1]

// hN @ tP
const h_tP = matMul3(hN, tP)
console.log('hN @ tP:')
console.log(h_tP.map(v => v.toFixed(6)))
console.log()

// tInvI @ (hN @ tP)
const H = matMul3(tInvI, h_tP)
console.log('tInvI @ (hN @ tP):')
console.log(H.map(v => v.toFixed(6)))
console.log()

// Normalize by H[8]
const H_normalized = H.map(v => v / H[8])
console.log('H (normalized, H8=1):')
console.log(H_normalized.map(v => v.toFixed(6)))
console.log()

// Compare with Python
console.log('Python result:')
console.log([[2, 0, 10], [0, 1.5, 5], [0, 0, 1]].flat().map(v => v.toFixed(6)))
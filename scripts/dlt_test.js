const { Matrix, SingularValueDecomposition } = require('ml-matrix');

// Replicate Python exactly
const H_true = [
  [2, 0, 10],
  [0, 1.5, 5],
  [0, 0, 1]
];
const plane_pts = [[0, 0], [1, 0], [0, 1], [1, 1]];
const img_pts = plane_pts.map(([x, y], i) => {
  const row = H_true[i];
  return { x: row[0] * x + row[1] * y + row[2], y: 0 };
});

console.log('img_pts:', img_pts);
console.log();

// Hartley normalization
function hartley2(pts) {
  const n = pts.length;
  const cx = pts.reduce((sum, p) => sum + p.x, 0) / n;
  const cy = pts.reduce((sum, p) => sum + p.y, 0) / n;
  let r2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = pts[i].x - cx;
    const dy = pts[i].y - cy;
    r2 += dx * dx + dy * dy;
  }
  const r = Math.sqrt((r2 / n) + 1e-18);
  const s = Math.sqrt(2) / r;
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}

const tP = hartley2(plane_pts);
const tI = hartley2(img_pts);

console.log('tP:', tP);
console.log('tI:', tI);
console.log();

// Compute normalized coordinates
const Xn = [];
const Un = [];
for (let i = 0; i < 4; i++) {
  Xn.push(
    tP[0] * plane_pts[i][0] + tP[1] * plane_pts[i][1],
    tP[3] * plane_pts[i][0] + tP[4] * plane_pts[i][1],
    1
  );
  Un.push(
    tI[0] * img_pts[i].x + tI[1] * img_pts[i].y,
    tI[3] * img_pts[i].x + tI[4] * img_pts[i].y,
    1
  );
}

console.log('Xn (8 elems):', Xn);
console.log('Un (8 elems):', Un);
console.log();

// Build A as Python does
const rows = 8;
const a = new Float64Array(rows * 9);

console.log('Building A...');
for (let r = 0; r < 4; r++) {
  const Xn0 = Xn[r * 2], Xn1 = Xn[r * 2 + 1];
  const Un0 = Un[r * 2], Un1 = Un[r * 2 + 1];

  console.log('r:', r, 'Xn:', [Xn0, Xn1], 'Un:', [Un0, Un1]);

  // Row 0: [Xn, Yn, 1, 0, 0, 0, -Xn*un, -Yn*un, -un]
  a[r * 9 + 0] = Xn0;
  a[r * 9 + 1] = Xn1;
  a[r * 9 + 2] = 1;
  a[r * 9 + 6] = -Xn0 * Un0;
  a[r * 9 + 7] = -Xn1 * Un0;
  a[r * 9 + 8] = -Un0;

  // Row 1: [0, 0, 0, Xn, Yn, 1, -Xn*vn, -Yn*vn, -vn]
  a[r * 9 + 3] = Xn0;
  a[r * 9 + 4] = Xn1;
  a[r * 9 + 5] = 1;
  a[r * 9 + 6] = -Xn0 * Un1;
  a[r * 9 + 7] = -Xn1 * Un1;
  a[r * 9 + 8] = -Un1;
}

console.log('\nConstraint matrix (rows as Python):');
for (let i = 0; i < 8; i++) {
  console.log('  Row ' + i + ':', Array.from({length: 9}, (_, j) => a[i * 9 + j]).map(v => v.toFixed(6)).join(', '));
}

// SVD
const M = Matrix.from1DArray(rows, 9, a);
const svd = new SingularValueDecomposition(M, {
  autoTranspose: true,
  computeLeftSingularVectors: false,
  computeRightSingularVectors: true,
});

const V = svd.rightSingularVectors;
const h = new Float64Array(9);
for (let i = 0; i < 9; i++) {
  h[i] = V.get(i, 8);
}

console.log('\nh from ml-matrix:', Array.from(h).map(v => v.toFixed(6)));

// Denormalize
const tInvI = [
  tI[0] / tI[8], tI[1] / tI[8], tI[2] / tI[8],
  tI[3] / tI[8], tI[4] / tI[8], tI[5] / tI[8],
  tI[6] / tI[8], tI[7] / tI[8], 1
];

function matMul3(a, b) {
  const result = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
      }
    }
  }
  return result;
}

const H = matMul3(tInvI, matMul3(h, tP));

console.log('\nH computed:', Array.from(H).map(v => v.toFixed(6)));
console.log('H true:', H_true.flat());
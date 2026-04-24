# Plan: Web-Based Camera Calibration Matching OpenCV Exactly

## Context

The current webcam-calibrator implementation uses Zhang's closed-form calibration method but lacks:
- **Distortion modeling** (radial + tangential)
- **Iterative optimization** (Levenberg-Marquardt refinement)
- **Full rational distortion model** support
- **Exact numerical matching** to OpenCV's output

The goal is to achieve bit-for-bit matching with OpenCV's `calibrateCamera` function.

## Libraries

### ml-matrix (already installed)
Provides:
- `SingularValueDecomposition` - Full SVD decomposition
- `Matrix` - Matrix operations (multiply, inverse, transpose)
- `determinant` - Matrix determinant
- `inverse` - Matrix inverse
- `solve` - Linear system solver

### mathjs (if needed)
- Complex number support
- Symbolic math for derivative verification

## Phase 1: Mathematical Foundation

### 1.1 Matrix Operations (Use ml-matrix)
- Import from `ml-matrix`
- Convert between row-major arrays and Matrix objects
- Key operations:
  - Matrix multiplication (3x3, 3xN)
  - Matrix inversion (3x3)
  - Determinant
  - Matrix transpose
  - SVD for null vector extraction

### 1.2 Vec3 Operations (Implement locally)
- Cross product
- Dot product
- Norm (L2)
- Normalization
- Element-wise operations

### 1.3 SVD Null Vector
Use ml-matrix's `SingularValueDecomposition`:
```typescript
import { SingularValueDecomposition } from 'ml-matrix';

function nullVector(A: number[], rows: number, cols: number): number[] {
  const M = new Matrix(rows, cols, A);
  const svd = new SingularValueDecomposition(M);
  // Last column of V contains null vector
  const V = svd.rightSingularVectors;
  return V.to1DArray().slice(-cols); // Last column
}
```

### 1.4 Tests
- A * A^{-1} = I (within tolerance)
- SVD on identity → singular values = 1
- SVD on rotation → singular values = 1

## Phase 2: Homography (DLT)

### 2.1 Homography DLT
**File**: `src/lib/dltHomography.ts` (existing, review)

Use ml-matrix SVD for null vector extraction:
1. Build 2n × 9 matrix A from correspondences
2. SVD decomposition of A
3. Extract null vector (last column of V)
4. Normalize so h33 = 1

**Critical detail**: OpenCV uses **normalized DLT** (precondition input points)

### 2.2 Homography Decomposition
**File**: `src/lib/homographyDecompose.ts` (new)

Extract R, t from homography:
1. H = K^{-1} H_raw
2. Normalize: λ = 2 / (||h1|| + ||h2||)
3. r1 = λh1, r2 = λh2, t = λh3
4. r3 = r1 × r2
5. Orthonormalize via Gram-Schmidt

### 2.3 Tests
- 4 points → known homography → verify reconstruction
- R must satisfy R^T R = I
- det(R) = +1

## Phase 3: Zhang's Method (Closed-Form)

### 3.1 V Matrix Construction
**File**: `src/lib/zhangCalibration.ts` (existing, review)

For homography H_i = [h1 h2 h3]:
```
v_ij = [h_i1*h_j1, h_i1*h_j2 + h_i2*h_j1, h_i2*h_j2,
        h_i3*h_j1 + h_i1*h_j3, h_i3*h_j2 + h_i2*h_j3, h_i3*h_j3]^T

Constraints:
v12^T b - v11^T b = 0
v11^T b - v22^T b = 0
```

### 3.2 Extract K from B
```
cy = (B12*B13 - B11*B23) / (B11*B22 - B12²)
λ = B33 - (B13² + cy*(B12*B13 - B11*B23)) / B11
fx = √(λ / B11)
fy = √(λ * B11 / (B11*B22 - B12²))
cx = (-B13 * fx²) / λ
```

### 3.3 Extrinsics from Homography
1. M = K^{-1} H
2. λ = 2 / (||m0|| + ||m1||)
3. r1 = λm0, r2 = λm1, t = λm2
4. r3 = r1 × r2
5. Orthonormalize via Gram-Schmidt

## Phase 4: Distortion Model (NEW)

### 4.1 Distortion Function
**File**: `src/lib/distortion.ts` (new)

Brown-Conrady rational model:
```
r² = x² + y²
x' = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)
     + (2*p1*x*y + p2*(r² + 2*x²))
y' = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)
     + (p1*(r² + 2*y²) + 2*p2*x*y)
```

Parameters: [k1, k2, p1, p2, k3, k4, k5, k6]

### 4.2 Inverse Distortion (Newton-Raphson)
Iterative inversion for undistortion:
1. Initial guess: x0 = distorted point
2. Iterations until convergence
3. Converge when ||delta|| < tolerance

### 4.3 projectPoints with Distortion
**File**: `src/lib/projectPoints.ts` (new)

1. Apply extrinsics: Xc = R*X + t
2. Normalize: x = Xc/Zc, y = Yc/Zc
3. Apply distortion: x', y' = distort(x, y, k)
4. Project: u = fx*x' + cx, v = fy*y' + cy

## Phase 5: Iterative Optimization

### 5.1 Levenberg-Marquardt Solver
**File**: `src/lib/levmarq.ts` (new)

Optimize parameters:
- fx, fy, cx, cy (4)
- k1, k2, p1, p2, k3, k4, k5, k6 (8)
- Per-frame R, t (6 each)

Algorithm:
```
while ||gradient|| > tau && iter < max_iter:
  Compute J (Jacobian)
  Update: (J^T J + λ*diag(J^T J)) * delta = -J^T r
  Try step, adjust λ on success/failure
```

### 5.2 Jacobian Computation
**File**: `src/lib/jacobian.ts` (new)

Analytical derivatives for:
- ∂error/∂fx, ∂error/∂fy, ∂error/∂cx, ∂error/∂cy
- ∂error/∂k1-∂k6, ∂error/∂p1, ∂error/∂p2
- ∂error/∂R, ∂error/∂t

### 5.3 Initial Distortion Estimation
Estimate initial k values from reprojection errors before LM.

## Phase 6: Integration

### 6.1 Complete calibrateCamera Function
**File**: `src/lib/calibrateCamera.ts` (new)

1. Compute homographies for all views
2. Solve for initial K via Zhang
3. Compute initial extrinsics R, t per view
4. Estimate initial distortion k1-k6
5. Run Levenberg-Marquardt optimization
6. Return K, distCoeffs, rvecs, tvecs

### 6.2 Rodrigues Conversion
**File**: `src/lib/rodrigues.ts` (new)

- Forward: R → rvec (rotation vector)
- Inverse: rvec → R

### 6.3 Test Datasets
Create `test/fixtures/opencv-comparison/` with OpenCV reference data.

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `src/lib/distortion.ts` | Distortion model + inverse |
| `src/lib/projectPoints.ts` | Pinhole + distortion projection |
| `src/lib/levmarq.ts` | Levenberg-Marquardt optimizer |
| `src/lib/jacobian.ts` | Analytical Jacobian |
| `src/lib/rodrigues.ts` | Rodrigues conversion |
| `src/lib/calibrateCamera.ts` | Main calibration entry point |

### Modified Files
| File | Changes |
|------|---------|
| `src/lib/zhangCalibration.ts` | Add distortion estimation |
| `src/lib/dltHomography.ts` | Use ml-matrix SVD |
| `src/lib/cameraModel.ts` | Add distCoeffs to types |

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
   ↓          ↓         ↓         ↓         ↓         ↓
ml-matrix DLT     Zhang    Distort   LM      Integrate
Verify   Verify    Verify    Model    Optim   & Test
```

Each phase ends with tests that match OpenCV before proceeding.

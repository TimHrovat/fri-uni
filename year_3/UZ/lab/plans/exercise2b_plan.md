# Exercise 2b: Fundamental Matrix Estimation - Implementation Plan

## Overview
Implement the eight-point algorithm to estimate the fundamental matrix between two camera views.

## Mathematical Background

### Fundamental Matrix
The fundamental matrix `F` (3×3) encodes the epipolar geometry between two views:
- For corresponding points `x` in image 1 and `x'` in image 2:
  - `x'^T F x = 0`
- Points are in homogeneous coordinates: `[u, v, 1]^T` where `u` is column, `v` is row

### Eight-Point Algorithm Steps

1. **Build Matrix A** (from equation 5 in PDF):
   For each correspondence `(u, v)` ↔ `(u', v')`, add row:
   ```
   [u*u', u'*v, u', u*v', v*v', v', u, v, 1]
   ```

2. **Solve for F using SVD**:
   - `A = U D V^T`
   - Solution: last column of `V` (eigenvector with smallest eigenvalue)
   - Reshape to 3×3 matrix `F`

3. **Enforce Rank-2 Constraint**:
   - `F = U D V^T`
   - Set smallest singular value (D[2,2]) to 0
   - Reconstruct: `F = U D V^T`

4. **Point Normalization** (for numerical stability):
   - Normalize points before computing F
   - Transform F back to original coordinate system
   - Formula: `F = T2^T F_normalized T1`

## Implementation Structure

### Function: `fundamental_matrix(pts1, pts2)`

**Input:**
- `pts1`: Nx2 array of points from image 1 `[u, v]`
- `pts2`: Nx2 array of points from image 2 `[u', v']`

**Output:**
- `F`: 3×3 fundamental matrix

**Algorithm:**
```python
def fundamental_matrix(pts1, pts2):
    # 1. Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # 2. Build matrix A
    N = len(pts1_norm)
    A = zeros((N, 9))
    for i in range(N):
        u, v = pts1_norm[i]
        u_p, v_p = pts2_norm[i]
        A[i] = [u*u_p, u_p*v, u_p, u*v_p, v*v_p, v_p, u, v, 1]

    # 3. Solve using SVD
    U, D, VT = svd(A)
    F = VT[-1].reshape(3, 3)  # last row of VT = last column of V

    # 4. Enforce rank-2 constraint
    U, D, VT = svd(F)
    D[2] = 0  # set smallest singular value to 0
    F = U @ diag(D) @ VT

    # 5. Denormalize
    F = T2.T @ F @ T1

    return F
```

### Function: `exercise2b()`

**Tasks:**
1. Load house data (images 7 and 8)
2. Load keypoints from .corners files
3. Load matches using `read_matches()`
4. Compute fundamental matrix
5. Visualize:
   - Original images with keypoints
   - Epipolar lines for each point
6. Compare with reference `F_7_8.txt`

## Data Loading

### House Dataset Structure
```
data/house/
  images/
    house.007.png
    house.008.png
  2D/
    house.007.corners  # keypoints for image 7
    house.008.corners  # keypoints for image 8
    house.nview-corners  # matches across all views
  F_7_8.txt  # reference fundamental matrix
```

### Loading Points
```python
# Load keypoints
pts1_all = np.loadtxt('data/house/2D/house.007.corners')
pts2_all = np.loadtxt('data/house/2D/house.008.corners')

# Load matches (helper function in a5_utils.py)
matches = read_matches('data/house/2D/house.nview-corners', 7, 8)
# Returns Nx2 array where each row is [idx_img7, idx_img8]

# Extract matching points
pts1 = pts1_all[matches[:, 0]]
pts2 = pts2_all[matches[:, 1]]
```

## Visualization

### Display Requirements
1. **Two subplots side-by-side**:
   - Left: Image 7 with keypoints and epipolar lines
   - Right: Image 8 with keypoints and epipolar lines

2. **Epipolar Lines**:
   - For each point in image 1, draw its epipolar line in image 2: `l' = F @ x`
   - For each point in image 2, draw its epipolar line in image 1: `l = F^T @ x'`
   - Use `draw_epiline()` from a5_utils.py

3. **Style**:
   - Different colors for different point correspondences
   - Points should lie on their corresponding epipolar lines

## Testing & Verification

### Test 1: Visual Inspection
- Epipolar lines should pass through corresponding points
- Lines should not converge inside the image

### Test 2: Numerical Comparison
```python
F_ref = np.loadtxt('data/house/F_7_8.txt')
F_computed = fundamental_matrix(pts1, pts2)

# Normalize both matrices
F_ref = F_ref / np.linalg.norm(F_ref)
F_computed = F_computed / np.linalg.norm(F_computed)

# Compute difference
diff = np.abs(F_ref - F_computed)
print(f"Max difference: {np.max(diff)}")
print(f"Mean difference: {np.mean(diff)}")
```

### Test 3: Epipolar Constraint
For all matching points:
```python
error = x2^T @ F @ x1  # should be close to 0
```

## Implementation Checklist

- [ ] Implement `fundamental_matrix()` function
- [ ] Test normalization workflow
- [ ] Implement `exercise2b()` function
- [ ] Load house.007 and house.008 images
- [ ] Load and display keypoints
- [ ] Load matches using `read_matches()`
- [ ] Compute fundamental matrix
- [ ] Draw epipolar lines for all points
- [ ] Compare with reference matrix
- [ ] Add visualizations with proper titles
- [ ] Test on sample points

## Expected Output

### Figure Layout
```
Exercise 2b: Fundamental Matrix Estimation
┌─────────────────────┬─────────────────────┐
│  house.007.png      │  house.008.png      │
│  + keypoints (dots) │  + keypoints (dots) │
│  + epipolar lines   │  + epipolar lines   │
└─────────────────────┴─────────────────────┘
```

### Console Output
```
Exercise 2b: Fundamental Matrix Estimation
Loaded 16 matching points
Computed Fundamental Matrix:
[[-1.095e-08,  2.522e-07,  4.712e-05],
 [ 1.117e-06, -9.612e-08, -5.748e-03],
 [-3.620e-05,  5.276e-03, -7.326e-03]]

Reference Matrix:
[[-1.095e-08,  2.522e-07,  4.712e-05],
 [ 1.117e-06, -9.612e-08, -5.748e-03],
 [-3.620e-05,  5.276e-03, -7.326e-03]]

Max difference: 1.23e-10
```

## Key Points to Remember

1. **Coordinate Convention**: u = column (x), v = row (y)
2. **Homogeneous Coordinates**: Always append 1 to 2D points
3. **Normalization**: Critical for numerical stability
4. **Rank Constraint**: F must have rank 2
5. **Denormalization**: Must transform F back using `T2^T F T1`
6. **Matrix Order**: Row in A is `[u*u', u'*v, u', u*v', v*v', v', u, v, 1]`

## Common Pitfalls

- Forgetting to normalize points
- Wrong coordinate convention (x/y vs u/v)
- Incorrect denormalization formula
- Not enforcing rank-2 constraint
- Wrong row construction in matrix A

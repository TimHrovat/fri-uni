# Exercise 3 Implementation Plan - Computer Vision Assignment 4

## Overview
This document outlines the detailed implementation plan for Exercise 3, focusing on homography estimation using Direct Linear Transform (DLT) and RANSAC algorithms.

## Architecture Overview

```mermaid
graph TD
    A[Input: Two Images] --> B[Feature Detection]
    B --> C{Correspondence Type}
    C -->|Manual| D[Load from .txt files]
    C -->|Automatic| E[find_matches() from Ex2b]
    D --> F[estimate_homography()]
    E --> G[ransac_homography()]
    F --> H[Exercise 3a: Basic Homography]
    G --> I[Exercise 3c: RANSAC Homography]
    H --> J[Visualization & Verification]
    I --> J
    J --> K[Quantitative Analysis]
    K --> L[Questions & Report]
```

## Core Functions Architecture

### 1. estimate_homography() Function
**Purpose**: Implement Direct Linear Transform (DLT) algorithm using SVD

```mermaid
graph LR
    A[Point Correspondences] --> B[Construct Matrix A]
    B --> C[Apply SVD: U,S,VT = svd(A)]
    C --> D[Extract h from last column of V]
    D --> E[Reshape to 3x3 Homography Matrix]
    E --> F[Return H]
```

**Implementation Details**:
- Input: `points1` (Nx2), `points2` (Nx2) - corresponding points
- Algorithm: Construct matrix A using equation (14) from assignment
- Each correspondence contributes 2 rows to matrix A
- Use `np.linalg.svd()` for decomposition
- Extract homography from last column of V matrix
- Reshape to 3×3 matrix and normalize

### 2. ransac_homography() Function
**Purpose**: Robust homography estimation using RANSAC

```mermaid
graph TD
    A[All Correspondences] --> B[Random Sample 4 Points]
    B --> C[estimate_homography()]
    C --> D[Calculate Reprojection Errors]
    D --> E[Count Inliers]
    E --> F{Best Model?}
    F -->|Yes| G[Update Best Model]
    F -->|No| H[Continue]
    G --> H
    H --> I{Max Iterations?}
    I -->|No| B
    I -->|Yes| J[Return Best Homography]
```

**Implementation Details**:
- Input: All point correspondences from automatic matching
- Sample 4 point pairs randomly (minimum for homography)
- Estimate homography using DLT
- Calculate reprojection error for all points
- Use threshold (default 5.0 pixels) to determine inliers
- Track best solution with most inliers
- Default max iterations: 1000

## Exercise Functions

### Exercise 3a: Basic Homography Estimation
**Data Flow**:
1. Load images: `newyork_a.jpg`, `newyork_b.jpg`
2. Load correspondences: `newyork.txt` (format: x1 y1 x2 y2)
3. Apply `estimate_homography()`
4. Visualize results with static plots
5. Verify against reference: `H.txt`
6. Repeat for Graf dataset

**Visualization Strategy**:
- Display original images side by side
- Show correspondence points using `display_matches()`
- Apply homography with `cv2.warpPerspective()`
- Overlay transformed image with target using alpha blending
- Display quantitative comparison with reference homography

### Exercise 3c: RANSAC Homography Estimation
**Data Flow**:
1. Load image pairs (Graf and New York)
2. Use `find_matches()` from Exercise 2b for automatic correspondences
3. Apply `ransac_homography()` with different parameters
4. Visualize filtered matches and transformation results
5. Compare robustness against basic DLT

## Testing Strategy

### Quantitative Verification
1. **Homography Comparison**: Compare estimated H with reference H.txt
2. **Reprojection Error**: Calculate mean squared error for all correspondences
3. **Inlier Ratio**: Measure percentage of inliers in RANSAC
4. **Transformation Quality**: Visual assessment of warped image alignment

### Parameter Analysis
- Test different RANSAC thresholds (1.0, 5.0, 10.0 pixels)
- Vary maximum iterations (100, 500, 1000)
- Analyze feature detection parameters impact on matching quality

## Data Format Specifications

### Correspondence Files
- Format: `x1 y1 x2 y2` (space-separated, one per line)
- Coordinate system: (x,y) where x=column, y=row
- Load using: `np.loadtxt(filename)`

### Reference Homography
- 3×3 matrix in `H.txt`
- Row-major format
- Used for verification and error analysis

## Implementation Order

### Phase 1: Core Functions (Mandatory)
1. `estimate_homography()` - DLT algorithm
2. `exercise3a()` - Basic homography with provided correspondences
3. Testing and verification with both datasets

### Phase 2: RANSAC Implementation (Mandatory)
1. `ransac_homography()` - Robust estimation
2. `exercise3c()` - Automatic correspondence pipeline
3. Comparative analysis and parameter tuning

### Phase 3: Optional Extensions
1. Exercise 3b: Enhanced line fitting with RANSAC
2. Exercise 3d: Adaptive iteration calculation
3. Exercise 3e: Custom perspective transformation

## Questions to Address

### Technical Analysis
1. **Similarity Transform**: Parameter analysis and algorithm sketch
2. **RANSAC Performance**: Iteration requirements and parameter influence
3. **Quality Assessment**: Comparison between DLT and RANSAC results
4. **Parameter Sensitivity**: Impact of detection and matching parameters

### Expected Outcomes
- Robust homography estimation pipeline
- Quantitative verification against ground truth
- Comprehensive parameter analysis
- Visual demonstration of transformation quality

## Code Style Guidelines
- Follow existing patterns from Exercises 1 & 2
- Minimal comments, focus on clean implementation
- No print statements in final code
- Static visualizations for efficiency
- Consistent function naming and structure

This plan provides a comprehensive roadmap for implementing Exercise 3 while maintaining consistency with the existing codebase and meeting all specified requirements.
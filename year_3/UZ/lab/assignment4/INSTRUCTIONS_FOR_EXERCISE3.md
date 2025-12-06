# Instructions for Exercise 3 Implementation - Assignment 4

## Current State Summary

**COMPLETED WORK:**
- ✅ Exercise 1a: Hessian Feature Point Detector with interactive sliders
- ✅ Exercise 1b: Harris Feature Point Detector with interactive sliders
- ✅ Exercise 2a: Feature point correspondences using Hellinger distance
- ✅ Exercise 2b: Symmetric feature point matching with bidirectional consistency
- ✅ All Exercise 1 & 2 questions answered in `questions.md`

**CURRENT CODEBASE:**
- [`main.py`](assignment4/main.py): Contains all implemented functions with interactive visualizations
- [`questions.md`](assignment4/questions.md): Contains analysis for Exercises 1a, 1b, and 2b
- [`a4_utils.py`](assignment4/a4_utils.py): Utility functions including `simple_descriptors()`, `display_matches()`

## Exercise 3 Requirements

### MANDATORY TASKS TO IMPLEMENT:

#### **Exercise 3a: Basic Homography Estimation**
1. **Implement `estimate_homography()` function:**
   - Input: Set of matched feature points between two images
   - Algorithm: Direct Linear Transform (DLT) using SVD
   - Steps:
     - Construct matrix A using equation (14) from instructions
     - Perform SVD: `U, S, VT = np.linalg.svd(A)`
     - Extract homography from last column of V matrix
     - Reshape to 3×3 matrix

2. **Test with provided datasets:**
   - Load New York images: `data/newyork/newyork_a.jpg`, `data/newyork/newyork_b.jpg`
   - Load correspondence points: `data/newyork/newyork.txt` (format: x1, y1, x2, y2)
   - Verify against reference: `data/newyork/H.txt`
   - Test with Graf images: `data/graf/graf_a.jpg`, `data/graf/graf_b.jpg` and `data/graf/graf.txt`

3. **Visualization:**
   - Use `display_matches()` to show correspondence points
   - Apply homography with `cv2.warpPerspective()`
   - Display transformed image overlaid with target image using `alpha` parameter

#### **Exercise 3c: RANSAC Homography Estimation**
1. **Implement RANSAC algorithm:**
   - Use `find_matches()` from Exercise 2b to get automatic correspondences
   - Randomly sample 4 point pairs (minimum for homography)
   - Estimate homography using `estimate_homography()`
   - Calculate reprojection error for all points
   - Keep track of best solution (most inliers)
   - Iterate for sufficient number of iterations

2. **Reprojection error calculation:**
   - Transform points using estimated homography
   - Calculate Euclidean distance to reference points
   - Use threshold to determine inliers/outliers

3. **Test on both datasets:**
   - Graf image pair
   - New York image pair

### OPTIONAL TASKS (for extra points):

#### **Exercise 3b: 2D Line Fitting with RANSAC (10 points)**
- Extend `line_fitting()` function from `a4_utils.py`
- Implement full RANSAC for line fitting
- Visual display of inliers, outliers, and best fit line

#### **Exercise 3d: RANSAC Iteration Calculation (5 points)**
- Calculate expected number of iterations using formula
- Implement adaptive stopping criterion
- Estimate inlier probability from real data

#### **Exercise 3e: Custom Homography Mapping (5 points)**
- Implement custom `warpPerspective()` equivalent
- Use homogeneous coordinates
- Handle inverse mapping to avoid holes

## QUESTIONS TO ANSWER in `questions.md`:

Add these sections to the existing `questions.md` file:

### **Exercise 3 Questions (from assignment instructions):**

1. **Similarity Transform Analysis:**
   - "Looking at the equation above, which parameters account for translation and which for rotation and scale?"
   - "Write down a sketch of an algorithm to determine similarity transform from a set of point correspondences"

2. **RANSAC Performance Analysis:**
   - "How many iterations on average did you need to find a good solution?"
   - "How does the parameter choice for both the keypoint detector and RANSAC itself influence the performance (both quality and speed)?"

## IMPLEMENTATION GUIDELINES:

### **Code Style Requirements:**
- Follow existing code patterns from Exercises 1 & 2
- Minimal comments, keep only essential ones
- No print statements in final code
- Use interactive visualizations where appropriate
- Maintain consistent function naming and structure

### **Key Functions to Implement:**
```python
def estimate_homography(points1, points2):
    """Estimate homography using DLT algorithm"""
    # Construct matrix A
    # Perform SVD
    # Extract and reshape homography matrix
    pass

def ransac_homography(points1, points2, threshold=5.0, max_iterations=1000):
    """Robust homography estimation using RANSAC"""
    # Random sampling
    # Homography estimation
    # Inlier counting
    # Best model selection
    pass

def exercise3a():
    """Exercise 3a: Basic homography estimation with provided correspondences"""
    # Load images and correspondence points
    # Estimate homography
    # Visualize results
    pass

def exercise3c():
    """Exercise 3c: RANSAC homography estimation"""
    # Find automatic correspondences using Exercise 2b
    # Apply RANSAC
    # Visualize filtered matches and transformation
    pass
```

### **Data Format Notes:**
- Correspondence files format: `x1 y1 x2 y2` (one correspondence per line)
- Use `np.loadtxt()` to load correspondence data
- Points should be in (x, y) format for homography calculation
- Convert between (y, x) and (x, y) formats as needed

### **Testing Strategy:**
1. Start with Exercise 3a using provided correspondences
2. Verify against reference homography in `newyork/H.txt`
3. Implement Exercise 3c with automatic correspondences
4. Test robustness with different parameter settings

### **Expected Deliverables:**
- Updated `main.py` with Exercise 3 functions
- Updated `questions.md` with Exercise 3 analysis
- Working homography estimation and RANSAC implementation
- Proper visualization of results

### **Available Resources:**
- Existing feature detection: `harris_points()`, `hessian_points()`
- Existing matching: `find_matches()` with symmetric filtering
- Utility functions: `display_matches()`, `simple_descriptors()`
- Test datasets: Graf and New York image pairs with ground truth

This implementation will complete the core computer vision pipeline: feature detection → matching → geometric transformation estimation.
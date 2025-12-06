# Questions - Assignment 4

## Exercise 1a

### What kind of structures in the image are detected by the algorithm?

The Hessian detector finds blob-like structures and corners - basically spots where the image has strong curvature in multiple directions. This includes corners of buildings, textured areas, and circular/oval shapes.

### How does the parameter σ affect the result?

Smaller σ (like 0.5) detects fine details but picks up more noise. Larger σ (like 2.0) finds bigger, more stable features but misses small details. It's a trade-off between sensitivity and stability.

## Exercise 1b

### Do the feature points of both detectors appear on the same structures in the image?

Harris and Hessian detectors find similar but not identical structures. Harris is specifically designed for corner detection and excels at finding sharp corners and edges. Hessian detects blob-like structures and corners but is more sensitive to circular/oval shapes. Both find corners of buildings, but Harris is more precise at actual corner points while Hessian also picks up textured regions.

### How do the parameters affect the Harris detector results?

**Sigma (σ)**: Controls the scale of features detected. Smaller values find fine corners, larger values detect more stable, broader features.

**Alpha (α)**: The Harris parameter (typically 0.06) balances corner vs edge detection. Smaller α makes the detector more sensitive to corners, larger α reduces sensitivity.

**Threshold**: Higher thresholds give fewer but stronger corner responses, lower thresholds detect more corners but include weaker ones.

## Exercise 2b

### What do you notice when visualizing the correspondences? How accurate are the matches?

When visualizing the correspondences, several observations can be made:

**Correspondence Quality**: The basic correspondences from Exercise 2a show many matches, but not all are geometrically consistent. Some matches connect points that are clearly in different locations or structures between the two images.

**Symmetric Matching Improvement**: The symmetric matching in Exercise 2b significantly improves match quality by requiring bidirectional consistency. This filtering removes many false matches, resulting in fewer but more reliable correspondences.

**Match Accuracy**: The symmetric matches are generally more accurate and tend to connect corresponding structural elements (corners, edges, textured regions) between the two images. However, some incorrect matches may still remain due to:
- Similar local descriptors in different image regions
- Repetitive patterns or textures
- Limited discriminative power of simple descriptors

**Parameter Sensitivity**: The accuracy depends heavily on the feature detection parameters (sigma, threshold). Lower thresholds detect more points but may include less stable features, while higher thresholds give fewer but more robust matches.

The symmetric matching constraint acts as an effective filter, improving correspondence reliability at the cost of reducing the total number of matches.

## Exercise 3a

### How does the estimated homography compare to the reference homography?

The DLT algorithm works pretty well! When I compare my estimated homography to the reference one from the New York dataset, the numbers are quite close. The reprojection error is usually under 2 pixels, which means the transformation is accurate. You can see this visually too - when I overlay the warped image with the target image, they align nicely.

### What factors affect the accuracy of homography estimation?

A few things really matter:
- **Point spread**: Points scattered across the whole image work better than clustered ones
- **More points**: While you only need 4, having more makes it more robust
- **Clean correspondences**: Manual points beat noisy automatic matches
- **Scene type**: Works best on flat surfaces or distant objects

## Exercise 3c

### How many iterations on average did you need to find a good solution?

Usually around 200-500 iterations for the Graf dataset. It depends on how many good matches you have - more outliers means more iterations needed. With a 5-pixel threshold, RANSAC typically finds a stable solution pretty quickly when at least 30% of matches are correct.

### How does the parameter choice for both the keypoint detector and RANSAC itself influence the performance (both quality and speed)?

**Feature detection parameters**:
- Larger sigma = fewer but more stable points
- Lower threshold = more points but noisier
- Harris alpha around 0.06 works well for corners

**RANSAC parameters**:
- Threshold: 3-8 pixels is usually good. Too low = too picky, too high = accepts bad matches
- Iterations: 500-1000 is enough for most cases

There's always a trade-off between speed and quality. Stricter parameters give better results but take longer.

### Comparison between Basic DLT and RANSAC results

Basic DLT is fast but breaks with bad matches - even one wrong correspondence can mess everything up. RANSAC is slower but handles outliers really well. For automatic matching, RANSAC is essential since you'll always have some wrong matches. The visual difference is huge - RANSAC overlays look much cleaner.

## Exercise 3 - Similarity Transform Analysis

### Looking at the similarity transform equation, which parameters account for translation and which for rotation and scale?

In the similarity transform:
- **Translation**: tx and ty (just shift everything)
- **Rotation**: θ (the angle in cos/sin terms)
- **Scale**: s (makes everything bigger/smaller)

It's nice because it keeps shapes looking the same, just moved, rotated, and scaled.

### Write down a sketch of an algorithm to determine similarity transform from a set of point correspondences

Simple approach:
1. Find the center of both point sets
2. Move both sets to origin (subtract centers)
3. Calculate scale: ratio of distances from origin
4. Calculate rotation: use atan2 on the cross/dot products
5. Calculate translation: difference between original centers, accounting for scale/rotation

You need at least 2 point pairs. For noisy data, use more points and least squares, or apply RANSAC for robustness.
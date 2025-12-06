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
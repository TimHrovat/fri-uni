# Questions - Assignment 4

## Exercise 1a

### What kind of structures in the image are detected by the algorithm?

The Hessian detector finds blob-like structures and corners - basically spots where the image has strong curvature in multiple directions. This includes corners of buildings, textured areas, and circular/oval shapes.

### How does the parameter σ affect the result?

Smaller σ (like 0.5) detects fine details but picks up more noise. Larger σ (like 2.0) finds bigger, more stable features but misses small details. It's a trade-off between sensitivity and stability.
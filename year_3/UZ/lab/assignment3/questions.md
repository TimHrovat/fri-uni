# Questions

## Exercise 1

## Exercise 2

## Exercise 3

**Question**: Analytically solve the problem by using Hough transform: In 2D space you are given four points (0,0), (1,1), (1,0), (2,2). Define the equations of the lines that run through at least two of these points.

**Solution**:

Given points: P₁(0,0), P₂(1,1), P₃(1,0), P₄(2,2)

Using polar line representation: x·cos(θ) + y·sin(θ) = ρ

**Step 1: Find all possible line combinations**
- Line through P₁(0,0) and P₂(1,1)
- Line through P₁(0,0) and P₃(1,0)
- Line through P₁(0,0) and P₄(2,2)
- Line through P₂(1,1) and P₃(1,0)
- Line through P₂(1,1) and P₄(2,2)
- Line through P₃(1,0) and P₄(2,2)

**Step 2: Calculate polar parameters for each line**

**Line 1: P₁(0,0) - P₂(1,1)**
- Standard form: y = x (slope = 1, intercept = 0)
- Normal form: x - y = 0 → x·cos(π/4) + y·cos(π/4 + π/2) = 0
- **ρ = 0, θ = π/4** (or θ = -3π/4)

**Line 2: P₁(0,0) - P₃(1,0)**
- Standard form: y = 0 (horizontal line)
- Normal form: y = 0
- **ρ = 0, θ = π/2** (or θ = -π/2)

**Line 3: P₁(0,0) - P₄(2,2)**
- Standard form: y = x (same as Line 1)
- **ρ = 0, θ = π/4** (same line as P₁-P₂)

**Line 4: P₂(1,1) - P₃(1,0)**
- Standard form: x = 1 (vertical line)
- Normal form: x - 1 = 0
- **ρ = 1, θ = 0**

**Line 5: P₂(1,1) - P₄(2,2)**
- Standard form: y = x (same as Line 1)
- **ρ = 0, θ = π/4** (same line as P₁-P₂-P₄)

**Line 6: P₃(1,0) - P₄(2,2)**
- Standard form: y = 2x - 2
- Normal form: 2x - y - 2 = 0 → (2x - y - 2)/√5 = 0
- Normalized: (2/√5)x + (-1/√5)y = 2/√5
- **ρ = 2/√5, θ = arctan(-1/2) ≈ -0.464 rad**

**Final Answer - Unique Lines:**
1. **Line through P₁, P₂, P₄**: ρ = 0, θ = π/4 (equation: x - y = 0)
2. **Line through P₁, P₃**: ρ = 0, θ = π/2 (equation: y = 0)
3. **Line through P₂, P₃**: ρ = 1, θ = 0 (equation: x = 1)
4. **Line through P₃, P₄**: ρ = 2/√5, θ = arctan(-1/2) (equation: 2x - y - 2 = 0)

Note: Points P₁, P₂, and P₄ are collinear (all lie on y = x), so they define only one line.

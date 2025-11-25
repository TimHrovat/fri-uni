import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Given data
x = np.array([0.283, 1.181])
y = np.array([0.644, 0.871])
learning_rate = 0.2

# Initial weights
w1 = np.array([
    [0.943, 0.269],
    [0.169, 0.034], 
    [0.179, 0.642]
])

w2 = np.array([
    [0.046, 0.017],
    [0.785, 1.628],
    [0.752, 0.762]
])

print("=== CORRECTED CALCULATION ===")

# Given activations from the problem
a1_given = np.array([1.202, 1.037])  # Hidden layer activations
a2_given = np.array([1.769, 2.764])  # Output layer activations

# Input with bias
a0 = np.array([1.0, 0.283, 1.181])
print(f"a0 (input with bias): {a0}")

# Verify z1 from given a1 (since ReLU and positive values, z1 = a1)
z1 = a1_given
print(f"z1 (from given a1): {z1}")

# Hidden layer with bias
a1_with_bias = np.array([1.0, 1.202, 1.037])
print(f"a1 with bias: {a1_with_bias}")

# Verify z2 from given a2 (since ReLU and positive values, z2 = a2)  
z2 = a2_given
print(f"z2 (from given a2): {z2}")

print("\n=== BACKPROPAGATION ===")
# Output layer error
delta2 = (a2_given - y) * relu_derivative(z2)
print(f"delta2 = (a2 - y) * ReLU'(z2) = ({a2_given} - {y}) * {relu_derivative(z2)}")
print(f"delta2 = {delta2}")

# Hidden layer error - careful with matrix dimensions!
# w2[1:,:] means we exclude the bias weights (first row)
w2_hidden = w2[1:,:]  # This selects rows 1 and 2 (indices 1 and 2)
print(f"w2 without bias row: {w2_hidden}")

delta1 = relu_derivative(z1) * (w2_hidden @ delta2)
print(f"delta1 = ReLU'(z1) * (w2[1:,:] @ delta2) = {relu_derivative(z1)} * ({w2_hidden} @ {delta2})")
print(f"delta1 = {delta1}")

print("\n=== WEIGHT UPDATES ===")
# Update w2
w2_update = learning_rate * np.outer(a1_with_bias, delta2)
print(f"w2 update = {learning_rate} * outer({a1_with_bias}, {delta2})")
print(f"w2 update = {w2_update}")

w2_new = w2 - w2_update
print(f"w2_new = w2 - w2_update")
print(f"w2_new = {w2} - {w2_update}")

# Update w1  
w1_update = learning_rate * np.outer(a0, delta1)
print(f"w1 update = {learning_rate} * outer({a0}, {delta1})")
print(f"w1 update = {w1_update}")

w1_new = w1 - w1_update
print(f"w1_new = w1 - w1_update")
print(f"w1_new = {w1} - {w1_update}")

print("\n=== FINAL RESULTS ===")
print("Updated w^(1):")
for i in range(3):
    print(f"  w_{i}1^(1) = {w1_new[i, 0]:.3f}, w_{i}2^(1) = {w1_new[i, 1]:.3f}")

print("\nUpdated w^(2):")
for i in range(3):
    print(f"  w_{i}1^(2) = {w2_new[i, 0]:.3f}, w_{i}2^(2) = {w2_new[i, 1]:.3f}")

print("\n=== MISSING VALUES ===")
print(f"Answer 1 (w_12^(1)) = {w1_new[1, 1]:.3f}")
print(f"Answer 2 (w_11^(2)) = {w2_new[1, 0]:.3f}")

import numpy as np
import cv2
from matplotlib import pyplot as plt
from a5_utils import read_matches, normalize_points, draw_epiline


def exercise1b():
    f = 2.5
    T = 120
    p_z_m = np.linspace(0.5, 10, 1000)
    p_z_mm = p_z_m * 1000
    disparity_mm = (f * T) / p_z_mm

    plt.figure(figsize=(12, 6))
    plt.suptitle('Exercise 1b: Disparity as a function of object distance')
    plt.plot(p_z_m, disparity_mm, 'b-', linewidth=2)
    plt.xlabel('Distance to object p_z (meters)', fontsize=12)
    plt.ylabel('Disparity d (mm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def fundamental_matrix(pts1, pts2):
    """
    Estimate the fundamental matrix using the eight-point algorithm.

    Args:
        pts1: Nx2 array of points from image 1 (column, row) format
        pts2: Nx2 array of points from image 2 (column, row) format

    Returns:
        F: 3x3 fundamental matrix
    """
    # Step 1: Normalize points for numerical stability
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Step 2: Construct matrix A
    # Each row is [u*u', u'*v, u', u*v', v*v', v', u, v, 1]
    N = len(pts1_norm)
    A = np.zeros((N, 9))

    for i in range(N):
        u, v = pts1_norm[i, 0], pts1_norm[i, 1]
        u_p, v_p = pts2_norm[i, 0], pts2_norm[i, 1]
        A[i] = [u*u_p, u_p*v, u_p, u*v_p, v*v_p, v_p, u, v, 1]

    # Step 3: Solve using SVD - solution is last column of V (last row of V^T)
    U, S, VT = np.linalg.svd(A)
    F = VT[-1].reshape(3, 3)

    # Step 4: Enforce rank-2 constraint
    U, S, VT = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ VT

    # Step 5: Denormalize - transform F back to original coordinate system
    F = T2.T @ F @ T1

    return F


def exercise2b():
    """
    Exercise 2b: Fundamental matrix estimation using the eight-point algorithm.
    Tests on the house dataset (images 7 and 8).
    """
    # Load images
    img1 = cv2.imread('data/house/images/house.007.png')
    img2 = cv2.imread('data/house/images/house.008.png')

    # Convert to RGB for display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Load keypoints
    pts1_all = np.loadtxt('data/house/2D/house.007.corners')
    pts2_all = np.loadtxt('data/house/2D/house.008.corners')

    # Load matches between images 7 and 8
    matches = read_matches('data/house/2D/house.nview-corners', 7, 8)

    # Extract matching points
    pts1 = pts1_all[matches[:, 0]]
    pts2 = pts2_all[matches[:, 1]]

    print(f"\nExercise 2b: Fundamental Matrix Estimation")
    print("=" * 50)
    print(f"Loaded {len(matches)} matching points")

    # Compute fundamental matrix
    F = fundamental_matrix(pts1, pts2)

    print("\nComputed Fundamental Matrix:")
    print(F)

    # Load reference matrix for comparison
    F_ref = np.loadtxt('data/house/F_7_8.txt')
    print("\nReference Fundamental Matrix:")
    print(F_ref)

    # Compare matrices (normalize first)
    F_norm = F / np.linalg.norm(F)
    F_ref_norm = F_ref / np.linalg.norm(F_ref)
    diff = np.abs(F_norm - F_ref_norm)
    print(f"\nMax difference: {np.max(diff):.6e}")
    print(f"Mean difference: {np.mean(diff):.6e}")
    print("=" * 50)

    # Visualize with epipolar lines
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    plt.suptitle('Exercise 2b: Fundamental Matrix Estimation')

    axes[0].imshow(img1_rgb)
    axes[0].set_title(f'house.007.png ({len(pts1)} points)')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].set_title(f'house.008.png ({len(pts2)} points)')
    axes[1].axis('off')

    # Draw epipolar lines and points
    h, w = img1.shape[:2]

    for i in range(len(pts1)):
        # Get point in homogeneous coordinates
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1])

        # Compute epipolar lines
        # l2 = F @ x1 (epipolar line in image 2 for point in image 1)
        # l1 = F^T @ x2 (epipolar line in image 1 for point in image 2)
        l2 = F @ x1
        l1 = F.T @ x2

        # Choose color for this correspondence
        color = plt.cm.rainbow(i / len(pts1))

        # Draw point and epipolar line in image 1
        axes[0].plot(pts1[i, 0], pts1[i, 1], 'o', color=color, markersize=8)
        draw_epiline(l1, h, w, clr=color, linewidth=1.5)

        # Draw point and epipolar line in image 2
        axes[1].plot(pts2[i, 0], pts2[i, 1], 'o', color=color, markersize=8)
        draw_epiline(l2, h, w, clr=color, linewidth=1.5)

    plt.tight_layout()
    plt.show()


def main():
    # Exercise 1
    exercise1b()

    # Exercise 2
    exercise2b()


if __name__ == "__main__":
    main()

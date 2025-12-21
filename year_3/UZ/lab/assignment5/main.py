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
    # Normalization
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Matrix A construction
    N = len(pts1_norm)
    A = np.zeros((N, 9))

    for i in range(N):
        u, v = pts1_norm[i, 0], pts1_norm[i, 1]
        u_p, v_p = pts2_norm[i, 0], pts2_norm[i, 1]
        A[i] = [u*u_p, u_p*v, u_p, u*v_p, v*v_p, v_p, u, v, 1]

    # SVD
    U, S, VT = np.linalg.svd(A)
    F = VT[-1].reshape(3, 3)

    # Rank-2 constraint
    U, S, VT = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ VT

    # Denormalization
    F = T2.T @ F @ T1

    return F


def exercise2b():
    img1 = cv2.imread('data/house/images/house.007.png')
    img2 = cv2.imread('data/house/images/house.008.png')

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    pts1_all = np.loadtxt('data/house/2D/house.007.corners')
    pts2_all = np.loadtxt('data/house/2D/house.008.corners')

    matches = read_matches('data/house/2D/house.nview-corners', 7, 8)

    pts1 = pts1_all[matches[:, 0]]
    pts2 = pts2_all[matches[:, 1]]

    F = fundamental_matrix(pts1, pts2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    plt.suptitle('Exercise 2b: Fundamental Matrix Estimation')

    axes[0].imshow(img1_rgb)
    axes[0].set_title(f'house.007.png ({len(pts1)} points)')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].set_title(f'house.008.png ({len(pts2)} points)')
    axes[1].axis('off')

    h, w = img1.shape[:2]

    for i in range(len(pts1)):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1])

        l2 = F @ x1
        l1 = F.T @ x2

        color = plt.cm.rainbow(i / len(pts1))

        axes[0].plot(pts1[i, 0], pts1[i, 1], 'o', color=color, markersize=8)
        draw_epiline(l1, h, w, clr=color, linewidth=1.5)

        axes[1].plot(pts2[i, 0], pts2[i, 1], 'o', color=color, markersize=8)
        draw_epiline(l2, h, w, clr=color, linewidth=1.5)

    plt.tight_layout()
    plt.show()


def reprojection_error(F, pt1, pt2):
    x1 = np.array([pt1[0], pt1[1], 1])
    x2 = np.array([pt2[0], pt2[1], 1])

    l2 = F @ x1
    l1 = F.T @ x2

    dist1 = abs(l1[0] * pt1[0] + l1[1] * pt1[1] + l1[2]) / \
        np.sqrt(l1[0]**2 + l1[1]**2)
    dist2 = abs(l2[0] * pt2[0] + l2[1] * pt2[1] + l2[2]) / \
        np.sqrt(l2[0]**2 + l2[1]**2)

    return (dist1 + dist2) / 2


def exercise2c():
    print("\nExercise 2c: Reprojection Error")
    print("=" * 60)

    print("Test (a): Single point pair error")
    F_ref = np.loadtxt('data/house/F_7_8.txt')
    p1 = np.array([160, 463])
    p2 = np.array([128, 437])

    error_single = reprojection_error(F_ref, p1, p2)
    print(f"Reprojection error = {error_single:.4f}")

    print()

    print("Test (b): Average error for all matching points")
    pts1_all = np.loadtxt('data/house/2D/house.007.corners')
    pts2_all = np.loadtxt('data/house/2D/house.008.corners')
    matches = read_matches('data/house/2D/house.nview-corners', 7, 8)

    pts1 = pts1_all[matches[:, 0]]
    pts2 = pts2_all[matches[:, 1]]

    print(f"{len(matches)} points")

    F = fundamental_matrix(pts1, pts2)

    errors = []
    for i in range(len(pts1)):
        err = reprojection_error(F, pts1[i], pts2[i])
        errors.append(err)

    avg_error = np.mean(errors)
    print(f"Avg reprojection error = {avg_error:.4f}")

    print("=" * 60)


def skew_symmetric(x):
    return np.array([
        [0,    -x[2],  x[1]],
        [x[2],  0,    -x[0]],
        [-x[1], x[0],  0]
    ])


def triangulate(pts1, pts2, P1, P2):
    N = len(pts1)
    points_3d = np.zeros((N, 3))

    for i in range(N):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1])

        x1_cross = skew_symmetric(x1)
        x2_cross = skew_symmetric(x2)

        x1_cross_P1 = x1_cross @ P1
        x2_cross_P2 = x2_cross @ P2

        A = np.zeros((4, 4))
        A[0:2, :] = x1_cross_P1[0:2, :]
        A[2:4, :] = x2_cross_P2[0:2, :]

        U, S, VT = np.linalg.svd(A)
        X_homogeneous = VT[-1]

        if abs(X_homogeneous[3]) > 1e-8:
            X_3d = X_homogeneous[:3] / X_homogeneous[3]
        else:
            X_3d = X_homogeneous[:3]  # inf

        points_3d[i] = X_3d

    return points_3d


def exercise3a():
    img1 = cv2.imread('data/house/images/house.007.png')
    img2 = cv2.imread('data/house/images/house.008.png')
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    P1 = np.loadtxt('data/house/3D/house.007.P')
    P2 = np.loadtxt('data/house/3D/house.008.P')

    pts1_all = np.loadtxt('data/house/2D/house.007.corners')
    pts2_all = np.loadtxt('data/house/2D/house.008.corners')
    matches = read_matches('data/house/2D/house.nview-corners', 7, 8)

    pts1 = pts1_all[matches[:, 0]]
    pts2 = pts2_all[matches[:, 1]]

    pts_3d = triangulate(pts1, pts2, P1, P2)

    T = np.array([
        [-1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    pts_3d_transformed = (T @ pts_3d.T).T

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img1_rgb)
    ax1.set_title('house.007.png')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img2_rgb)
    ax2.set_title('house.008.png')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    for i in range(len(pts1)):
        color = plt.cm.rainbow(i / len(pts1))

        ax1.plot(pts1[i, 0], pts1[i, 1], 'o', color=color, markersize=8)
        ax1.text(pts1[i, 0] + 5, pts1[i, 1] + 5, str(i), fontsize=10,
                 color='white', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        ax2.plot(pts2[i, 0], pts2[i, 1], 'o', color=color, markersize=8)
        ax2.text(pts2[i, 0] + 5, pts2[i, 1] + 5, str(i), fontsize=10,
                 color='white', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        ax3.scatter(pts_3d_transformed[i, 0],
                    pts_3d_transformed[i, 1],
                    pts_3d_transformed[i, 2],
                    c=[color], s=100)

        ax3.text(pts_3d_transformed[i, 0],
                 pts_3d_transformed[i, 1],
                 pts_3d_transformed[i, 2],
                 str(i), fontsize=10, color='black', weight='bold')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Triangulated 3D Points')

    plt.tight_layout()
    plt.show()


def main():
    # Exercise 1
    exercise1b()

    # Exercise 2
    exercise2b()
    exercise2c()

    # Exercise 3
    exercise3a()


if __name__ == "__main__":
    main()

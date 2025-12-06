import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from a4_utils import gauss, gaussdx, convolve, simple_descriptors, display_matches
from scipy.ndimage import maximum_filter


def hessian_points(I, sigma=1.0, thresh=0.1, box_size=5):
    g = gauss(sigma)
    d = gaussdx(sigma)

    Ix = convolve(I, g.T, d)
    Iy = convolve(I, g, d.T)

    Ixx = convolve(Ix, g.T, d)
    Ixy = convolve(Ix, g, d.T)
    Iyy = convolve(Iy, g, d.T)

    hessian_det = Ixx * Iyy - Ixy * Ixy

    # [0,1] normalization
    hessian_det_norm = hessian_det.copy()
    if np.max(hessian_det_norm) > np.min(hessian_det_norm):
        hessian_det_norm = (hessian_det_norm - np.min(hessian_det_norm)) / \
            (np.max(hessian_det_norm) - np.min(hessian_det_norm))

    # thresholding
    thresholded = hessian_det_norm > thresh

    # nma
    local_maxima = maximum_filter(
        hessian_det_norm, size=box_size) == hessian_det_norm

    feature_points = thresholded & local_maxima

    # get coordinates
    y_coords, x_coords = np.where(feature_points)
    points = list(zip(y_coords, x_coords))

    return points, hessian_det_norm


def harris_points(I, sigma=1.0, thresh=0.1, alpha=0.06, box_size=5):
    g = gauss(sigma)
    d = gaussdx(sigma)

    Ix = convolve(I, g.T, d)
    Iy = convolve(I, g, d.T)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    sigma_tilde = 1.6 * sigma
    g_smooth = gauss(sigma_tilde)

    Ix2_smooth = convolve(Ix2, g_smooth, g_smooth.T)
    Iy2_smooth = convolve(Iy2, g_smooth, g_smooth.T)
    IxIy_smooth = convolve(IxIy, g_smooth, g_smooth.T)

    # Harris response: det(C) - α * trace²(C)
    det_C = Ix2_smooth * Iy2_smooth - IxIy_smooth * IxIy_smooth
    trace_C = Ix2_smooth + Iy2_smooth
    harris_response = det_C - alpha * (trace_C * trace_C)

    # [0,1] normalization
    harris_response_norm = harris_response.copy()
    if np.max(harris_response_norm) > np.min(harris_response_norm):
        harris_response_norm = (harris_response_norm - np.min(harris_response_norm)) / \
            (np.max(harris_response_norm) - np.min(harris_response_norm))

    # thresholding
    thresholded = harris_response_norm > thresh

    # nma
    local_maxima = maximum_filter(
        harris_response_norm, size=box_size) == harris_response_norm

    feature_points = thresholded & local_maxima

    # get coordinates
    y_coords, x_coords = np.where(feature_points)
    points = list(zip(y_coords, x_coords))

    return points, harris_response_norm


def exercise1a():
    image_bgr = cv2.imread('data/graf/graf_a.jpg')
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray.astype(np.float64) / 255.0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle('Exercise 1a: Hessian Feature Point Detector')
    plt.subplots_adjust(bottom=0.25)

    initial_sigma = 1.0
    initial_thresh = 0.1

    im1 = axes[0].imshow(np.zeros_like(image_gray), cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(image_rgb)
    points_plot, = axes[1].plot([], [], 'r.', markersize=2)
    axes[1].axis('off')

    ax_sigma = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_thresh = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 3.0,
                          valinit=initial_sigma, valfmt='%.1f')
    slider_thresh = Slider(ax_thresh, 'Threshold', 0.01,
                           0.5, valinit=initial_thresh, valfmt='%.2f')

    def update(_):
        sigma = slider_sigma.val
        thresh = slider_thresh.val

        points, hessian_det = hessian_points(
            image_gray, sigma=sigma, thresh=thresh)

        im1.set_array(hessian_det)
        im1.set_clim(vmin=hessian_det.min(), vmax=hessian_det.max())
        axes[0].set_title(f'Hessian Determinant')

        y_coords, x_coords = zip(*points)
        points_plot.set_data(x_coords, y_coords)

        axes[1].set_title(f'Detected Points ({len(points)})')

        fig.canvas.draw()

    slider_sigma.on_changed(update)
    slider_thresh.on_changed(update)
    update(None)

    plt.show()


def exercise1b():
    image_bgr = cv2.imread('data/graf/graf_a.jpg')
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray.astype(np.float64) / 255.0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle('Exercise 1b: Harris Feature Point Detector')
    plt.subplots_adjust(bottom=0.3)

    initial_sigma = 1.0
    initial_thresh = 0.1
    initial_alpha = 0.06

    im1 = axes[0].imshow(np.zeros_like(image_gray), cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(image_rgb)
    points_plot, = axes[1].plot([], [], 'r.', markersize=2)
    axes[1].axis('off')

    ax_sigma = plt.axes([0.2, 0.15, 0.5, 0.03])
    ax_thresh = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_alpha = plt.axes([0.2, 0.05, 0.5, 0.03])

    slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 3.0,
                          valinit=initial_sigma, valfmt='%.1f')
    slider_thresh = Slider(ax_thresh, 'Threshold', 0.01,
                           0.5, valinit=initial_thresh, valfmt='%.2f')
    slider_alpha = Slider(ax_alpha, 'Alpha', 0.01,
                          0.2, valinit=initial_alpha, valfmt='%.3f')

    def update(_):
        sigma = slider_sigma.val
        thresh = slider_thresh.val
        alpha = slider_alpha.val

        points, harris_response = harris_points(
            image_gray, sigma=sigma, thresh=thresh, alpha=alpha)

        im1.set_array(harris_response)
        im1.set_clim(vmin=harris_response.min(), vmax=harris_response.max())
        axes[0].set_title(f'Harris Response')

        y_coords, x_coords = zip(*points)
        points_plot.set_data(x_coords, y_coords)

        axes[1].set_title(f'Detected Points ({len(points)})')

        fig.canvas.draw()

    slider_sigma.on_changed(update)
    slider_thresh.on_changed(update)
    slider_alpha.on_changed(update)
    update(None)

    plt.show()


def hellinger_distance(h1, h2):
    return np.sqrt(0.5 * np.sum((np.sqrt(h1) - np.sqrt(h2))**2))


def find_correspondences(descriptors1, descriptors2):
    correspondences = []

    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        best_distance = float('inf')

        for j, desc2 in enumerate(descriptors2):
            distance = hellinger_distance(desc1, desc2)
            if distance < best_distance:
                best_distance = distance
                best_match_idx = j

        correspondences.append([i, best_match_idx])

    return correspondences


def exercise2a():
    image1_bgr = cv2.imread('data/graf/graf_a_small.jpg')
    image2_bgr = cv2.imread('data/graf/graf_b_small.jpg')

    image1_gray = cv2.cvtColor(
        image1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    image2_gray = cv2.cvtColor(
        image2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    plt.suptitle('Exercise 2a: Feature Point Correspondences')
    plt.subplots_adjust(bottom=0.25)

    initial_sigma = 1.0
    initial_thresh = 0.1

    ax_sigma = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_thresh = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 3.0,
                          valinit=initial_sigma, valfmt='%.1f')
    slider_thresh = Slider(ax_thresh, 'Threshold', 0.01,
                           0.5, valinit=initial_thresh, valfmt='%.2f')

    def update(_):
        sigma = slider_sigma.val
        thresh = slider_thresh.val

        points1, _ = harris_points(image1_gray, sigma=sigma, thresh=thresh)
        points2, _ = harris_points(image2_gray, sigma=sigma, thresh=thresh)

        if len(points1) == 0 or len(points2) == 0:
            ax.clear()
            ax.text(0.5, 0.5, 'No feature points detected',
                    ha='center', va='center', transform=ax.transAxes)
            fig.canvas.draw()
            return

        y1_coords, x1_coords = zip(*points1)
        y2_coords, x2_coords = zip(*points2)

        descriptors1 = simple_descriptors(
            image1_gray, y1_coords, x1_coords, n_bins=16, window_size=20)
        descriptors2 = simple_descriptors(
            image2_gray, y2_coords, x2_coords, n_bins=16, window_size=20)

        correspondences = find_correspondences(descriptors1, descriptors2)

        pts1 = np.array([[x, y] for y, x in points1])
        pts2 = np.array([[x, y] for y, x in points2])

        ax.clear()
        I = np.hstack((image1_rgb, image2_rgb))
        w = image1_rgb.shape[1]
        ax.imshow(I)

        for i, j in correspondences:
            p1 = pts1[i]
            p2 = pts2[j]
            clr = np.random.rand(3,)
            ax.plot(p1[0], p1[1], color=clr, marker='.', markersize=8)
            ax.plot(p2[0]+w, p2[1], color=clr, marker='.', markersize=8)
            ax.plot([p1[0], p2[0]+w], [p1[1], p2[1]], color=clr, linewidth=1.5)

        ax.set_title(f'Correspondences ({len(correspondences)})')
        ax.axis('off')
        fig.canvas.draw()

    slider_sigma.on_changed(update)
    slider_thresh.on_changed(update)
    update(None)

    plt.show()


def find_matches(image1, image2, sigma=1.0, thresh=0.1, alpha=0.06):
    points1, _ = harris_points(image1, sigma=sigma, thresh=thresh, alpha=alpha)
    points2, _ = harris_points(image2, sigma=sigma, thresh=thresh, alpha=alpha)

    if len(points1) == 0 or len(points2) == 0:
        return [], points1, points2

    y1_coords, x1_coords = zip(*points1)
    y2_coords, x2_coords = zip(*points2)

    descriptors1 = simple_descriptors(
        image1, y1_coords, x1_coords, n_bins=16, window_size=20)
    descriptors2 = simple_descriptors(
        image2, y2_coords, x2_coords, n_bins=16, window_size=20)

    matches_1to2 = find_correspondences(descriptors1, descriptors2)
    matches_2to1 = find_correspondences(descriptors2, descriptors1)

    symmetric_matches = []

    for i, j in matches_1to2:
        reverse_match = matches_2to1[j]
        if reverse_match[1] == i:
            symmetric_matches.append([i, j])

    return symmetric_matches, points1, points2


def exercise2b():
    image1_bgr = cv2.imread('data/graf/graf_a_small.jpg')
    image2_bgr = cv2.imread('data/graf/graf_b_small.jpg')

    image1_gray = cv2.cvtColor(
        image1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    image2_gray = cv2.cvtColor(
        image2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    plt.suptitle('Exercise 2b: Symmetric Feature Point Matches')
    plt.subplots_adjust(bottom=0.25)

    initial_sigma = 1.0
    initial_thresh = 0.1

    ax_sigma = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_thresh = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 3.0,
                          valinit=initial_sigma, valfmt='%.1f')
    slider_thresh = Slider(ax_thresh, 'Threshold', 0.01,
                           0.5, valinit=initial_thresh, valfmt='%.2f')

    def update(_):
        sigma = slider_sigma.val
        thresh = slider_thresh.val

        symmetric_matches, points1, points2 = find_matches(
            image1_gray, image2_gray, sigma=sigma, thresh=thresh)

        if len(symmetric_matches) == 0:
            ax.clear()
            ax.text(0.5, 0.5, 'No symmetric matches found',
                    ha='center', va='center', transform=ax.transAxes)
            fig.canvas.draw()
            return

        pts1 = np.array([[x, y] for y, x in points1])
        pts2 = np.array([[x, y] for y, x in points2])

        ax.clear()
        I = np.hstack((image1_rgb, image2_rgb))
        w = image1_rgb.shape[1]
        ax.imshow(I)

        for i, j in symmetric_matches:
            p1 = pts1[i]
            p2 = pts2[j]
            clr = np.random.rand(3,)
            ax.plot(p1[0], p1[1], color=clr, marker='.', markersize=8)
            ax.plot(p2[0]+w, p2[1], color=clr, marker='.', markersize=8)
            ax.plot([p1[0], p2[0]+w], [p1[1], p2[1]], color=clr, linewidth=1.5)

        ax.set_title(f'Symmetric Matches ({len(symmetric_matches)})')
        ax.axis('off')
        fig.canvas.draw()

    slider_sigma.on_changed(update)
    slider_thresh.on_changed(update)
    update(None)

    plt.show()


def estimate_homography(points1, points2):
    n = len(points1)
    if n < 4:
        raise ValueError("At least 4 point correspondences are required")

    A = np.zeros((2 * n, 9))

    for i in range(n):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[2*i] = [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2]
        A[2*i + 1] = [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]

    U, S, VT = np.linalg.svd(A)
    h = VT[-1]
    H = h.reshape(3, 3)

    if H[2, 2] != 0:
        H = H / H[2, 2]

    return H


def exercise3a():
    img1 = cv2.imread('data/newyork/newyork_a.jpg')
    img2 = cv2.imread('data/newyork/newyork_b.jpg')
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    correspondences = np.loadtxt('data/newyork/newyork.txt')
    points1 = correspondences[:, :2]
    points2 = correspondences[:, 2:]

    H_estimated = estimate_homography(points1, points2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Exercise 3a: Basic Homography Estimation - New York Dataset')

    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')

    img1_warped = cv2.warpPerspective(img1, H_estimated, (img2.shape[1], img2.shape[0]))
    img1_warped_rgb = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB)

    axes[1, 0].imshow(img1_warped_rgb)
    axes[1, 0].set_title('Image 1 Warped')
    axes[1, 0].axis('off')

    overlay = cv2.addWeighted(img1_warped, 0.5, img2, 0.5, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    img1_graf = cv2.imread('data/graf/graf_a.jpg')
    img2_graf = cv2.imread('data/graf/graf_b.jpg')
    img1_graf_rgb = cv2.cvtColor(img1_graf, cv2.COLOR_BGR2RGB)
    img2_graf_rgb = cv2.cvtColor(img2_graf, cv2.COLOR_BGR2RGB)

    correspondences_graf = np.loadtxt('data/graf/graf.txt')
    points1_graf = correspondences_graf[:, :2]
    points2_graf = correspondences_graf[:, 2:]

    H_graf = estimate_homography(points1_graf, points2_graf)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Exercise 3a: Basic Homography Estimation - Graf Dataset')

    axes[0, 0].imshow(img1_graf_rgb)
    axes[0, 0].set_title('Graf Image A')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_graf_rgb)
    axes[0, 1].set_title('Graf Image B')
    axes[0, 1].axis('off')

    img1_graf_warped = cv2.warpPerspective(img1_graf, H_graf, (img2_graf.shape[1], img2_graf.shape[0]))
    img1_graf_warped_rgb = cv2.cvtColor(img1_graf_warped, cv2.COLOR_BGR2RGB)

    axes[1, 0].imshow(img1_graf_warped_rgb)
    axes[1, 0].set_title('Graf A Warped')
    axes[1, 0].axis('off')

    overlay_graf = cv2.addWeighted(img1_graf_warped, 0.5, img2_graf, 0.5, 0)
    overlay_graf_rgb = cv2.cvtColor(overlay_graf, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(overlay_graf_rgb)
    axes[1, 1].set_title('Graf Overlay')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def ransac_homography(points1, points2, threshold=5.0, max_iterations=1000):
    n = len(points1)
    if n < 4:
        raise ValueError("At least 4 point correspondences are required")

    best_H = None
    best_inliers = None
    best_inlier_count = 0

    for _ in range(max_iterations):
        sample_indices = np.random.choice(n, 4, replace=False)
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]

        H = estimate_homography(sample_points1, sample_points2)

        points1_homo = np.column_stack([points1, np.ones(n)])
        points2_projected = (H @ points1_homo.T).T
        points2_projected = points2_projected[:, :2] / points2_projected[:, 2:3]

        distances = np.linalg.norm(points2 - points2_projected, axis=1)
        inliers = distances < threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_H = H
            best_inliers = inliers
            best_inlier_count = inlier_count



    return best_H, best_inliers, best_inlier_count


def exercise3c():
    img1_bgr = cv2.imread('data/graf/graf_a_small.jpg')
    img2_bgr = cv2.imread('data/graf/graf_b_small.jpg')

    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    symmetric_matches, points1_detected, points2_detected = find_matches(
        img1_gray, img2_gray, sigma=1.0, thresh=0.1)

    if len(symmetric_matches) < 4:
        return

    points1_matched = np.array([[x, y] for y, x in [points1_detected[i] for i, j in symmetric_matches]])
    points2_matched = np.array([[x, y] for y, x in [points2_detected[j] for i, j in symmetric_matches]])

    H_ransac, inliers, inlier_count = ransac_homography(
        points1_matched, points2_matched, threshold=5.0, max_iterations=1000)

    H_basic = estimate_homography(points1_matched, points2_matched)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exercise 3c: RANSAC Homography Estimation - Graf Dataset')

    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Graf A')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Graf B')
    axes[0, 1].axis('off')

    I_combined = np.hstack((img1_rgb, img2_rgb))
    w = img1_rgb.shape[1]
    axes[0, 2].imshow(I_combined)

    for idx, (i, j) in enumerate(symmetric_matches):
        p1 = points1_matched[idx]
        p2 = points2_matched[idx]
        color = 'green' if inliers[idx] else 'red'
        axes[0, 2].plot(p1[0], p1[1], 'o', color=color, markersize=3)
        axes[0, 2].plot(p2[0] + w, p2[1], 'o', color=color, markersize=3)
        axes[0, 2].plot([p1[0], p2[0] + w], [p1[1], p2[1]], color=color, linewidth=1, alpha=0.7)

    axes[0, 2].set_title(f'Matches: {inlier_count}/{len(symmetric_matches)} inliers')
    axes[0, 2].axis('off')

    img1_warped_ransac = cv2.warpPerspective(img1_bgr, H_ransac, (img2_bgr.shape[1], img2_bgr.shape[0]))
    img1_warped_ransac_rgb = cv2.cvtColor(img1_warped_ransac, cv2.COLOR_BGR2RGB)

    axes[1, 0].imshow(img1_warped_ransac_rgb)
    axes[1, 0].set_title('Graf A Warped (RANSAC)')
    axes[1, 0].axis('off')

    overlay_ransac = cv2.addWeighted(img1_warped_ransac, 0.5, img2_bgr, 0.5, 0)
    overlay_ransac_rgb = cv2.cvtColor(overlay_ransac, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(overlay_ransac_rgb)
    axes[1, 1].set_title('RANSAC Overlay')
    axes[1, 1].axis('off')

    img1_warped_basic = cv2.warpPerspective(img1_bgr, H_basic, (img2_bgr.shape[1], img2_bgr.shape[0]))
    img1_warped_basic_rgb = cv2.cvtColor(img1_warped_basic, cv2.COLOR_BGR2RGB)

    overlay_basic = cv2.addWeighted(img1_warped_basic, 0.5, img2_bgr, 0.5, 0)
    overlay_basic_rgb = cv2.cvtColor(overlay_basic, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(overlay_basic_rgb)
    axes[1, 2].set_title('Basic DLT Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    img1_ny_bgr = cv2.imread('data/newyork/newyork_a.jpg')
    img2_ny_bgr = cv2.imread('data/newyork/newyork_b.jpg')

    img1_ny_gray = cv2.cvtColor(img1_ny_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    img2_ny_gray = cv2.cvtColor(img2_ny_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    img1_ny_rgb = cv2.cvtColor(img1_ny_bgr, cv2.COLOR_BGR2RGB)
    img2_ny_rgb = cv2.cvtColor(img2_ny_bgr, cv2.COLOR_BGR2RGB)

    symmetric_matches_ny, points1_ny, points2_ny = find_matches(
        img1_ny_gray, img2_ny_gray, sigma=2.0, thresh=0.05)

    if len(symmetric_matches_ny) >= 4:
        points1_ny_matched = np.array([[x, y] for y, x in [points1_ny[i] for i, j in symmetric_matches_ny]])
        points2_ny_matched = np.array([[x, y] for y, x in [points2_ny[j] for i, j in symmetric_matches_ny]])

        H_ny_ransac, inliers_ny, inlier_count_ny = ransac_homography(
            points1_ny_matched, points2_ny_matched, threshold=10.0, max_iterations=1000)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Exercise 3c: RANSAC Homography Estimation - New York Dataset')

        axes[0].imshow(img1_ny_rgb)
        axes[0].set_title('New York A')
        axes[0].axis('off')

        img1_ny_warped = cv2.warpPerspective(img1_ny_bgr, H_ny_ransac, (img2_ny_bgr.shape[1], img2_ny_bgr.shape[0]))
        img1_ny_warped_rgb = cv2.cvtColor(img1_ny_warped, cv2.COLOR_BGR2RGB)
        axes[1].imshow(img1_ny_warped_rgb)
        axes[1].set_title('New York A Warped')
        axes[1].axis('off')

        overlay_ny = cv2.addWeighted(img1_ny_warped, 0.5, img2_ny_bgr, 0.5, 0)
        overlay_ny_rgb = cv2.cvtColor(overlay_ny, cv2.COLOR_BGR2RGB)
        axes[2].imshow(overlay_ny_rgb)
        axes[2].set_title(f'Overlay ({inlier_count_ny}/{len(symmetric_matches_ny)} inliers)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # exercise1a()
    # exercise1b()
    # exercise2a()
    # exercise2b()
    exercise3a()
    exercise3c()

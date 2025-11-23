import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob
import matplotlib.colors as mcolors
from a3_utils import draw_line


def gauss(sigma):
    '''
    gauss kernel
    '''
    half_size = int(np.floor(3 * sigma))
    x = np.arange(-half_size, half_size + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussdx(sigma):
    '''
    gauss kernel derivative
    '''
    half_size = int(np.floor(3 * sigma))
    x = np.arange(-half_size, half_size + 1)

    coefficient = -1 / (np.sqrt(2 * np.pi) * sigma**3)
    kernel = coefficient * x * np.exp(-(x**2) / (2 * sigma**2))

    kernel = kernel / np.sum(np.abs(kernel))  # normalize

    return kernel


def get_kernels(sigma):
    G = gauss(sigma)
    D = gaussdx(sigma)
    G_2D = G.reshape(1, -1).astype(np.float32)
    G_2D_T = G.reshape(-1, 1).astype(np.float32)
    D_2D = D.reshape(1, -1).astype(np.float32)
    D_2D_T = D.reshape(-1, 1).astype(np.float32)

    return G, D, G_2D, G_2D_T, D_2D, D_2D_T


def exercise1b():
    sigmas = [1.0, 2.0, 3.0]

    plt.figure(figsize=(15, 5))

    for i, sigma in enumerate(sigmas):
        plt.subplot(1, 3, i + 1)

        g = gauss(sigma)
        gx = gaussdx(sigma)

        half_size = int(np.floor(3 * sigma))
        x = np.arange(-half_size, half_size + 1)

        plt.plot(x, g, 'b-', label=f'Gaussian σ={sigma}', linewidth=2)
        plt.plot(
            x, gx, 'r-', label=f'Gaussian derivative σ={sigma}', linewidth=2)
        plt.grid(True)
        plt.legend()
        plt.title(f'Gaussian and its derivative (σ={sigma})')
        plt.xlabel('x')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.show()


def exercise1c():
    '''
    Impulse response analysis
    '''
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    G, D, G_2D, G_2D_T, D_2D, D_2D_T = get_kernels(sigma=6.0)

    combinations = [
        [G_2D, G_2D_T, '(a) G * G^T'],
        [G_2D, D_2D_T, '(b) G * D^T'],
        [D_2D, G_2D_T, '(c) D * G^T'],
        [G_2D_T, D_2D, '(d) G^T * D'],
        [D_2D_T, G_2D, '(e) D^T * G']
    ]

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(impulse, cmap='gray')
    plt.title('Original Impulse')

    for i, (k1, k2, title) in enumerate(combinations):
        result = cv2.filter2D(
            impulse.astype(np.float32), -1, k1)
        result = cv2.filter2D(result, -1, k2)

        plt.subplot(2, 3, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(title)

    plt.tight_layout()
    plt.show()


def partial_derivatives(image, sigma):
    img = image.astype(np.float32)

    G, D, G_2D, G_2D_T, D_2D, D_2D_T = get_kernels(sigma)

    temp = cv2.filter2D(img, -1, G_2D_T)
    Ix = cv2.filter2D(temp, -1, D_2D)

    temp = cv2.filter2D(img, -1, G_2D)
    Iy = cv2.filter2D(temp, -1, D_2D_T)

    return Ix, Iy


def second_derivatives(image, sigma):
    Ix, Iy = partial_derivatives(image, sigma)

    G, D, G_2D, G_2D_T, D_2D, D_2D_T = get_kernels(sigma)

    temp = cv2.filter2D(Ix, -1, G_2D_T)
    Ixx = cv2.filter2D(temp, -1, D_2D)

    temp = cv2.filter2D(Iy, -1, G_2D)
    Iyy = cv2.filter2D(temp, -1, D_2D_T)

    temp = cv2.filter2D(Ix, -1, G_2D)
    Ixy = cv2.filter2D(temp, -1, D_2D_T)

    return Ixx, Iyy, Ixy


def gradient_magnitude(image, sigma):
    Ix, Iy = partial_derivatives(image, sigma)

    magnitude = np.sqrt(Ix**2 + Iy**2)
    angles = np.arctan2(Iy, Ix)

    return magnitude, angles


def exercise1d():
    '''
    Test derivative functions on museum image
    '''
    museum = cv2.imread('./images/museum.jpg', cv2.IMREAD_GRAYSCALE)
    museum = museum.astype(np.float32) / 255.0

    sigma = 1.0

    Ix, Iy = partial_derivatives(museum, sigma)
    Ixx, Iyy, Ixy = second_derivatives(museum, sigma)
    magnitude, angles = gradient_magnitude(museum, sigma)

    hsv = np.zeros((museum.shape[0], museum.shape[1], 3))
    hsv[:, :, 0] = (angles + np.pi) / (2 * np.pi)  # hue
    hsv[:, :, 1] = 1  # saturation
    hsv[:, :, 2] = magnitude / np.max(magnitude)  # value

    rgb_directions = mcolors.hsv_to_rgb(hsv)

    plt.figure(figsize=(15, 15))

    for i, (img, title, cmap) in enumerate([
        [museum, 'I (Original)', 'gray'],
        [Ix, 'Ix', 'gray'],
        [Iy, 'Iy', 'gray'],
        [Ixx, 'Ixx', 'gray'],
        [Iyy, 'Iyy', 'gray'],
        [Ixy, 'Ixy', 'gray'],
        [magnitude, 'Imag (Magnitude)', 'gray'],
        [angles, 'Idir (Directions)', 'gray'],
        [rgb_directions, 'Idir (HSV)', None]
    ]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def findedges(image, sigma, theta):
    magnitude, angles = gradient_magnitude(image, sigma)
    edges = np.where(magnitude >= theta, magnitude, 0)

    return edges


def nonmaxima_suppression(magnitude, angles):
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle_deg = np.rad2deg(angles) % 180  # rad -> deg & normalize

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if magnitude[i, j] == 0:
                continue

            angle = angle_deg[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]

            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def exercise2a():
    '''
    Edge detection with different threshold values
    '''
    museum = cv2.imread('./images/museum.jpg', cv2.IMREAD_GRAYSCALE)
    museum = museum.astype(np.float32) / 255.0

    sigma = 0.5
    theta_values = [0.1, 0.2, 0.3]

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(museum, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for i, theta in enumerate(theta_values):
        edges = findedges(museum, sigma, theta)

        plt.subplot(2, 2, i + 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Edges (thr={theta})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def exercise2b():
    '''
    Non-maxima suppression comparison
    '''
    museum = cv2.imread('./images/museum.jpg', cv2.IMREAD_GRAYSCALE)
    museum = museum.astype(np.float32) / 255.0

    sigma = 1.0
    theta_values = [0.1, 0.2, 0.3]

    magnitude, angles = gradient_magnitude(museum, sigma)

    plt.figure(figsize=(15, 10))

    for i, theta in enumerate(theta_values):
        edges = np.where(magnitude >= theta, magnitude, 0)

        edges_nma = nonmaxima_suppression(magnitude, angles)
        edges_nma = np.where(edges_nma >= theta, edges_nma, 0)

        plt.subplot(2, 3, i + 1)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Thresholded (thr={theta})')
        plt.axis('off')

        plt.subplot(2, 3, i + 4)
        plt.imshow(edges_nma, cmap='gray')
        plt.title(f'Nonmax. supp. (thr={theta})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def hysteresis_thresholding(edges, t_low, t_high):
    strong_edges = edges >= t_high
    result = strong_edges.astype(np.uint8)
    all_edges = (edges >= t_low).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(all_edges)

    for label in range(1, num_labels):
        component_mask = (labels == label)

        # keep components that have strong edges
        if np.any(strong_edges & component_mask):
            result = result | component_mask

    return result.astype(np.float32)


def exercise2c():
    '''
    Canny edge
    '''
    museum = cv2.imread('./images/museum.jpg', cv2.IMREAD_GRAYSCALE)
    museum = museum.astype(np.float32) / 255.0

    sigma = 1.0
    threshold = 0.16
    t_high = 0.16
    t_low = 0.04

    magnitude, angles = gradient_magnitude(museum, sigma)
    nms_result = nonmaxima_suppression(magnitude, angles)

    steps = [
        [museum, 'Original'],
        [np.where(magnitude >= threshold, magnitude, 0), 'Thresholded'],
        [np.where(nms_result >= threshold, nms_result, 0), 'NMS'],
        [hysteresis_thresholding(nms_result, t_low, t_high), 'Hysteresis']
    ]

    plt.figure(figsize=(12, 12))

    for i, (image, label) in enumerate(steps):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_hough_params(h, w, bins_theta, bins_rho):
    theta_range = np.linspace(-np.pi/2, np.pi/2, bins_theta)

    diagonal = np.sqrt(h**2 + w**2)
    rho_range = np.linspace(-diagonal, diagonal, bins_rho)

    return theta_range, rho_range


def hough_single_point(point, h, w, bins_theta, bins_rho):
    x, y = point
    theta_range, rho_range = get_hough_params(h, w, bins_theta, bins_rho)
    accumulator = np.zeros((bins_rho, bins_theta))

    for i, theta in enumerate(theta_range):
        rho = x * np.cos(theta) + y * np.sin(theta)

        rho_idx = np.argmin(np.abs(rho_range - rho))
        accumulator[rho_idx, i] += 1

    return accumulator, theta_range, rho_range


def hough_find_lines(binary_image, bins_theta, bins_rho):
    h, w = binary_image.shape
    theta_range, rho_range = get_hough_params(h, w, bins_theta, bins_rho)

    accumulator = np.zeros((bins_rho, bins_theta))

    edge_pixels = np.where(binary_image > 0)

    for row, col in zip(edge_pixels[0], edge_pixels[1]):
        # Convert image array indices to (x,y) coordinates
        x, y = col, row
        for i, theta in enumerate(theta_range):
            rho = x * np.cos(theta) + y * np.sin(theta)

            rho_idx = np.argmin(np.abs(rho_range - rho))
            accumulator[rho_idx, i] += 1

    return accumulator, theta_range, rho_range


def exercise3a():
    """
    Single point Hough transform
    """
    point = (50, 90)
    h, w = 100, 100
    bins_theta = 180
    bins_rho = 200

    accumulator, theta_range, rho_range = hough_single_point(
        point, h, w, bins_theta, bins_rho)

    plt.figure(figsize=(12, 5))

    img = np.zeros((h, w))
    img[point[1], point[0]] = 1

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', extent=[0, w, h, 0])

    for i in range(0, len(theta_range), 1):
        theta = theta_range[i]
        rho = point[0] * np.cos(theta) + point[1] * np.sin(theta)
        draw_line(rho, theta, h, w, clr='g', linewidth=0.2)

    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.imshow(accumulator, cmap='hot', aspect='auto', origin='upper',
               extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]])
    plt.xlabel('θ (radians)')
    plt.ylabel('ρ (pixels)')

    plt.tight_layout()
    plt.show()


def exercise3b():
    """
    Synthetic image Hough transform
    """
    bins_theta = 180
    bins_rho = 200

    synthetic = np.zeros((100, 100))
    synthetic[10, 10] = 1
    synthetic[10, 20] = 1

    oneline = cv2.imread('./images/oneline.png', cv2.IMREAD_GRAYSCALE)
    rectangle = cv2.imread('./images/rectangle.png', cv2.IMREAD_GRAYSCALE)

    oneline_edges = np.where(oneline > 128, 1, 0)
    rectangle_edges = np.where(rectangle > 128, 1, 0)

    images_data = [
        [synthetic, 'synthetic'],
        [oneline_edges, 'oneline.png'],
        [rectangle_edges, 'rectangle.png']
    ]

    plt.figure(figsize=(15, 5))

    for i, (image, title) in enumerate(images_data):
        accumulator, theta_range, rho_range = hough_find_lines(
            image, bins_theta, bins_rho)

        plt.subplot(1, 3, i + 1)
        plt.imshow(accumulator, cmap='viridis', aspect='auto', origin='upper',
                   extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]])
        plt.xlabel('θ (radians)')
        plt.ylabel('ρ (pixels)')
        plt.title(title)

    plt.tight_layout()
    plt.show()


def nonmaxima_suppression_box(accumulator, k):
    h, w = accumulator.shape
    suppressed = np.copy(accumulator)

    nonzero_indices = np.nonzero(accumulator)

    for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
        i_min = max(0, i - k//2)
        i_max = min(h, i + k//2 + 1)
        j_min = max(0, j - k//2)
        j_max = min(w, j + k//2 + 1)

        neighborhood = accumulator[i_min:i_max, j_min:j_max]

        max_val = np.max(neighborhood)
        if accumulator[i, j] < max_val:
            suppressed[i, j] = 0
        elif accumulator[i, j] == max_val:
            max_positions = np.where(neighborhood == max_val)

            global_i = max_positions[0] + i_min
            global_j = max_positions[1] + j_min

            keep_i = global_i[0]
            keep_j = global_j[0]

            if i != keep_i or j != keep_j:
                suppressed[i, j] = 0

    return suppressed


def exercise3c():
    """
    Non-maxima suppression
    """
    bins_theta = 180
    bins_rho = 200
    k = 5

    rectangle = cv2.imread('./images/rectangle.png', cv2.IMREAD_GRAYSCALE)
    rectangle_edges = np.where(rectangle > 128, 1, 0)

    accumulator, theta_range, rho_range = hough_find_lines(
        rectangle_edges, bins_theta, bins_rho)

    suppressed_accumulator = nonmaxima_suppression_box(accumulator, k)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(rectangle_edges, cmap='gray')
    plt.title('Edge Image (Rectangle)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(accumulator, cmap='viridis', aspect='auto', origin='upper',
               extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]])
    plt.xlabel('θ (radians)')
    plt.ylabel('ρ (pixels)')
    plt.title('Original Accumulator')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(suppressed_accumulator, cmap='viridis', aspect='auto', origin='upper',
               extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]])
    plt.xlabel('θ (radians)')
    plt.ylabel('ρ (pixels)')
    plt.title(f'After Non-maxima Suppression (k={k})')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def extract_lines_from_accumulator(accumulator, theta_range, rho_range, threshold=None, top_n=None):
    """
    Extract parameter pairs (rho, theta) from accumulator above threshold or top N lines.
    Similar to MATLAB hough_draw_lines function (lines 20-27).
    """
    rho_list = []
    theta_list = []

    if top_n is not None:
        # Select top N strongest lines (similar to hough_draw_lines_top)
        flat_acc = accumulator.flatten()
        sorted_indices = np.argsort(flat_acc)[::-1]  # Sort in descending order

        # Take only top_n non-zero values
        count = 0
        for idx in sorted_indices:
            if count >= top_n:
                break
            if flat_acc[idx] > 0:  # Only consider non-zero accumulator values
                i, j = np.unravel_index(idx, accumulator.shape)
                rho_list.append(rho_range[i])
                theta_list.append(theta_range[j])
                count += 1
    else:
        # Use threshold approach
        max_val = np.max(accumulator)
        threshold_val = max_val * threshold

        # Find all accumulator cells above threshold
        for i in range(accumulator.shape[0]):  # rho dimension
            for j in range(accumulator.shape[1]):  # theta dimension
                if accumulator[i, j] >= threshold_val:
                    rho_list.append(rho_range[i])
                    theta_list.append(theta_range[j])

    return rho_list, theta_list


def findedges_modified(image, sigma, theta):
    """
    Edge detection with non-maxima suppression, similar to MATLAB findedges_modified.
    """
    # Add padding to handle border effects
    pad_size = 20
    padded_image = np.pad(image, pad_size, mode='symmetric')

    # Apply gradient magnitude and non-maxima suppression
    magnitude, angles = gradient_magnitude(padded_image, sigma)
    suppressed = nonmaxima_suppression(magnitude, angles)

    # Apply threshold
    edges = np.where(suppressed >= theta, 1, 0)

    # Remove padding
    edges = edges[pad_size:-pad_size, pad_size:-pad_size]

    return edges


def exercise3d():
    """
    Exercise 3D: Search parameter space and extract parameter pairs (rho, theta)
    whose corresponding accumulator cell value is greater than a specified threshold.
    Draw the lines using draw_line() function.
    """
    plt.figure(figsize=(15, 5))

    # 1. Synthetic image (from exercise3b)
    synthetic = np.zeros((100, 100))
    synthetic[10, 10] = 1
    synthetic[10, 20] = 1

    # Apply Hough transform
    bins_theta = synthetic.shape[1]  # Use image dimensions as in MATLAB
    bins_rho = synthetic.shape[0]
    accumulator, theta_range, rho_range = hough_find_lines(synthetic, bins_theta, bins_rho)

    # Apply non-maxima suppression
    suppressed_acc = nonmaxima_suppression_box(accumulator, k=5)

    # Extract lines above threshold and draw
    rho_list, theta_list = extract_lines_from_accumulator(suppressed_acc, theta_range, rho_range, 0.5)

    plt.subplot(1, 3, 1)
    plt.imshow(synthetic, cmap='gray')
    plt.title('Synthetic Image')

    # Draw detected lines
    for rho, theta in zip(rho_list, theta_list):
        draw_line(rho, theta, synthetic.shape[0], synthetic.shape[1], clr='r', linewidth=1.0)

    plt.axis('off')

    # 2. Oneline image
    oneline = cv2.imread('./images/oneline.png', cv2.IMREAD_GRAYSCALE)
    oneline = oneline.astype(np.float32) / 255.0

    # Simple edge detection using gradient for black-to-white transitions
    # Calculate gradients to find edges
    grad_x = np.abs(np.diff(oneline, axis=1))  # Horizontal edges
    grad_y = np.abs(np.diff(oneline, axis=0))  # Vertical edges

    # Create edge image
    line_edges = np.zeros_like(oneline)
    line_edges[:, :-1] += grad_x > 0.3  # Vertical edges (black to white transition)
    line_edges[:-1, :] += grad_y > 0.3  # Horizontal edges
    line_edges = np.where(line_edges > 0, 1, 0)

    print(f"Oneline edges: max={np.max(line_edges)}, non-zero count={np.count_nonzero(line_edges)}")

    # Hough transform
    bins_rho = oneline.shape[0]
    bins_theta = oneline.shape[1]
    line_accumulator, theta_range, rho_range = hough_find_lines(line_edges, bins_theta, bins_rho)

    # Debug accumulator values
    print(f"Oneline: Original accumulator max: {np.max(line_accumulator)}, non-zero count: {np.count_nonzero(line_accumulator)}")

    # Non-maxima suppression
    line_suppressed = nonmaxima_suppression_box(line_accumulator, k=5)
    print(f"Oneline: After NMS max: {np.max(line_suppressed)}, non-zero count: {np.count_nonzero(line_suppressed)}")

    # Extract and draw lines (use original accumulator if NMS fails)
    if np.max(line_suppressed) > 0:
        rho_list, theta_list = extract_lines_from_accumulator(line_suppressed, theta_range, rho_range, threshold=0.8)
    else:
        rho_list, theta_list = extract_lines_from_accumulator(line_accumulator, theta_range, rho_range, threshold=0.8)
    print(f"Oneline: Found {len(rho_list)} lines")

    plt.subplot(1, 3, 2)
    plt.imshow(oneline, cmap='gray')
    plt.title('Oneline Image')

    for rho, theta in zip(rho_list, theta_list):
        draw_line(rho, theta, oneline.shape[0], oneline.shape[1], clr='r', linewidth=1.0)

    plt.axis('off')

    # 3. Rectangle image
    rectangle = cv2.imread('./images/rectangle.png', cv2.IMREAD_GRAYSCALE)
    rectangle = rectangle.astype(np.float32) / 255.0

    # Simple edge detection using gradient for rectangle edges
    # Calculate gradients to find edges
    grad_x = np.abs(np.diff(rectangle, axis=1))  # Horizontal edges
    grad_y = np.abs(np.diff(rectangle, axis=0))  # Vertical edges

    # Create edge image
    rect_edges = np.zeros_like(rectangle)
    rect_edges[:, :-1] += grad_x > 0.3  # Vertical edges (rectangle sides)
    rect_edges[:-1, :] += grad_y > 0.3  # Horizontal edges (rectangle top/bottom)
    rect_edges = np.where(rect_edges > 0, 1, 0)

    print(f"Rectangle edges: max={np.max(rect_edges)}, non-zero count={np.count_nonzero(rect_edges)}")

    # Hough transform
    bins_rho = rectangle.shape[0]
    bins_theta = rectangle.shape[1]
    rect_accumulator, theta_range, rho_range = hough_find_lines(rect_edges, bins_theta, bins_rho)

    # Debug accumulator values
    print(f"Rectangle: Original accumulator max: {np.max(rect_accumulator)}, non-zero count: {np.count_nonzero(rect_accumulator)}")

    # Non-maxima suppression
    rect_suppressed = nonmaxima_suppression_box(rect_accumulator, k=5)
    print(f"Rectangle: After NMS max: {np.max(rect_suppressed)}, non-zero count: {np.count_nonzero(rect_suppressed)}")

    # Extract and draw lines (use original accumulator if NMS fails)
    if np.max(rect_suppressed) > 0:
        rho_list, theta_list = extract_lines_from_accumulator(rect_suppressed, theta_range, rho_range, threshold=0.6)
    else:
        rho_list, theta_list = extract_lines_from_accumulator(rect_accumulator, theta_range, rho_range, threshold=0.6)
    print(f"Rectangle: Found {len(rho_list)} lines")

    plt.subplot(1, 3, 3)
    plt.imshow(rectangle, cmap='gray')
    plt.title('Rectangle Image')

    for rho, theta in zip(rho_list, theta_list):
        draw_line(rho, theta, rectangle.shape[0], rectangle.shape[1], clr='r', linewidth=1.0)

    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Exercise 1
    # Exercise 1a: solved on my ipad
    # exercise1b()
    # exercise1c()
    # exercise1d()
    #
    # # Exercise 2
    # exercise2a()
    # exercise2b()
    # exercise2c()
    #
    # # Exercise 3
    # exercise3a()
    # exercise3b()
    # exercise3c()
    exercise3d()


if __name__ == "__main__":
    main()

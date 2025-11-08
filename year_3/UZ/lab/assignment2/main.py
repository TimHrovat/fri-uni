import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import read_data, gauss_noise, sp_noise
import os
import glob


def simple_convolution(signal, kernel):
    k_flipped = kernel[::-1]

    N = len(kernel) // 2

    result_length = len(signal) - 2 * N
    result = np.zeros(result_length)

    for i in range(N, len(signal) - N):
        conv_sum = 0
        for j in range(len(kernel)):
            conv_sum += k_flipped[j] * signal[i - j]
        result[i - N] = conv_sum

    return result


def exercise_1b():
    signal = read_data('./signal.txt')
    kernel = read_data('./kernel.txt')

    conv_result = simple_convolution(signal, kernel)

    signal_2d = signal.reshape(1, -1).astype(np.float32)
    kernel_2d = kernel.reshape(1, -1).astype(np.float32)
    cv2_result = cv2.filter2D(signal_2d, -1, kernel_2d)
    cv2_result_1d = cv2_result.flatten()

    N = len(kernel) // 2

    cv2_valid = cv2_result_1d[N:len(signal)-N]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(signal, 'b-', linewidth=2)
    plt.title('Original Signal')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(kernel, 'r-', linewidth=2, marker='o')
    plt.title('Kernel')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    valid_indices = np.arange(N, len(signal) - N)
    plt.plot(valid_indices, conv_result, 'g-',
             linewidth=2, label='Manual Convolution')
    plt.plot(valid_indices, cv2_valid, 'r--',
             linewidth=2, alpha=0.7, label='cv2.filter2D')
    plt.title('Convolution Result')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def simple_convolution_improved(signal, kernel):
    k_flipped = kernel[::-1]
    N = len(kernel) // 2

    padded_signal = np.pad(signal, N, mode='constant', constant_values=0)
    result = np.zeros(len(signal))

    for i in range(len(signal)):
        conv_sum = 0
        for j in range(len(kernel)):
            conv_sum += k_flipped[j] * padded_signal[i + N - j]
        result[i] = conv_sum

    return result


def exercise_1c():
    signal = read_data('./signal.txt')
    kernel = read_data('./kernel.txt')

    conv_result_original = simple_convolution(signal, kernel)

    conv_result_improved = simple_convolution_improved(signal, kernel)

    signal_2d = signal.reshape(1, -1).astype(np.float32)
    kernel_2d = kernel.reshape(1, -1).astype(np.float32)
    cv2_result = cv2.filter2D(signal_2d, -1, kernel_2d)
    cv2_result_1d = cv2_result.flatten()

    N = len(kernel) // 2

    plt.figure(figsize=(15, 10))

    plt.subplot(4, 1, 1)
    plt.plot(signal, 'b-', linewidth=2)
    plt.title('Original Signal')
    plt.grid(True)

    plt.plot(kernel, 'r-', linewidth=2, marker='o')
    plt.title('Kernel')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    valid_indices = np.arange(N, len(signal) - N)
    plt.plot(valid_indices, conv_result_original, 'g-',
             linewidth=2, label='Original Convolution (shorter)')
    plt.title('Original Convolution Result (No Padding)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(range(len(signal)), conv_result_improved, 'purple',
             linewidth=2, label='Improved Convolution (with padding)')
    plt.plot(range(len(signal)), cv2_result_1d, 'r--',
             linewidth=2, alpha=0.7, label='cv2.filter2D')
    plt.title('Improved Convolution Result (With Zero Padding)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def gauss(sigma):
    half_size = int(np.floor(3 * sigma))

    x = np.arange(-half_size, half_size + 1)

    kernel = np.exp(-(x**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    return kernel


def exercise_1d():
    sigma_values = [0.5, 1, 2, 3, 4]

    plt.figure(figsize=(12, 8))

    for i, sigma in enumerate(sigma_values):
        kernel = gauss(sigma)

        half_size = len(kernel) // 2
        x_coords = np.arange(-half_size, half_size + 1)

        plt.subplot(len(sigma_values), 1, i + 1)
        plt.plot(x_coords, kernel, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title(f'Gaussian Kernel (σ = {sigma})')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def exercise_1e():
    signal = read_data('./signal.txt')

    k1 = gauss(2)
    k2 = np.array([0.1, 0.6, 0.4])

    # (s * k1) * k2
    s_conv_k1 = simple_convolution_improved(signal, k1)
    result1 = simple_convolution_improved(s_conv_k1, k2)

    # (s * k2) * k1
    s_conv_k2 = simple_convolution_improved(signal, k2)
    result2 = simple_convolution_improved(s_conv_k2, k1)

    # k3 = k1 * k2
    k3 = np.convolve(k1, k2, mode='full')
    result3 = simple_convolution_improved(signal, k3)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(signal, 'b-', linewidth=2)
    plt.title('S', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(signal)-1)

    plt.subplot(2, 2, 2)
    plt.plot(result1, 'g-', linewidth=2)
    plt.title('(S * k1) * k2', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(result1)-1)

    plt.subplot(2, 2, 3)
    plt.plot(result2, 'r-', linewidth=2)
    plt.title('(S * k2) * k1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(result2)-1)

    plt.subplot(2, 2, 4)
    plt.plot(result3, 'm-', linewidth=2)
    plt.title('S * (k1 * k2)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(result3)-1)

    plt.tight_layout()
    plt.show()


def gaussfilter(image, sigma):
    kernel_1d = gauss(sigma)
    kernel_horizontal = np.expand_dims(kernel_1d, axis=0).astype(np.float32)
    temp_result = cv2.filter2D(image.astype(np.float32), -1, kernel_horizontal)
    kernel_vertical = kernel_horizontal.T
    result = cv2.filter2D(temp_result, -1, kernel_vertical)

    return result


def exercise_2a():
    lena_color = cv2.imread('./images/lena.png')
    lena_gray = cv2.cvtColor(lena_color, cv2.COLOR_BGR2GRAY)

    lena_normalized = lena_gray.astype(np.float64) / 255.0

    lena_gauss_noise = gauss_noise(lena_normalized, magnitude=0.1)
    lena_sp_noise = sp_noise(lena_normalized, percent=0.1)

    sigma = 2.0
    lena_filtered_gauss = gaussfilter(lena_gauss_noise, sigma)
    lena_filtered_sp = gaussfilter(lena_sp_noise, sigma)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(lena_normalized, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(lena_gauss_noise, cmap='gray')
    plt.title('Gaussian noise')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(lena_sp_noise, cmap='gray')
    plt.title('Salt-and-pepper noise')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(lena_filtered_gauss, cmap='gray')
    plt.title('Filtered Gaussian noise')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(lena_filtered_sp, cmap='gray')
    plt.title('Filtered salt-and-pepper noise')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def exercise_2b():
    fox_color = cv2.imread('./images/fox.jpg')
    fox_gray = cv2.cvtColor(fox_color, cv2.COLOR_BGR2GRAY)
    fox_normalized = fox_gray.astype(np.float32) / 255.0

    sharpening_kernel = np.array([[0, -1,  0],
                                  [-1,  5, -1],
                                  [0, -1,  0]], dtype=np.float32)

    fox_sharpened = cv2.filter2D(fox_normalized, -1, sharpening_kernel)
    fox_sharpened = np.clip(fox_sharpened, 0, 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(fox_normalized, cmap='gray')
    plt.title('Original Fox Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fox_sharpened, cmap='gray')
    plt.title('Sharpened Fox Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def simple_median(signal, w):
    half_w = w // 2
    result = np.zeros_like(signal)
    padded_signal = np.pad(signal, half_w, mode='edge')

    for i in range(len(signal)):
        window = padded_signal[i:i + w]
        result[i] = np.median(window)

    return result


def exercise_2c():
    t = np.linspace(0, 4*np.pi, 100)
    clean_signal = np.sin(t) + 0.5 * np.sin(3*t)

    noisy_signal = clean_signal.copy()
    noise_indices = np.random.choice(len(clean_signal), size=int(
        0.1 * len(clean_signal)), replace=False)
    noisy_signal[noise_indices] = np.random.choice(
        [-2, 2], size=len(noise_indices))

    window_sizes = [3, 5, 9]
    gaussian_sigma = 1.0

    gaussian_kernel = gauss(gaussian_sigma)
    gaussian_filtered = simple_convolution_improved(
        noisy_signal, gaussian_kernel)

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 2, 1)
    plt.plot(clean_signal, 'b-', linewidth=2, label='Clean signal')
    plt.plot(noisy_signal, 'r-', alpha=0.7, label='Noisy signal')
    plt.title('Original vs Noisy Signal')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(gaussian_filtered, 'g-', linewidth=2,
             label=f'Gaussian (σ={gaussian_sigma})')
    plt.plot(clean_signal, 'b--', alpha=0.7, label='Clean signal')
    plt.title('Gaussian Filter Result')
    plt.legend()
    plt.grid(True)

    for i, w in enumerate(window_sizes):
        median_filtered = simple_median(noisy_signal, w)

        plt.subplot(3, 2, 3 + i)
        plt.plot(median_filtered, 'purple',
                 linewidth=2, label=f'Median (w={w})')
        plt.plot(clean_signal, 'b--', alpha=0.7, label='Clean signal')
        plt.title(f'Median Filter (window={w})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def myhist3(image, n_bins):
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    H = np.zeros((n_bins, n_bins, n_bins))

    height, width = rgb_image.shape[:2]

    r_indices = np.floor(rgb_image[:, :, 0].astype(
        np.float64) * n_bins / 256.0).astype(int)
    g_indices = np.floor(rgb_image[:, :, 1].astype(
        np.float64) * n_bins / 256.0).astype(int)
    b_indices = np.floor(rgb_image[:, :, 2].astype(
        np.float64) * n_bins / 256.0).astype(int)

    r_indices = np.clip(r_indices, 0, n_bins - 1)
    g_indices = np.clip(g_indices, 0, n_bins - 1)
    b_indices = np.clip(b_indices, 0, n_bins - 1)

    for i in range(height):
        for j in range(width):
            H[r_indices[i, j], g_indices[i, j], b_indices[i, j]] += 1

    total_pixels = height * width
    H = H.astype(np.float64) / total_pixels

    return H


def compare_histograms(h1, h2, method):
    h1_flat = h1.flatten()
    h2_flat = h2.flatten()

    if method == 'L2':
        return np.sqrt(np.sum((h1_flat - h2_flat) ** 2))

    elif method == 'chi2':
        epsilon = 1e-10
        numerator = (h1_flat - h2_flat) ** 2
        denominator = h1_flat + h2_flat + epsilon
        return 0.5 * np.sum(numerator / denominator)

    elif method == 'intersection':
        return 1 - np.sum(np.minimum(h1_flat, h2_flat))

    elif method == 'hellinger':
        sqrt_h1 = np.sqrt(h1_flat)
        sqrt_h2 = np.sqrt(h2_flat)
        return np.sqrt(0.5 * np.sum((sqrt_h1 - sqrt_h2) ** 2))


def exercise_3c():
    img1 = cv2.imread('./dataset/object_01_1.png')
    img2 = cv2.imread('./dataset/object_02_1.png')
    img3 = cv2.imread('./dataset/object_03_1.png')

    n_bins = 8
    hist1 = myhist3(img1, n_bins)
    hist2 = myhist3(img2, n_bins)
    hist3 = myhist3(img3, n_bins)

    hist1_1d = hist1.flatten()
    hist2_1d = hist2.flatten()
    hist3_1d = hist3.flatten()

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    # Images
    plt.subplot(2, 3, 1)
    plt.imshow(img1_rgb)
    plt.title('Object 01_1')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img2_rgb)
    plt.title('Object 02_1')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(img3_rgb)
    plt.title('Object 03_1')
    plt.axis('off')

    # Histograms
    plt.subplot(2, 3, 4)
    plt.bar(range(len(hist1_1d)), hist1_1d)
    plt.title('Histogram Object 01_1')
    plt.xlabel('Bin Index')
    plt.ylabel('Probability')
    plt.ylim(0, 1.0)

    plt.subplot(2, 3, 5)
    plt.bar(range(len(hist2_1d)), hist2_1d)
    plt.title('Histogram Object 02_1')
    plt.xlabel('Bin Index')
    plt.ylabel('Probability')
    plt.ylim(0, 1.0)

    plt.subplot(2, 3, 6)
    plt.bar(range(len(hist3_1d)), hist3_1d)
    plt.title('Histogram Object 03_1')
    plt.xlabel('Bin Index')
    plt.ylabel('Probability')
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

    # for Question
    methods = ['L2', 'chi2', 'intersection', 'hellinger']
    print("\nAll distance measures:")
    print("Object 01_1 vs Object 02_1:")
    for method in methods:
        dist = compare_histograms(hist1, hist2, method)
        print(f"  {method}: {dist:.4f}")

    print("Object 01_1 vs Object 03_1:")
    for method in methods:
        dist = compare_histograms(hist1, hist3, method)
        print(f"  {method}: {dist:.4f}")


def compute_dataset_histograms(dataset_path, n_bins=8):
    image_files = glob.glob(os.path.join(dataset_path, '*.png'))
    image_files.sort()

    histograms = []
    filenames = []

    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            hist = myhist3(img, n_bins)
            histograms.append(hist.flatten())
            filenames.append(os.path.basename(img_file))

    return np.array(histograms), filenames


def exercise_3d():
    dataset_path = './dataset'
    n_bins = 8

    histograms, filenames = compute_dataset_histograms(dataset_path, n_bins)

    np.save('histograms.npy', histograms)
    np.save('filenames.npy', filenames)

    ref_filename = 'object_05_4.png'
    ref_idx = filenames.index(ref_filename)

    ref_hist = histograms[ref_idx]

    methods = ['L2', 'chi2', 'intersection', 'hellinger']

    for method in methods:
        distances = []
        for i, hist in enumerate(histograms):
            if i != ref_idx:
                hist_3d = hist.reshape(n_bins, n_bins, n_bins)
                ref_hist_3d = ref_hist.reshape(n_bins, n_bins, n_bins)
                dist = compare_histograms(ref_hist_3d, hist_3d, method)
                distances.append((dist, i, filenames[i]))

        distances.sort(key=lambda x: x[0])

        plt.figure(figsize=(18, 12))

        ref_img = cv2.imread(os.path.join(dataset_path, ref_filename))
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        plt.subplot(3, 6, 1)
        plt.imshow(ref_img_rgb)
        plt.title(f'Reference\n{ref_filename}')
        plt.axis('off')

        plt.subplot(3, 6, 7)
        plt.bar(range(len(ref_hist)), ref_hist)
        plt.title('Reference Histogram')
        plt.xlabel('Bin Index')
        plt.ylabel('Probability')

        for i in range(min(5, len(distances))):
            dist, img_idx, filename = distances[i]

            img = cv2.imread(os.path.join(dataset_path, filename))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(3, 6, i + 2)
            plt.imshow(img_rgb)
            plt.title(f'Match {i+1}\n{filename}\nDist: {dist:.3f}')
            plt.axis('off')

            plt.subplot(3, 6, i + 8)
            plt.bar(range(len(histograms[img_idx])), histograms[img_idx])
            plt.title(f'Histogram {i+1}')
            plt.xlabel('Bin Index')
            plt.ylabel('Probability')

        plt.suptitle(
            f'Image Retrieval Results - {method} Distance', fontsize=16)
        plt.tight_layout()
        plt.show()


def exercise_3e():
    histograms = np.load('histograms.npy')
    filenames = np.load('filenames.npy')

    ref_filename = 'object_05_4.png'
    ref_idx = list(filenames).index(ref_filename)

    ref_hist = histograms[ref_idx]
    n_bins = int(round(histograms.shape[1] ** (1/3)))

    methods = ['L2', 'chi2', 'intersection', 'hellinger']

    for method in methods:
        distances = []
        for i, hist in enumerate(histograms):
            hist_3d = hist.reshape(n_bins, n_bins, n_bins)
            ref_hist_3d = ref_hist.reshape(n_bins, n_bins, n_bins)
            dist = compare_histograms(ref_hist_3d, hist_3d, method)
            distances.append(dist)

        distances = np.array(distances)

        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        top5_indices = []
        for idx in sorted_indices:
            if idx != ref_idx and len(top5_indices) < 5:
                top5_indices.append(idx)

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(distances)), distances, 'b-', alpha=0.7)
        plt.plot(ref_idx, distances[ref_idx], 'ro',
                 markersize=8, label='Reference')
        for i, idx in enumerate(top5_indices):
            plt.plot(idx, distances[idx], 'go', markersize=8)
        if top5_indices:
            plt.plot([], [], 'go', markersize=8, label='Top 5 Matches')
        plt.title(f'Unsorted Distances - {method}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(sorted_distances)),
                 sorted_distances, 'r-', alpha=0.7)
        plt.plot(0, sorted_distances[0], 'ro',
                 markersize=8, label='Reference (Self)')
        for i in range(min(5, len(sorted_distances)-1)):
            plt.plot(i+1, sorted_distances[i+1], 'go', markersize=8)
        if len(sorted_distances) > 1:
            plt.plot([], [], 'go', markersize=8, label='Top 5 Matches')
        plt.title(f'Sorted Distances - {method}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(
            f'Distance Analysis - Reference: {ref_filename}', fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    # Exercise 1
    exercise_1b()
    exercise_1c()
    exercise_1d()
    exercise_1e()

    # Exercise 2
    exercise_2a()
    exercise_2b()
    exercise_2c()

    # Exercise 3
    exercise_3c()
    exercise_3d()
    exercise_3e()


if __name__ == "__main__":
    main()

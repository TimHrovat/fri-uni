import numpy as np
import cv2
from matplotlib import pyplot as plt


def exercise_1a():
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    height, width, channels = I.shape
    print(f"Image shape: {I.shape}, dtype: {I.dtype}")

    I_float = I.astype(np.float64)

    plt.imshow(I)
    plt.show()


def exercise_1b():
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    I_float = I.astype(np.float64)

    red = I_float[:, :, 0]
    green = I_float[:, :, 1]
    blue = I_float[:, :, 2]

    grayscale = (red + green + blue) / 3

    plt.imshow(grayscale, cmap='gray')
    plt.show()


def exercise_1c():
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    cutout = I[130:260, 240:450, 1]

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(I)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cutout)
    plt.title('Cutout (Green Channel) - cmap viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cutout, cmap="gray")
    plt.title('Cutout (Green Channel) - cmap gray')
    plt.axis('off')

    plt.show()


def exercise_1d():
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    inverted_image = np.copy(I)
    inverted_image[100:300, 200:400] = 255 - inverted_image[100:300, 200:400]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(inverted_image)
    plt.title('With Inverted Rectangle')
    plt.axis('off')

    plt.show()


def exercise_1e():
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(np.copy(I), cv2.COLOR_RGB2GRAY)

    max_val = grayscale.max()
    rescaled = grayscale * (0.3 / max_val)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Original Grayscale (auto-scaled)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rescaled, cmap='gray')
    plt.title('Rescaled to max=0.3 (auto-scaled)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(rescaled, cmap='gray', vmin=0, vmax=1)
    plt.title('rescaled with vmin=0, vmax=1')
    plt.axis('off')

    plt.show()


def exercise_2a():
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    threshold = 0.3

    mask1 = np.copy(grayscale_normalized)
    mask1[mask1 < threshold] = 0
    mask1[mask1 >= threshold] = 1

    mask2 = np.where(grayscale_normalized >= threshold, 1, 0)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(grayscale_normalized, cmap='gray')
    plt.title('Original Grayscale (bird.jpg)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask1, cmap='gray')
    plt.title(f'Binary Mask (NumPy syntax, threshold={threshold})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask2, cmap='gray')
    plt.title(f'Binary Mask (np.where, threshold={threshold})')
    plt.axis('off')

    plt.show()


def myhist(image, n_bins):
    H = np.zeros(n_bins)

    pixels = image.reshape(-1)
    bin_width = 1.0 / n_bins

    for pixel in pixels:
        pixel = np.clip(pixel, 0, 1)
        bin_index = int(pixel / bin_width)

        if bin_index >= n_bins:
            bin_index = n_bins - 1

        H[bin_index] += 1

    return H


def exercise_2b():
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.imshow(grayscale_normalized, cmap='gray')
    plt.title('Original Grayscale Image (bird.jpg)')
    plt.axis('off')

    for i, n_bins in enumerate([10, 20, 50, 100]):
        hist = myhist(grayscale_normalized, n_bins)

        plt.subplot(3, 2, i + 3)
        plt.bar(range(n_bins), hist, width=0.8)
        plt.title(f'Histogram with {n_bins} bins')
        plt.xlabel('Bin')
        plt.ylabel('Pixel Count')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    hist_100 = myhist(grayscale_normalized, 100)
    hist_100_normalized = hist_100 / np.sum(hist_100)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(100), hist_100)
    plt.title('Original Histogram (100 bins)')
    plt.xlabel('Bin')
    plt.ylabel('Pixel Count')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(range(100), hist_100_normalized)
    plt.title('Normalized Histogram (100 bins)')
    plt.xlabel('Bin')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def exercise_3a():
    I = cv2.imread('images/mask.png', cv2.IMREAD_GRAYSCALE)
    _, I = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(20, 15))

    plt.subplot(4, 5, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Mask')
    plt.axis('off')

    for i, n in enumerate([3, 5, 7, 9]):
        SE = np.ones((n, n), np.uint8)
        I_eroded = cv2.erode(I, SE)

        plt.subplot(4, 5, i + 2)
        plt.imshow(I_eroded, cmap='gray')
        plt.title(f'Erosion (SE size {n}x{n})')
        plt.axis('off')

    for i, n in enumerate([3, 5, 7, 9]):
        SE = np.ones((n, n), np.uint8)
        I_dilated = cv2.dilate(I, SE)

        plt.subplot(4, 5, i + 7)
        plt.imshow(I_dilated, cmap='gray')
        plt.title(f'Dilation (SE size {n}x{n})')
        plt.axis('off')

    for i, n in enumerate([3, 5, 7, 9]):
        SE = np.ones((n, n), np.uint8)
        I_opening = cv2.dilate(cv2.erode(I, SE), SE)

        plt.subplot(4, 5, i + 12)
        plt.imshow(I_opening, cmap='gray')
        plt.title(f'Opening (SE size {n}x{n})')
        plt.axis('off')

    for i, n in enumerate([3, 5, 7, 9]):
        SE = np.ones((n, n), np.uint8)
        I_closing = cv2.erode(cv2.dilate(I, SE), SE)

        plt.subplot(4, 5, i + 17)
        plt.imshow(I_closing, cmap='gray')
        plt.title(f'Closing (SE size {n}x{n})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def exercise_3b():
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    threshold = 0.3
    mask = np.where(grayscale_normalized >= threshold, 255, 0).astype(np.uint8)

    plt.figure(figsize=(18, 10))

    plt.subplot(3, 6, 1)
    plt.imshow(I)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 6, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Binary Mask')
    plt.axis('off')

    for i, n in enumerate([3, 5, 24]):
        SE_square = np.ones((n, n), np.uint8)
        opening_square = cv2.dilate(cv2.erode(mask, SE_square), SE_square)

        plt.subplot(3, 6, i + 3)
        plt.imshow(opening_square, cmap='gray')
        plt.title(f'Opening\nSquare SE {n}x{n}')
        plt.axis('off')

        SE_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        opening_ellipse = cv2.dilate(cv2.erode(mask, SE_ellipse), SE_ellipse)

        plt.subplot(3, 6, i + 9)
        plt.imshow(opening_ellipse, cmap='gray')
        plt.title(f'Opening\nEllipse SE {n}x{n}')
        plt.axis('off')

        closing_ellipse = cv2.erode(cv2.dilate(mask, SE_ellipse), SE_ellipse)

        plt.subplot(3, 6, i + 15)
        plt.imshow(closing_ellipse, cmap='gray')
        plt.title(f'Closing\nEllipse SE {n}x{n}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def immask(image, mask):
    mask_normalized = mask / 255.0

    mask_3d = np.expand_dims(mask_normalized, axis=2)

    masked_image = image * mask_3d

    return masked_image.astype(image.dtype)


def exercise_3c():
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    threshold = 0.3
    mask = np.where(grayscale_normalized >= threshold, 255, 0).astype(np.uint8)

    SE_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask_cleaned = cv2.dilate(cv2.erode(mask, SE_ellipse), SE_ellipse)

    masked_result = immask(I, mask_cleaned)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(I)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_cleaned, cmap='gray')
    plt.title('Cleaned Binary Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(masked_result)
    plt.title('Immask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def exercise_3d():
    I = cv2.imread('images/eagle.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    threshold = 0.4
    mask = np.where(grayscale_normalized >= threshold, 255, 0).astype(np.uint8)

    masked_result = immask(I, mask)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(I)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(masked_result)
    plt.title('Immask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return mask


def exercise_3e():
    I = cv2.imread('images/coins.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    grayscale_normalized = grayscale / 255.0

    threshold = 0.8
    mask = np.where(grayscale_normalized < threshold, 255, 0).astype(np.uint8)

    SE_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE_close)

    SE_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, SE_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_cleaned, connectivity=8)

    small_coins_mask = np.zeros_like(mask_cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= 700:
            component_mask = (labels == i)
            small_coins_mask[component_mask] = 255

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(small_coins_mask, cmap='gray')
    plt.title('Small Coins Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Exercise 1 - Basic image processing
    exercise_1a()
    exercise_1b()
    exercise_1c()
    exercise_1d()
    exercise_1e()

    # Exercise 2 - Thresholding and histograms
    exercise_2a()
    exercise_2b()

    # Exercise 3 - Morphological operations
    exercise_3a()
    exercise_3b()
    exercise_3c()
    exercise_3d()
    exercise_3e()


if __name__ == "__main__":
    main()

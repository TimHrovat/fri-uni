import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob
import matplotlib.colors as mcolors


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


def main():
    # Exercise 1
    # Exercise 1a: solved on my ipad
    exercise1b()
    exercise1c()
    exercise1d()

    # Exercise 2

    # Exercise 3


if __name__ == "__main__":
    main()

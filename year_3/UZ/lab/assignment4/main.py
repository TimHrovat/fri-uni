import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from a4_utils import gauss, gaussdx, convolve
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


if __name__ == "__main__":
    exercise1a()

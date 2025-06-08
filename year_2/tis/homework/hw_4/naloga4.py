import numpy as np


def naloga4(image: np.array, threshold: float) -> float:
    if image.max() > 1.0:
        image = image / 255.0

    freq_repr = np.fft.fft2(image)
    freq_repr_shifted = np.fft.fftshift(freq_repr)

    height, width = image.shape
    center_y = height // 2
    center_x = width // 2

    y = (np.arange(height) - center_y) / height
    x = (np.arange(width) - center_x) / width
    distance_squared = y[:, None]**2 + x[None, :]**2
    mask = (distance_squared >= threshold**2).astype(float)

    filtered_shifted = freq_repr_shifted * mask
    filtered = np.fft.ifftshift(filtered_shifted)
    image_filtered_complex = np.fft.ifft2(filtered)
    image_filtered = np.real(image_filtered_complex)
    image_filtered = np.clip(image_filtered, 0, 1)

    levels = 256
    bins = np.linspace(0, 1, levels + 1)

    hist_input, _ = np.histogram(image, bins=bins)
    p_input = hist_input / (height * width)
    entropy_input = -np.sum(p_input * np.log2(p_input + (p_input == 0)))

    hist_output, _ = np.histogram(image_filtered, bins=bins)
    p_output = hist_output / (height * width)
    entropy_output = -np.sum(p_output * np.log2(p_output + (p_output == 0)))

    joint_hist, _, _ = np.histogram2d(
        image.flatten(), image_filtered.flatten(), bins=[bins, bins])
    p_joint = joint_hist / (height * width)
    joint_entropy = -np.sum(p_joint * np.log2(p_joint + (p_joint == 0)))

    return entropy_input + entropy_output - joint_entropy

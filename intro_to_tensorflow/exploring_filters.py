from scipy.datasets import ascent
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def plot(image):
    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def manual_conv(image: np.ndarray, conv_filter: Union[np.ndarray, list]) -> np.ndarray:
    if not isinstance(conv_filter, np.ndarray):
        conv_filter = np.array(conv_filter)
    Nx = image_transformed.shape[0]  # Get the dimensions of the image
    Ny = image_transformed.shape[1]
    Sx = conv_filter.shape[0]
    Sy = conv_filter.shape[1]
    output = np.zeros((Nx - Sx + 1, Ny - Sy + 1))  # conv output size

    for i in range(Nx - Sx + 1):
        for j in range(Ny - Sy + 1):
            output[i, j] = np.sum(np.multiply(conv_filter, image[i: i + Sx, j: j + Sy]))
            if output[i, j] < 0:
                output[i, j] = 0
            if output[i, j] > 255:
                output[i, j] = 255
    return output


def max_pool(image: np.ndarray, filter_size=(2, 2)) -> np.ndarray:
    Nx = image.shape[0] // filter_size[0]
    Ny = image.shape[1] // filter_size[1]
    Sx = filter_size[0]
    Sy = filter_size[1]
    new_image = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            new_image[i, j] = np.max(image[i * Sx: (i + 1) * Sx, j * Sy: (j + 1) * Sy])
            if new_image[i, j] < 0:
                new_image[i, j] = 0
            if new_image[i, j] > 255:
                new_image[i, j] = 255
    return new_image


if __name__ == '__main__':
    ascent_image = ascent()
    Nx = ascent_image.shape[0]  # Get the dimensions of the image
    Ny = ascent_image.shape[1]

    filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    filter2 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    filter3 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    weight = 1
    image_transformed = np.copy(ascent_image)

    img1 = manual_conv(image_transformed, filter)
    plot(img1)
    img1_pooled = max_pool(img1, (2, 2))
    plot(img1_pooled)
    # img2 = manual_conv(image_transformed, filter2)
    # plot(img2)
    # img3 = manual_conv(image_transformed, filter3)
    # plot(img3)

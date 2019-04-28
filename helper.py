import numpy as np


def normalise_image(image):
    """
    make image zero mean and unit standard deviation
    """

    img = np.float32(image.copy())
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        return 0
    return np.divide((img - mean), std)

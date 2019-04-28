import numpy as np
from keras import backend as K
from tensorflow.python.client import device_lib


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


def check_gpu_usage():
    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())
import numpy as np
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.python.client import device_lib

import config


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


def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(optimizer_name))
    return optimizers[optimizer_name](**optimizer_args)


def create_optimizer():
    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr': config.learning_rate,
        'momentum': config.momentum,
        'decay': config.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(config.optimizer, optimizer_args)
    return optimizer
import metrics

from deeplabv3plus import Deeplabv3
from helper import create_optimizer
import config
import keras.backend as K


def dice_loss(y_true, y_pred):
    batch_dice_coefs = metrics.dice_coef(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(config.weights) / sum(config.weights)
    return 1 - K.sum(w * dice_coefs)


def dice(y_true, y_pred):
    batch_dice_coefs = metrics.dice_coef(K.round(y_true), K.round(y_pred), axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    return dice_coefs[1]  # HACK for 2-class case


def deeplabv3plus_model(input_shape, classes, weights_path=None):
    model = Deeplabv3(input_shape=input_shape, classes=classes, weights=None)
    if weights_path is not None:
        model.load_weights(weights_path)
    optimizer = create_optimizer()
    model.compile(optimizer=optimizer, loss=dice_loss,
                  metrics=["accuracy", dice, metrics.jaccard_coef])
    return model


if __name__  == "__main__":
    model = deeplabv3plus_model((100, 100, 1), 2)
    model.summary()

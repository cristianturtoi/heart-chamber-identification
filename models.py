import metrics

from deeplabv3plus import Deeplabv3
from helper import create_optimizer
import config
import keras.backend as K
import tensorflow as tf


def dice_loss(y_true, y_pred):
    batch_dice_coefs = metrics.dice_coef(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(config.weights) / sum(config.weights)
    return 1 - K.sum(w * dice_coefs)


def dice(y_true, y_pred):
    batch_dice_coefs = metrics.dice_coef(K.round(y_true), K.round(y_pred), axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    return dice_coefs[1]  # HACK for 2-class case


def dice_weighted_loss6(y_true, y_pred):
    dice_coefs = calculate_dices(y_true, y_pred)
    d0 = tf.scalar_mul(config.weights[0], dice_coefs[0])
    d1 = tf.scalar_mul(config.weights[1], dice_coefs[1])
    d2 = tf.scalar_mul(config.weights[2], dice_coefs[2])
    return 1 - tf.math.add_n([d0, d1, d2])


def dice_loss6(y_true, y_pred):
    return 1 - dice6(y_true, y_pred)


def dice6(y_true, y_pred, smooth=1.0):
    res0, res1, res2 = calculate_dices(y_true, y_pred, smooth)

    print("Dice0 = ", res0, ";Dice1 = ", res1, ";Dice2 = ", res2)
    return (res0 + res1 + res2) / 3.0


def calculate_dices(y_true, y_pred, smooth=1.0):
    # y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)),
                       K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)),
                       K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)),
                       K.ones_like(y_true), K.zeros_like(y_true))
    y_pred0 = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 1])
    # y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0, 0, 0, 1], [-1, -1, -1, 1])
    # y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0, 0, 0, 2], [-1, -1, -1, 1])
    # y_pred_f2 = K.flatten(y_pred2)
    # intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1, 2, 3])
    sum0 = K.sum(y_true0, axis=[1, 2, 3]) + K.sum(y_pred0, axis=[1, 2, 3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)
    # intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1, 2, 3])
    sum1 = K.sum(y_true1, axis=[1, 2, 3]) + K.sum(y_pred1, axis=[1, 2, 3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1 + smooth), axis=0)
    # intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1, 2, 3])
    sum2 = K.sum(y_true2, axis=[1, 2, 3]) + K.sum(y_pred2, axis=[1, 2, 3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)
    return res0, res1, res2


def deeplabv3plus_model(input_shape, classes, weights_path=None):
    model = Deeplabv3(input_shape=input_shape, classes=classes, weights=None)
    if weights_path is not None:
        model.load_weights(weights_path)
    optimizer = create_optimizer()
    model.compile(optimizer=optimizer, loss=dice_weighted_loss6,
                  metrics=["accuracy", dice6, metrics.jaccard_coef])
    return model


if __name__  == "__main__":
    model = deeplabv3plus_model((100, 100, 1), 2)
    model.summary()

import metrics

from deeplabv3plus import Deeplabv3
from helper import create_optimizer


def deeplabv3plus_model(input_shape, classes, weights_path=None):
    model = Deeplabv3(input_shape=input_shape, classes=classes, weights=None)
    if weights_path is not None:
        model.load_weights(weights_path)
    optimizer = create_optimizer()
    model.compile(optimizer=optimizer, loss=metrics.dice_loss,
                  metrics=["accuracy", metrics.dice_coef, metrics.jaccard_coef])
    return model


if __name__  == "__main__":
    model = deeplabv3plus_model((100, 100, 1), 2)
    model.summary()

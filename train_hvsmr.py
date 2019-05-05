import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import augmentation
import config
import helper
import models
from config import MODEL_CHECKPOINT


def train():
    helper.check_gpu_usage()

    augmentation_args = {
        'rotation_range': config.rotation_range,
        'width_shift_range': config.width_shift_range,
        'height_shift_range': config.height_shift_range,
        'shear_range': config.shear_range,
        'zoom_range': config.zoom_range,
        'fill_mode': config.fill_mode,
        'horizontal_flip': config.horizontal_flip,
        'vertical_flip': config.vertical_flip,
        'alpha': config.alpha,
        'sigma': config.sigma,
    }

    train_generator, train_steps_per_epoch, \
    val_generator, val_steps_per_epoch = augmentation.create_generators(
        config.batch_size, config.crop_size,
        validation_split=config.validation_split,
        seed=config.seed,
        augmentation_args=augmentation_args)

    model = models.deeplabv3plus_model((config.crop_size, config.crop_size, 1), config.classes)

    callbacks = [
        # EarlyStopping(patience=10, verbose=1, monitor="val_dice"),
        ReduceLROnPlateau(factor=0.1, patience=25, min_lr=0.00001, verbose=1, monitor="val_dice6"),
        ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True, mode='max', monitor="val_dice6")
    ]

    results = model.fit_generator(train_generator,
                                  epochs=config.epochs,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  callbacks=callbacks,
                                  verbose=1)

    model.save("final-model-hvsmr.h5")

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["dice6"], label="dice6")
    plt.plot(results.history["val_dice6"], label="val_dice6")
    plt.plot(np.argmin(results.history["val_dice6"]), np.max(results.history["val_dice6"]), marker="x",
             color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

import augmentation
import config
import helper
import models
from augmentation import load_preprocess_data
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
        config.batch_size, config.im_height,
        validation_split=config.validation_split,
        seed=config.seed,
        # augment_training=args.augment_training,
        augmentation_args=augmentation_args)

    # X_train, X_valid, y_train, y_valid = get_resized_data3()

    # X_train, X_valid, y_train, y_valid = load_preprocess_data(crop_size=config.im_height)
    # X_train, y_train = shuffle(X_train, y_train)
    # X_valid, y_valid = shuffle(X_valid, y_valid)

    model = models.deeplabv3plus_model((config.im_height, config.im_width, 1), config.classes)

    callbacks = [
        # EarlyStopping(patience=10, verbose=1, monitor="val_dice"),
        ReduceLROnPlateau(factor=0.1, patience=25, min_lr=0.00001, verbose=1, monitor="val_dice"),
        ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True, mode='max', monitor="val_dice")
    ]

    # augment_options = dict(
    #     rotation_range=180,
    #     zoom_range=0.0,
    #     width_shift_range=0.0,
    #     height_shift_range=0.0,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    #
    # image_datagen = ImageDataGenerator(**augment_options)
    # mask_datagen = ImageDataGenerator(**augment_options)
    #
    # image_generator = image_datagen.flow(X_train, shuffle=False, batch_size=config.batch_size, seed=config.seed)
    # mask_generator = mask_datagen.flow(y_train, shuffle=False, batch_size=config.batch_size, seed=config.seed)
    #
    # train_generator = zip(image_generator, mask_generator)
    #
    # results = model.fit_generator(train_generator,
    #                               epochs=config.epochs,
    #                               steps_per_epoch=(len(X_train) // config.batch_size),
    #                               validation_data=(X_valid, y_valid),
    #                               callbacks=callbacks,
    #                               verbose=1)

    results = model.fit_generator(train_generator,
                                  epochs=config.epochs,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  callbacks=callbacks,
                                  verbose=1)

    model.save("final-model-hvsmr.h5.h5")

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["dice"], label="dice")
    plt.plot(results.history["val_dice"], label="val_dice")
    plt.plot(np.argmin(results.history["val_dice"]), np.min(results.history["val_dice"]), marker="x",
             color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()

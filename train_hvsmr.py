import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.utils import shuffle

import config
import loader
import helper
import models
from config import MODEL_CHECKPOINT


def resize_images2(images):
    resized_imgs = np.zeros((len(images), config.im_height, config.im_width, 1), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        resized_imgs[idx] = resize(img, (config.im_height, config.im_width, 1), mode='constant', preserve_range=True)

    print(np.unique(resized_imgs))
    return resized_imgs


def get_resized_data2():
    X_train, X_valid, y_train, y_valid = loader.read_images_with_groundtruth()
    return resize_images2(X_train), resize_images2(X_valid), resize_images2(y_train), resize_images2(y_valid)


def crop_images(images, crop_size):
    resized_images = np.zeros((len(images), config.im_height, config.im_width, 1), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        resized_images[idx] = helper.center_crop(img, crop_size)[:, :, np.newaxis]

    print(np.unique(resized_images))
    return resized_images


def get_resized_data3():
    X_train, X_valid, y_train, y_valid = loader.read_images_with_groundtruth()
    return crop_images(X_train, config.im_height), crop_images(X_valid, config.im_height), \
           crop_images(y_train, config.im_height), crop_images(y_valid, config.im_height)


def load_preprocess_data(crop_size):
    """
    Load, crop (resize) and normalize data
    :return:
    """
    X_train, X_valid, y_train, y_valid = loader.read_images_with_groundtruth()
    X_train = crop_images(X_train, crop_size)
    X_valid = crop_images(X_valid, crop_size)
    y_train = crop_images(y_train, crop_size)
    y_valid = crop_images(y_valid, crop_size)
    return helper.normalize(X_train), helper.normalize(X_valid), y_train, y_valid


def train():
    helper.check_gpu_usage()

    # X_train, X_valid, y_train, y_valid = get_resized_data2()
    # X_train, X_valid, y_train, y_valid = get_resized_data3()
    X_train, X_valid, y_train, y_valid = load_preprocess_data(crop_size=config.im_height)
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)

    model = models.deeplabv3plus_model((config.im_height, config.im_width, 1), config.classes)

    callbacks = [
        # EarlyStopping(patience=10, verbose=1, monitor="val_dice"),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1, monitor="val_dice"),
        ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True, save_weights_only=True,
                        monitor="val_dice")
    ]

    augment_options = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_datagen = ImageDataGenerator(**augment_options)
    mask_datagen = ImageDataGenerator(**augment_options)

    image_generator = image_datagen.flow(X_train, shuffle=False, batch_size=config.batch_size, seed=config.seed)
    mask_generator = mask_datagen.flow(y_train, shuffle=False, batch_size=config.batch_size, seed=config.seed)

    train_generator = zip(image_generator, mask_generator)

    results = model.fit_generator(train_generator,
                                  epochs=1000,
                                  steps_per_epoch=(len(X_train) // config.batch_size),
                                  validation_data=(X_valid, y_valid),
                                  callbacks=callbacks,
                                  verbose=1)

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

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

import config
import load_2D
import helper
import models
from config import MODEL_CHECKPOINT


def resize_images(images_path, is_groundtruth=False):
    """
    Load, resize and normalize the images from images_path
    NOTE: the ground truth images are not normalized
    """
    if is_groundtruth:
        resized_imgs = np.zeros((len(images_path), config.im_height, config.im_width, 1), dtype=np.uint8)
    else:
        resized_imgs = np.zeros((len(images_path), config.im_height, config.im_width, 1), dtype=np.float32)
    for n, img_path in enumerate(images_path):
        img = load_2D.load_img_to_array(img_path)
        resized_img = resize(img, (config.im_height, config.im_width, 1), mode='constant', preserve_range=True)
        if is_groundtruth:
            normalized_img = resized_img
        else:
            normalized_img = helper.normalise_image(resized_img)
        resized_imgs[n] = normalized_img
    print(np.unique(resized_imgs))
    return resized_imgs


def get_resized_data():
    X_train, X_valid, y_train, y_valid = load_2D.get_shuffled_records()
    return resize_images(X_train), resize_images(y_train), resize_images(X_valid, is_groundtruth=True), resize_images(
        y_valid, is_groundtruth=True)


def train():
    helper.check_gpu_usage()

    X_train, X_valid, y_train, y_valid = get_resized_data()

    classes = 1
    model = models.deeplabv3plus_model((config.im_height, config.im_width, 1), classes)

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True, save_weights_only=True)
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

    seed = 2019
    batch_size = 16
    image_generator = image_datagen.flow(X_train, shuffle=False, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, shuffle=False, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    results = model.fit_generator(train_generator,
                        epochs=50,
                        steps_per_epoch=(len(X_train) // batch_size),
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks,
                        verbose=1)

    # results = model.fit(X_train, y_train, batch_size=16, epochs=50, callbacks=callbacks,
    #                     validation_data=(X_valid, y_valid))

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()

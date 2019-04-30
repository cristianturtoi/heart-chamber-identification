import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from tqdm import tqdm

import config
import loader
import models


def resize_gt_imgs(images):
    resized_imgs = np.zeros((len(images), config.im_height, config.im_width, 1), dtype=np.uint8)
    for n, img_path in tqdm(enumerate(images)):
        img = load_img(img_path, color_mode='grayscale')
        img = img_to_array(img)
        reImg = resize(img, (config.im_height, config.im_width, 1), mode='constant', preserve_range=True)
        resized_imgs[n] = reImg
    print(np.unique(resized_imgs))
    return resized_imgs


def get_resized_data():
    X_train, X_valid, y_train, y_valid = loader.get_shuffled_records()
    return resize_gt_imgs(X_train), resize_gt_imgs(y_train), resize_gt_imgs(X_valid), resize_gt_imgs(y_valid)


def resize_images2(images):
    resized_imgs = np.zeros((len(images), config.im_height, config.im_width, 1), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        resized_imgs[idx] = resize(img, (config.im_height, config.im_width, 1), mode='constant', preserve_range=True)

    print(np.unique(resized_imgs))
    return resized_imgs


def get_resized_data2():
    X_train, X_valid, y_train, y_valid = loader.read_images_with_groundtruth()
    return resize_images2(X_train), resize_images2(X_valid), resize_images2(y_train), resize_images2(y_valid)


def predict():
    model_path = os.path.join(config.code_root, "model-hvsmr.h5")
    classes = 1
    model = models.deeplabv3plus_model((config.im_height, config.im_width, 1), classes, weights_path=model_path)

    print('This model has {} parameters'.format(model.count_params()))

    X_train, X_valid, y_train, y_valid = get_resized_data2()
    # X_train, X_valid, y_train, y_valid = get_resized_data()
    print(np.unique(y_train))

    # Evaluate on validation set (this must be equals to the best log_loss)
    result = model.evaluate(X_valid, y_valid, verbose=1)
    print(result)

    # Predict on train, val and test
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_valid, verbose=1)

    # Threshold predictions
    # preds_train_t = (preds_train > 0.5).astype(np.uint8)
    # preds_val_t = (preds_val > 0.5).astype(np.uint8)

    # Check if training data looks all right
    # plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)

    # Check if valid data looks all right
    # plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=19)

    # plt.show()


if __name__ == "__main__":
    predict()

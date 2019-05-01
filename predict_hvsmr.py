import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from tqdm import tqdm

import config
import helper
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
    model = models.deeplabv3plus_model((config.im_height, config.im_width, 1), config.classes, weights_path=model_path)

    print('This model has {} parameters'.format(model.count_params()))

    # Predict on train, val and test
    for idx in range(config.nr_patients):
        images, masks = loader.load_patient_images(idx)
        h, w = images[0].shape

        images = resize_images2(images)
        predictions = []
        for image in images:
            img = image[np.newaxis, :, :, :]
            mask_pred = model.predict(img, batch_size=config.batch_size)
            mask_pred = np.argmax(mask_pred.squeeze(), -1)

            # resize back to original image
            print(idx, mask_pred.shape, np.unique(mask_pred))
            orig_size_pred = helper.reshape(mask_pred, to_shape=(h, w))
            predictions.append(np.uint8(orig_size_pred))

        # RECONSTRUCT image, and save as .nii.gz
        predictions = np.array(predictions)
        predictions = np.transpose(predictions, [1, 2, 0])

        print(idx, predictions.shape, np.unique(predictions))


if __name__ == "__main__":
    predict()

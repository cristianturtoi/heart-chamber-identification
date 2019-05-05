import os

import numpy as np
from PIL import Image
import nibabel as nib

import config
import helper
import loader
import models
from helper import save_nii


def crop_images(images, crop_size):
    resized_images = np.zeros((len(images), crop_size, crop_size, 1), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        resized_images[idx] = helper.center_crop(img, crop_size)[:, :, np.newaxis]

    print(np.unique(resized_images))
    return resized_images


def preprocess_data(data, crop_size):
    data = crop_images(data, crop_size)
    return helper.normalize(data)


def predict():
    model_path = os.path.join(config.code_root, "final-model-hvsmr.h5")
    model = models.deeplabv3plus_model((config.crop_size, config.crop_size, 1), config.classes, weights_path=model_path)

    print('This model has {} parameters'.format(model.count_params()))

    path_predictions_png = os.path.join(config.code_root, "predictions")
    if not os.path.exists(path_predictions_png):
        os.makedirs(path_predictions_png)

    # Predict on train, val and test
    for idx in range(config.nr_patients):
        images, masks = loader.load_patient_images(idx)
        h, w = images[0].shape

        gt_path = config.label_path
        # gt_filename = 'training_sa_crop_pat' + str(idx) + '-label.nii.gz'
        test_img_path = os.path.join(gt_path, "training_sa_crop_pat{}-label.nii.gz".format(idx))
        img_nii = nib.load(test_img_path)
        output_header = img_nii.header
        output_affine = img_nii.affine

        images = preprocess_data(images, config.crop_size)
        predictions = []
        for slicex, image in enumerate(images):
            img = image[np.newaxis, :, :, :]
            mask_pred = model.predict(img, batch_size=config.batch_size)
            mask_pred = np.argmax(mask_pred.squeeze(), axis=-1)

            # resize back to original image
            # print(idx, mask_pred.shape, np.unique(mask_pred))
            orig_size_pred = helper.reshape_2D(mask_pred, to_shape=(h, w))

            # save slice as png
            stacked_image = np.stack((masks[slicex],) * 3, axis=-1)
            Image.fromarray(np.uint8(stacked_image * 25)).save(os.path.join(
                path_predictions_png, "training_sa_crop_pat{}_s{}_orig.png".format(idx, slicex)))
            stacked_pred = np.stack((orig_size_pred,) * 3, axis=-1)
            Image.fromarray(np.uint8(stacked_pred * 25)).save(os.path.join(
                path_predictions_png, "training_sa_crop_pat{}_s{}_pred.png".format(idx, slicex)))

            predictions.append(np.uint8(orig_size_pred))

        # RECONSTRUCT image, and save as .nii.gz
        predictions = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))
        # predictions = np.fliplr(predictions)

        img_path = "training_sa_crop_pat{}-label.nii.gz".format(idx)
        save_nii(img_path, predictions, output_affine, output_header)

        print(idx, predictions.shape, np.unique(predictions))


if __name__ == "__main__":
    predict()

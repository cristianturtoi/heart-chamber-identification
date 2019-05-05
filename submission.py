import glob
import os
import random

import config
import models
import nibabel as nib
import numpy as np
from skimage.transform import resize


def main():
    test_path = config.test_path
    tested_patient_id = random.randint(10, 19)
    model_path = os.path.join(config.code_root, "model-hvsmr.h5")

    test_img_path = os.path.join(test_path, "testing_sa_crop_pat{}.nii.gz".format(tested_patient_id))

    img_nii = nib.load(test_img_path)
    img_data = img_nii.get_data().copy()
    output_header = img_nii.header
    output_affine = img_nii.affine

    classes = 1
    model = models.deeplabv3plus_model((config.crop_size, config.crop_size, 1), classes, model_path)
    print('This model has {} parameters'.format(model.count_params()))

    max_pixel_value = np.array(img_data).max()
    if max_pixel_value > 0:
        multiplier = 255.0 / max_pixel_value
    else:
        multiplier = 1.0
    slices = img_data.shape[2]
    images = np.zeros((slices, config.crop_size, config.crop_size, 1))
    for slice_idx in range(slices):
        img = (np.rot90(img_data[:, ::-1, slice_idx] * multiplier)).astype('uint8')
        img = resize(img, (config.crop_size, config.crop_size, 1), mode='constant', preserve_range=True)
        images[slice_idx] = img

    pred_masks = model.predict(images, batch_size=32, verbose=1)
    output_height = img_data.shape[0]
    output_width = img_data.shape[1]
    final_pred_mask = np.zeros((slices, output_height, output_width))
    for idx, pred_img in enumerate(pred_masks):
        final_pred = resize(pred_img, (output_height, output_width, 1), mode='constant', preserve_range=True)
        final_pred_mask[idx] = final_pred.squeeze()

    final_pred_mask = np.transpose(final_pred_mask, [1, 2, 0])
    nimg = nib.Nifti1Image(final_pred_mask, affine=output_affine, header=output_header)
    nimg.to_filename("test.nii.gz")

    import matplotlib.pyplot as plt
    plt.imshow(final_pred_mask[:, :, 100])
    plt.show()


if __name__ == '__main__':
    main()

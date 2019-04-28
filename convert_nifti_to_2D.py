import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

import config


def convert_nifti_to_2D():
    train_path = config.train_path
    gt_path = config.label_path

    original_2D_path = os.path.join(config.data_root, "original_2D")
    if not os.path.exists(original_2D_path):
        os.makedirs(original_2D_path)

    for id in tqdm(range(config.nr_patients)):
        img_filename = 'training_sa_crop_pat' + str(id) + '.nii.gz'
        gt_filename = 'training_sa_crop_pat' + str(id) + '-label.nii.gz'

        img_file = os.path.join(train_path, img_filename)

        # If the short-axis image file exists, read the data and perform the conversion
        if os.path.isfile(img_file):
            img = nib.load(img_file)
            data = img.get_data()
            data_np = np.array(data)

            max_pixel_value = data_np.max()
            if max_pixel_value > 0:
                multiplier = 255.0 / max_pixel_value
            else:
                multiplier = 1.0

            print('max_pixel_value = {},  multiplier = {}'.format(max_pixel_value, multiplier))

            print(data.shape)
            slices = data.shape[2]

            for s in range(slices):
                slice_image_file = os.path.join(original_2D_path, 'original_2D_p{}_{}.png'.format(str(id), str(s).zfill(2)))
                Image.fromarray((np.rot90(data[:, ::-1, s], 1) * multiplier).astype('uint8')).save(slice_image_file)

        else:
            print('There is no SA image file for {}'.format(img_filename))

        gt_file = os.path.join(gt_path, gt_filename)

        # If the ground-truth file exists, read the data and perform the conversion
        if os.path.isfile(gt_file):
            gt = nib.load(gt_file)
            gt_data = gt.get_data()
            gt_data_np = np.array(gt_data)

            # print(np.unique(gt_data_np))

            slices = gt_data.shape[2]
            if gt_data_np[:, :, :].max() > 0:
                for s in range(slices):
                    slice_image_gt_file = os.path.join(original_2D_path, 'original_gt_2D_p{}_{}.png'.format(str(id), str(s).zfill(2)))
                    rotated_img = (np.rot90(gt_data[:, ::-1, s], 1)).astype('uint8')
                    # print(np.unique(rotated_img))
                    Image.fromarray(rotated_img).save(slice_image_gt_file)

        else:
            print('There is no SA label image file for {}'.format(gt_filename))


if __name__ == "__main__":
    convert_nifti_to_2D()

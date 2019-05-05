import os

import h5py
import nibabel as nib
from tqdm import tqdm

import config


def print_images_shape(path, prefix, patient_start_index, patient_end_index):
    for idx in tqdm(range(patient_start_index, patient_end_index)):
        img_filename = prefix + str(idx) + '.nii.gz'
        img_file = os.path.join(path, img_filename)
        if os.path.isfile(img_file):
            img = nib.load(img_file)
            print(idx, img.get_data().shape)


def save_images_as_png():
    import numpy as np
    from PIL import Image
    png_path = os.path.join(config.code_root, "pngs")
    if not os.path.exists(png_path):
        os.makedirs(png_path)

    for idx in range(config.nr_patients):
        training_file_prefix = 'training_sa_crop_pat'
        img_filename = training_file_prefix + str(idx) + '-label.nii.gz'
        img_file = os.path.join(config.label_path, img_filename)
        image = nib.load(img_file).get_data()
        for slice_idx in range(image.shape[2]):
            slice_img = image[:, :, slice_idx]
            stacked_image = np.stack((slice_img,) * 3, axis=-1)
            Image.fromarray(np.uint8(stacked_image * 25)).save(os.path.join(
                png_path, "training_sa_crop_pat{}_s{}_orig.png".format(idx, slice_idx)))


def convert_nifti_to_2D():
    train_path = config.train_path
    gt_path = config.label_path
    training_file_prefix = 'training_sa_crop_pat'
    file_extension = '.nii.gz'

    original_2D_path = os.path.join(config.data_root, "original_2D")
    if not os.path.exists(original_2D_path):
        os.makedirs(original_2D_path)

    for idx in tqdm(range(config.nr_patients)):
        img_filename = training_file_prefix + str(idx) + file_extension
        gt_filename = training_file_prefix + str(idx) + '-label.nii.gz'

        img_file = os.path.join(train_path, img_filename)
        gt_file = os.path.join(gt_path, gt_filename)

        # If the short-axis image file exists, read the data and perform the conversion
        # If the ground-truth file exists, read the data and perform the conversion
        if os.path.isfile(img_file) and os.path.isfile(gt_file):
            # Load image
            image = nib.load(img_file).get_data()
            print(image.shape)

            # Load ground truth
            ground_truth_img = nib.load(gt_file).get_data()

            for slice_idx in range(image.shape[2]):
                # h5 file path
                slice_image_file = os.path.join(original_2D_path,
                                                'original_2D_p{}_{}.h5'.format(str(idx), str(slice_idx).zfill(3)))
                if os.path.exists(slice_image_file):
                    os.remove(slice_image_file)

                slice_img = image[:, :, slice_idx]
                data_dims = [slice_img.shape[0], slice_img.shape[1]]

                slice_gt_img = ground_truth_img[:, :, slice_idx]
                gt_dims = [slice_gt_img.shape[0], slice_gt_img.shape[1]]

                # store as h5 format
                h5file = h5py.File(slice_image_file, 'w')
                h5file.create_dataset('data', data_dims, dtype=None, data=slice_img)
                h5file.create_dataset('label', gt_dims, dtype=None, data=slice_gt_img)
                h5file.close()

        elif os.path.isfile(img_file):
            # Load only the image - this is for testing phase where we don't have ground truth
            pass

        else:
            print('There is no SA image file for {}'.format(img_filename))


if __name__ == "__main__":
    convert_nifti_to_2D()

    # save_images_as_png()

    # print("training")
    # print_images_shape(config.train_path, "training_sa_crop_pat", 0, 10)
    # print("testing")
    # print_images_shape(config.test_path, "testing_sa_crop_pat", 10, 20)

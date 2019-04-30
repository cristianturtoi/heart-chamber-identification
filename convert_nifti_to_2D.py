import os

import h5py
import nibabel as nib
from tqdm import tqdm

import config
import helper


def convert_nifti_to_2D():
    train_path = config.train_path
    gt_path = config.label_path

    original_2D_path = os.path.join(config.data_root, "original_2D")
    if not os.path.exists(original_2D_path):
        os.makedirs(original_2D_path)

    for idx in tqdm(range(config.nr_patients)):
        img_filename = 'training_sa_crop_pat' + str(idx) + '.nii.gz'
        gt_filename = 'training_sa_crop_pat' + str(idx) + '-label.nii.gz'

        img_file = os.path.join(train_path, img_filename)
        gt_file = os.path.join(gt_path, gt_filename)

        # If the short-axis image file exists, read the data and perform the conversion
        # If the ground-truth file exists, read the data and perform the conversion
        if os.path.isfile(img_file) and os.path.isfile(gt_file):
            # Load image
            img = nib.load(img_file)
            data = img.get_data()

            # Load ground truth
            gt = nib.load(gt_file)
            gt_data = gt.get_data()

            print(data.shape)

            slices = data.shape[2]
            for s in range(slices):
                # h5 file path
                slice_image_file = os.path.join(original_2D_path, 'original_2D_p{}_{}.h5'.format(str(idx), str(s).zfill(3)))
                if os.path.exists(slice_image_file):
                    os.remove(slice_image_file)

                slice_data = data[:, ::-1, s]
                slice_data = helper.normalise_image(slice_data)
                data_dims = [slice_data.shape[0], slice_data.shape[1]]

                slice_gt_data = gt_data[:, ::-1, s]
                gt_dims = [slice_gt_data.shape[0], slice_gt_data.shape[1]]

                # store as h5 format
                h5file = h5py.File(slice_image_file, 'w')
                h5file.create_dataset('data', data_dims, dtype=None, data=slice_data)
                h5file.create_dataset('label', gt_dims, dtype=None, data=slice_gt_data)
                h5file.close()

        elif os.path.isfile(img_file):
            # Load only the image - this is for testing phase where we don't have ground truth
            pass

        else:
            print('There is no SA image file for {}'.format(img_filename))


if __name__ == "__main__":
    convert_nifti_to_2D()

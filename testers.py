# methods used to test clues
import os

import nibabel as nib
import numpy as np

import config
from loader import load_patient_images


def check_transpose():
    patient_image, _ = load_patient_images(3)
    patient_image = np.transpose(patient_image, (1, 2, 0))
    # patient_image = np.fliplr(patient_image)
    img_file = os.path.join(config.train_path, 'training_sa_crop_pat' + str(3) + '.nii.gz')
    image = nib.load(img_file).get_data()
    print("Images equal ", np.array_equal(patient_image, image))


if __name__ == "__main__":
    check_transpose()

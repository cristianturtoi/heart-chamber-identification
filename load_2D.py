import glob
import os
import random

import config

from keras.preprocessing.image import load_img, img_to_array


def get_shuffled_records():
    original_2D_path = os.path.join(config.data_root, "original_2D")

    train_no_sample = round(0.9 * config.nr_patients)
    test_no_sample = config.nr_patients - train_no_sample

    test_records = random.sample(range(1, config.nr_patients), test_no_sample)

    train_subjects = []
    train_subjects_gt = []
    test_subjects = []
    test_subjects_gt = []

    for id in range(config.nr_patients):
        patient_filename = "original_2D_p" + str(id) + "_*.png"
        gt_filename = "original_gt_2D_p" + str(id) + "_*.png"

        patients_slices = sorted(glob.glob(os.path.join(original_2D_path, patient_filename)))
        gt_slices = sorted(glob.glob(os.path.join(original_2D_path, gt_filename)))

        if id in test_records:
            test_subjects.extend(patients_slices)
            test_subjects_gt.extend(gt_slices)
        else:
            train_subjects.extend(patients_slices)
            train_subjects_gt.extend(gt_slices)

    return train_subjects, train_subjects_gt, test_subjects, test_subjects_gt


def load_img_to_array(img_path):
    """
    Loads image as grayscale and returns the image as a numpy array
    """
    img = load_img(img_path, color_mode='grayscale')
    return img_to_array(img)


if __name__ == "__main__":
    train_subjects, train_subjects_gt, test_subjects, test_subjects_gt = get_shuffled_records()

import os

code_root = "/home/cristi/PycharmProjects/heart-chamber-identification"
data_root = "/home/cristi/PycharmProjects/segmentation/datasets/HVSMR2016/Training short-axis cropped"
train_path = os.path.join(data_root, "Training dataset")
label_path = os.path.join(data_root, "Ground truth")

nr_patients = 10
im_height = 192
im_width = 192
MODEL_CHECKPOINT = 'model-hvsmr.h5'
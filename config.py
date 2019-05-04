import os

code_root = "/home/cristi/PycharmProjects/heart-chamber-identification"
data_root = "/home/cristi/PycharmProjects/segmentation/datasets/HVSMR2016/Training short-axis cropped"
train_path = os.path.join(data_root, "Training dataset")
label_path = os.path.join(data_root, "Ground truth")
test_path = "/home/cristi/PycharmProjects/segmentation/datasets/HVSMR2016/Test short-axis cropped/Test dataset"

nr_patients = 10
im_height = 192
im_width = 192
batch_size = 24
seed = 2019
classes = 3
weights = [0.1, 0.45, 0.45]
epochs = 200
MODEL_CHECKPOINT = 'model-hvsmr.h5'

### TRAINING ###
# optimizer options
optimizer = "adam"
learning_rate = 0.001
momentum = None  # Used only by SGD optimizer
decay = 0.

# augmentation
validation_split = 0.1
rotation_range = 180
width_shift_range = 0.1
height_shift_range = 0.1
shear_range = 0.1
zoom_range = 0.05
fill_mode = 'nearest'
vertical_flip = True
horizontal_flip = True
alpha = 500
sigma = 20

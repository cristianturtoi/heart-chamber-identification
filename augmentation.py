import numpy as np

import config
import helper
import loader
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def crop_images(images, crop_size):
    resized_images = np.zeros((len(images), config.crop_size, config.crop_size, 1), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        resized_images[idx] = helper.center_crop(img, crop_size)[:, :, np.newaxis]

    print(np.unique(resized_images))
    return resized_images


def load_preprocess_data(crop_size):
    """
    Load, crop (resize) and normalize data
    :return:
    """
    X_train, X_valid, y_train, y_valid = loader.read_images_with_groundtruth()
    X_train = crop_images(X_train, crop_size)
    X_valid = crop_images(X_valid, crop_size)
    y_train = crop_images(y_train, crop_size)
    y_valid = crop_images(y_valid, crop_size)
    return helper.normalize(X_train), helper.normalize(X_valid), y_train, y_valid


def random_elastic_deformation(image, alpha, sigma, mode='nearest',
                               random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x + dx), channels),
               np.repeat(np.ravel(y + dy), channels),
               np.tile(np.arange(channels), height * width))

    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))


class Iterator(object):
    def __init__(self, images, masks, batch_size,
                 shuffle=True,
                 rotation_range=180,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.01,
                 fill_mode='nearest',
                 horizontal_flip=True,
                 vertical_flip=True,
                 alpha=500,
                 sigma=20):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
            'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip,
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.alpha = alpha
        self.sigma = sigma
        self.fill_mode = fill_mode
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))

        augmented_images = []
        augmented_masks = []
        for n in self.index[start:end]:
            image = self.images[n]
            mask = self.masks[n]

            _, _, channels = image.shape

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((image, mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.idg.random_transform(stacked)

            # maybe apply elastic deformation
            if self.alpha != 0 and self.sigma != 0:
                augmented = random_elastic_deformation(
                    augmented, self.alpha, self.sigma, self.fill_mode)

            # split image and mask back apart
            augmented_image = augmented[:,:,:channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:,:,channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def create_generators(batch_size, crop_size, validation_split=0.1, shuffle=True, seed=None, augment_training=True, augmentation_args={}):
    images, masks = loader.load_patients()

    # resize images to have the same size
    images = crop_images(images, crop_size)
    masks = crop_images(masks, crop_size)

    # normalize images using mean variance normalization
    images = helper.normalize(images)

    if seed is not None:
        np.random.seed(seed)

    # shuffle images and masks in parallel
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(masks)

    # split out last %(validation_split) of images as validation set
    split_index = int((1 - validation_split) * len(images))

    if augment_training:
        train_generator = Iterator(
            images[:split_index], masks[:split_index],
            batch_size, shuffle=shuffle, **augmentation_args)
    else:
        idg = ImageDataGenerator()
        train_generator = idg.flow(images[:split_index], masks[:split_index],
                                   batch_size=batch_size, shuffle=shuffle)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        idg = ImageDataGenerator()
        val_generator = idg.flow(images[split_index:], masks[split_index:], batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)

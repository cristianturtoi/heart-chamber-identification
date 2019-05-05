import nibabel as nib
import numpy as np
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from math import ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import config


def reshape_2D(ndarray, to_shape):
    """Reshapes a center cropped (or padded) array back to its original shape."""
    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape
    if h_in > h_out:  # center crop along h dimension
        h_offset = (h_in - h_out) // 2
        ndarray = ndarray[h_offset:(h_offset + h_out), :]
    else:  # zero pad along h dimension
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    if w_in > w_out:  # center crop along w dimension
        w_offset = (w_in - w_out) // 2
        ndarray = ndarray[:, w_offset:(w_offset + w_out)]
    else:  # zero pad along w dimension
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem)
        npad = ((0, 0), pad_dim_w)
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    return ndarray  # reshaped


def center_crop(ndarray, crop_size):
    """Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    """
    h, w = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w)
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w = ndarray.shape
    # center crop
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    cropped = ndarray[h_offset:(h_offset + crop_size), w_offset:(w_offset + crop_size)]

    return cropped


def check_gpu_usage():
    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())


def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(optimizer_name))
    return optimizers[optimizer_name](**optimizer_args)


def create_optimizer():
    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr': config.learning_rate,
        'momentum': config.momentum,
        'decay': config.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(config.optimizer, optimizer_args)
    return optimizer


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
            augmented_image = augmented[:, :, :channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:, :, channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def normalize(x, epsilon=1e-7, axis=(1,2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon
    return x


def load_images(data_dir, mask):
    return data_dir, mask


def create_generators(data_dir, batch_size, validation_split=0.0, mask='both',
                      shuffle_train_val=True, shuffle=True, seed=None,
                      normalize_images=True, augment_training=False,
                      augment_validation=False, augmentation_args={}):
    images, masks = load_images(data_dir, mask)

    # before: type(masks) = uint8 and type(images) = uint16
    # convert images to double-precision
    images = images.astype('float64')

    # maybe normalize image
    if normalize_images:
        normalize(images, axis=(0,1))

    if seed is not None:
        np.random.seed(seed)

    if shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(masks)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(images))

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
        if augment_validation:
            val_generator = Iterator(
                images[split_index:], masks[split_index:],
                batch_size, shuffle=shuffle, **augmentation_args)
        else:
            idg = ImageDataGenerator()
            val_generator = idg.flow(images[split_index:], masks[split_index:],
                                     batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)


def normalize_3d(img, filename):
    # Normalize the intensity of images
    mask = img > 0
    mean_value = np.mean(img[mask])
    std_value = np.std(img[mask])
    img = (img - mean_value) / std_value
    img = np.float32(img)
    # seg = np.uint(seg)
    print(filename, np.unique(img))
    return img


def save_nii(img_path, data, output_affine, output_header):
    nimg = nib.Nifti1Image(data, affine=output_affine, header=output_header)
    nimg.to_filename(img_path)
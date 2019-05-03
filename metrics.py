from medpy.metric.binary import hd, assd
from keras import backend as K


def hausdorff_distance(gt, pred, zooms):
    return hd(gt, pred, voxelspacing=zooms, connectivity=1)


def average_symmetric_surface_distance(gt, pred, zooms):
    return assd(gt, pred, voxelspacing=zooms, connectivity=1)


def dice_coef(gt, pred, axis=None, smooth=1):
    intersection = K.sum(gt * pred, axis=axis)
    union = K.sum(gt, axis=axis) + K.sum(pred, axis=axis)
    return (2 * intersection + smooth) / (union + smooth)


def dice_loss(gt, pred, axis=None, smooth=1):
    return 1 - dice_coef(gt, pred, axis, smooth)


def jaccard_coef(gt, pred, axis=None, smooth=1):
    intersection = K.sum(gt * pred, axis=axis)
    area_true = K.sum(gt, axis=axis)
    area_pred = K.sum(pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_loss(gt, pred, axis=None, smooth=1):
    return 1 - jaccard_coef(gt, pred, axis, smooth)

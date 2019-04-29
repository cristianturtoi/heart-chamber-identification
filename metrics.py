from medpy.metric.binary import hd, dc, assd
from keras import backend as K


def hausdorff_distance(gt, pred, zooms):
    return hd(gt, pred, voxelspacing=zooms, connectivity=1)


def hausdorff_loss(gt, pred):
    return hd(gt, pred)


def dice_coefficient(gt, pred):
    return dc(gt, pred)


def dice_loss(gt, pred):
    1 - dice_coefficient(gt, pred)


def dice_coefficient_smooth(gt, pred, smooth=0.0):
    y_true_flat = K.flatten(gt)
    y_pred_flat = K.flatten(pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat)
    return (2. * intersection + smooth) / (union + smooth)


def dice_loss_smooth(gt, pred, smooth=0.0):
    return dice_coefficient_smooth(gt, pred, smooth)


def average_symmetric_surface_distance(gt, pred, zooms):
    return assd(gt, pred, voxelspacing=zooms, connectivity=1)


def assd_loss(gt, pred):
    return assd(gt, pred)


def jaccard_coefficient(gt, pred, axis=None, smooth=0.0):
    intersection = K.sum(gt * pred, axis=axis)
    area_true = K.sum(gt, axis=axis)
    area_pred = K.sum(pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_loss(gt, pred, axis=None, smooth=0.0):
    return 1 - jaccard_coefficient(gt, pred, axis, smooth)

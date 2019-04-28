import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize

import config
import load_2D
from config import MODEL_CHECKPOINT
from deeplabv3plus import Deeplabv3


def resize_gt_imgs(images):
    resized_imgs = np.zeros((len(images), config.im_height, config.im_width, 1), dtype=np.uint8)
    for n, img_path in enumerate(images):
        img = load_img(img_path, color_mode='grayscale')
        img = img_to_array(img)
        reImg = resize(img, (config.im_height, config.im_width, 1), mode='constant', preserve_range=True)
        resized_imgs[n] = reImg
    print(np.unique(resized_imgs))
    return resized_imgs


def get_resized_data():
    X_train, X_valid, y_train, y_valid = load_2D.load_data_shuffled()
    return resize_gt_imgs(X_train), resize_gt_imgs(y_train), resize_gt_imgs(X_valid), resize_gt_imgs(y_valid)


X_train, X_valid, y_train, y_valid = get_resized_data()
print(np.unique(y_train))

input_img = Input((config.im_height, config.im_width, 1), name='img')
model = Deeplabv3(input_shape=(config.im_height, config.im_width, 1), classes=1, weights=None)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=16, epochs=50, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plt.show()

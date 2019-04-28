from keras.optimizers import Adam

import config
from deeplabv3plus import Deeplabv3

model = Deeplabv3(input_shape=(config.im_height, config.im_width, 3), classes=3, weights=None)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

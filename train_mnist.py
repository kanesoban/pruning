from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


BATCH_SIZE = 32
mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


INPUT_SHAPE = (28, 28, 1)
train_images = np.expand_dims(x_train / 255.0, axis=-1)
train_labels = np.expand_dims(y_train / 255.0, axis=-1)
test_images = np.expand_dims(x_test / 255.0, axis=-1)
test_labels = np.expand_dims(y_test / 255.0, axis=-1)


model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=BATCH_SIZE, epochs=10)

model.save("model.h5")

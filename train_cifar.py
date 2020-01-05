from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Is this eager mode ?
tf.executing_eagerly()
assert (tf.__version__ == '2.0.0')
tf.config.experimental_run_functions_eagerly(True)


BATCH_SIZE = 32
mnist = datasets.mnist

INPUT_SHAPE = (32, 32, 3)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


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

model.save("model_dir")


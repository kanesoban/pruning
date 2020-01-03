from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import numpy as np


BATCH_SIZE = 32
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


data_train_batch = np.expand_dims(x_train / 255.0, axis=-1)
label_train_batch = np.expand_dims(y_train / 255.0, axis=-1)
data_test_batch = np.expand_dims(x_train / 255.0, axis=-1)
label_test_batch = np.expand_dims(y_train / 255.0, axis=-1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data_train_batch, label_train_batch, validation_data=(data_test_batch, label_test_batch), batch_size=BATCH_SIZE, epochs=10)

model.save("model.h5")

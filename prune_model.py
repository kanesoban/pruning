from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import numpy as np

# Is this eager mode ?
tf.executing_eagerly()
assert (tf.__version__ == '2.0.0')
tf.config.experimental_run_functions_eagerly(True)

from pruning import prune_model


BATCH_SIZE = 32
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


data_train = np.expand_dims(x_train / 255.0, axis=-1)
label_train = np.expand_dims(y_train / 255.0, axis=-1)
data_test = np.expand_dims(x_train / 255.0, axis=-1)
label_test = np.expand_dims(y_train / 255.0, axis=-1)

model = tf.keras.models.load_model('model.h5')

prune_iterations = 1
model = prune_model(model, data_train, label_train, data_test, label_test, BATCH_SIZE, prune_iterations)

model.save("pruned_model.h5")

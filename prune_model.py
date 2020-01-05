from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np

# Is this eager mode ?
tf.executing_eagerly()
assert (tf.__version__ == '2.0.0')
tf.config.experimental_run_functions_eagerly(True)

from pruning import prune_model


BATCH_SIZE = 32

INPUT_SHAPE = (32, 32, 3)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.models.load_model('model.h5')

prune_iterations = 10
model = prune_model(model, train_images, train_labels, test_images, test_labels, BATCH_SIZE, prune_iterations)

model.save("pruned_model.h5")

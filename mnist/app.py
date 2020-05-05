from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import Sequential, layers
from IPython import display
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import os
import PIL
import time

# download Mnist data set
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# making data batch and suffle
trian_dataset = tf.data.Dataset.from_tensor_slices(train_images).suffle(BUFFER_SIZE).batch(BATCH_SIZE)

def generator_model():
  model = Sequential()
  model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100, )))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256) # *Batch size is None!!

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model
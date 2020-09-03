from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import Sequential, layers, losses
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

# making data batch and shuffle
trian_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def generator_model():
  model = Sequential()
  model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100, )))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256) # *Batch size is None!!

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model

def discriminator_model():
  model = Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

def discriminator_loss(real, fake):
  cross_entropy = losses.BinaryCrossentropy(from_logits=True)

  real_loss = cross_entropy(tf.ones_like(real), real)
  fake_loss = cross_entropy(tf.ones_like(real), real)

  total_loss = real_loss + fake_loss

  return total_loss

def main():
  generator = generator_model()

  noise = tf.random.normal([1, 100])
  generated_image = generator(noise, training=False)

  plt.imshow(generated_image[0, :, :, 0], cmap='gray')

  discriminator = discriminator_model()
  decision = discriminator(generated_image)
  print (decision)

  
if __name__ == "__main__":
  main()
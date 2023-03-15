
import matplotlib.pyplot as plt
from matplotlib.image import imread

#plotting different datapoints
# for i in range(9):
#  plt.subplot(330 + 1 + i)
#  filename = 'Cat/' + str(i) + '.jpg'
#  image = imread(filename)
#  plt.imshow(image)
# plt.show()

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator

def train_model():
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_iterator = train_gen.flow_from_directory('train/',
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('validation/',
                                                             target_size=(150, 150),
                                                             batch_size=10,
                                                             class_mode='binary')

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        steps_per_epoch=1000,
                        epochs=50,
                        validation_steps=500)

    model.save('dogs-vs-cats.h5')

    return history

def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

results = train_model()
plot_result(results)

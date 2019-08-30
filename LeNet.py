from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.datasets import mnist

import numpy as np


class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential

        # conv > relu > pooling
        # first  wave
        model.add(Conv2D(20,kernel_size=5,padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        # second wave, set the filter to 50
        model.add(Conv2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D( pool_size=(2, 2), strides=(2, 2)))

        # third wave
        model.add(Conv2D(70, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # dense layer, final layer

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # soft max classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

num_epoch = 20
batch_szie = 128
num_labels = 10
optimizer = Adam()
validation_split = 0.2
verbose = 1
image_rows, image_cols = 28, 28
input_shape = (1, image_rows, image_rows)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_train.astype('float32')
test_images = x_test.astype('float32')
images /= 255
test_images /= 255
images = images[:, np.newaxis, :, :]
test_images = test_images[:, np.newaxis, :, :]

# convert labels to numerical vectors
labels = y_train
one_hot_labels = np.zeros(len(labels), num_labels)
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_labels = np.zeros(len(y_test), num_labels)
for i, l in enumerate(y_test):
    test_labels[i][l] = 1
# OR
labels1 = np_utils.to_categorical(y_train, num_labels)
t_labels = np_utils.to_categorical(y_test, num_labels)

# Initialise the optimiser and the model

model = LeNet.build(input_shape=input_shape, classes=num_labels)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
history = model.fit(images, labels, batch_size=batch_szie, epochs=num_epoch, verbose=verbose,
                    validation_split=validation_split)
score = model.evaluate(images, labels, verbose=verbose)

from keras.datasets import mnist
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.utils import np_utils
# this would have been useful in converting the labels to categorical features but i already did that with a simple
# python for loop.
from keras.regularizers import l2
np.random.seed(1)

def relu(x):
    return (x >= 0) * x


def relu2deriv(output):
    return output > 0


def soft_max(x):
    temp = np.exp(x)
    return temp / np.sum(temp, keepdims=True)


def get_images_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section[-1:1, row_from, row_to - row_from, col_from, col_to - col_from]


x_train, x_test, y_train, y_test = mnist.load_data()
images = x_train[0:40000].reshape(28 * 28) / 255
labels = y_train[0:40000]
one_hot_labels = np.zeros(len(labels), 10)
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28, 28)
test_labels = np.zeros(len(y_test), 10)
for i, l in enumerate(test_labels):
    test_labels[i][l] = 1

epochs, batch_size = 200, 128  # epochs refers to the number of iterations, batch_size refers to batch gd config
num_labels, verbose, n_hidden = (10, 1, 128)  # num labels is the num of categories in the labels
# n hidden is the number of hidden layers.
reshaped = 28 * 28  # reshaped shape lol
drop_out = 3  # a regularization parameter
validation_split = 0.2  # We reserved part of the training set for validation. The key idea is that we reserve a
# part of the training data for measuring the performance on the validation while training.
optimizer = SGD()  # optimizer function
optimizer1 = RMSprop()
optimizer2 = Adam()

# model
model = Sequential()
model.add(Dense(n_hidden, input_shape=28*28, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(drop_out))
model.add(Dense(n_hidden, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(drop_out))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer2,
              metrics=['accuracy'],)

history = model.fit(images, labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
                    validation_split=validation_split)

score = model.evaluate(test_images, test_labels, verbose=verbose)

print('test score', score[0])
print('Test Accuracy', score[1])

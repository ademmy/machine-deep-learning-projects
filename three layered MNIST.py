import numpy as np
import sys
from keras.datasets import mnist
def relu(x):
    return(x >= 0) * x
def relu2deriv(output):
    return output >= 0

x_train, x_test, y_train, y_test = mnist.load_data()

images = x_train[0:1000].reshape(1000, 28*28)/255
labels = y_train[0:1000]
one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28*28) / 255

test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1


np.random.seed(1)
alpha, iterations, hidden_size, pixels_per_image, num_labels = (0.001, 300, 40, 784, 10)
batch_size = 100  # batch is added to increase the speed of training and rate of convergence,
# this is the process of batch gradient descent
weights0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for k in range(int(len(images) / batch_size)):  # the only difference between batch gd and stochastic gd is the,
        # addition of batch_size in the code.
        batch_start, batch_end = ((1 * batch_size), ((k + 1) * batch_size))
        layer_0 = images[batch_start: batch_end]
        layer_1 = relu(np.dot(layer_0, weights0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)  # regularizing with dropout
        layer_1 *= 2 * dropout_mask
        layer_2 = np.dot(layer_1, weights1_2)

        error += np.sum((labels[batch_start: batch_start] - layer_2) ** 2)
        for i in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k + 1]) == np.argmax(labels[batch_start + k:batch_start + k + 1]))

            layer2_delta = (labels[batch_start: batch_end] - layer_2) / batch_size
            layer1_delta = layer2_delta.dot(weights1_2.T) * relu2deriv(layer_1)
            layer1_delta *= dropout_mask

            weights1_2 += alpha * layer_1.T.dot(layer2_delta)
            weights0_1 += alpha * layer_0.T.dot(layer1_delta)

    sys.stdout.write("\r" + " I:" + str(j) + " Error:" + str(error / float(len(images)))[0:5] + " Correct:" +
                     str(correct_cnt / float(len(images))))

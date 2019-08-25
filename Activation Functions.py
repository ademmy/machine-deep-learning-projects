from keras.datasets import mnist
import numpy as np
np.random.seed(1)

# Various Activation Functions.
def relu(x):
    return (x >= 0) * x

def relu2deriv(output):  # basically is used to find the delta in the back propagation phase.
    return output >= 0

def tanh(x):  # tanh hidden layer activation function
    return np.tanh(x)
def tanh2deriv(output):  # tanh delta back propagation phase
    return 1 - (output ** 2)

def softmax(x):  # efficient for output layers.
    temp = np.exp(x)
    return temp / np.sum(temp, keepdims=True, axis=1)


x_train, x_test, y_train, y_test = mnist.load_data()

images = x_train[0:1000].reshape(28 * 28) / 255
labels = x_train[0:1000]
one_hot_labels = np.zeros(len(labels), 10)

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = y_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros(len(y_test), 10)
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

alpha, iterations, num_labels = (0.001, 300, 10)
hidden_size, batch_size, pixels_per_image = (150, 100, 784)

weights0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
weights1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    correct_cnt = 0
    for k in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((1 * batch_size), ((k+1) * batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = tanh(np.dot(layer_0, weights0_1))
        layer_2 = softmax(np.dot(layer_1, weights1_2))
        dropout = np.random.randint(2, size=len(layer_1.shape))
        layer_2 *= dropout * 2
        for l in range(batch_size):
            label_set = labels[batch_start + l:  batch_start + l + 1]
            correct_cnt += int(np.argmax(layer_2[l:l+1]) == np.argmax(label_set))
        layer_2_delta = (labels[batch_start: batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout

        weights1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights0_1 += alpha * layer_0.T.dot(layer_1_delta)

# we got rid of the mean because softmax makes use of an error calculation procedure called cross entropy

test_correct_cnt = 0
for h in range(len(test_images)):
    test_layer0 = test_images[h: h+1]
    test_layer1 = tanh(np.dot(test_layer0, weights0_1))
    test_layer2 = np.dot(test_layer1, weights1_2)
    test_correct_cnt += int(np.argmax(test_layer2) == np.argmax(test_labels[h: h+1]))


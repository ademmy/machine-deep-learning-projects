import numpy as np
from keras.datasets import mnist

def tanh(x):
    return np.tanh(x)
def tan2deriv(output):
    return 1 - (output ** 2)

def soft_max(x):
    temp = np.exp(x)
    return temp / np.sum(temp, keepdims=True, axis=1)

x_train, x_test, y_train, y_test = mnist.load_data()

images = x_train[0:1000].reshape(28 * 28)/255
labels = y_train[0:1000]
one_hot_labels = np.zeros[len(labels), 10]
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_image = x_test.reshape(len(x_test), 28 * 28)
test_labels = np.zeros(len(y_test), 10)
for i, l in enumerate(test_labels):
    test_labels[i][l] = 1

alpha, iterations, pixels_per_image = (2, 300, 784)
num_labels = 10
kernel_rows, kernel_cols = (3, 3)
num_of_kernels = 16
batch_size = 128
input_cols, input_rows = (28, 28)
hidden_size = ((input_cols - kernel_cols) * (input_rows - kernel_rows)) * num_of_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_of_kernels)) - 0.01
weights1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

def get_image_section(layer, row_from, row_to, col_from, col_to):
    # this function is what sets the various kernels to different parts of the image.
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

for j in range(iterations):
    correct_cnt = 0
    for k in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((1 * batch_size), ((1 + 1) * batch_size))
        layer_0 = images[batch_start: batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        print(layer_0.shape)

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)
        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= 2 * dropout_mask
        layer_2 = soft_max(np.dot(layer_1, weights1_2))

        for h in range(batch_size):
            label_set = labels[batch_start + h:batch_start + h + 1]
        _inc = int(np.argmax(layer_2[h:h + 1]) == np.argmax(label_set))
        correct_cnt += _inc

        layer2_delta = ((labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0]))
        layer1_delta = layer2_delta.dot(weights1_2.T) * tan2deriv(layer_1)
        layer1_delta *= dropout_mask
        # weight update format: first multiply input by delta,
        # multiply product by alpha,
        # then add the weights to the solution
        weights1_2 += alpha * layer_1.T.dot(layer2_delta)
        lid_reshape = layer1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(lid_reshape)
        kernels = alpha * k_update

    test_correct_cnt = 0

    for i in range(len(test_image)):
        test_l0 = test_image[i: i+1]
        test_l0 = test_l0.reshape(test_l0.shape[0], 28, 28)

        test_sects = 0
        for row_start in range(test_l0.shape[0] - kernel_rows):
            for col_start in range(test_l0.shape[1] - kernel_cols):
                test_sect = get_image_section(test_l0, row_start, row_start + kernel_rows, col_start, col_start + kernel_rows)
            test_sects.append(test_sect)










from keras.datasets import mnist
import numpy as np

x_train, x_test, y_train, y_test = mnist.load_data()

images = x_train[0: 2000].reshape(28, 28) / 255
labels = y_train[0: 2000]
one_hot_labels = np.zeros(len(labels), 10)
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28, 28)
test_labels = np.zeros(len(y_test), 10)
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

def relu(x):
    return (x >= 0) * x
def relu2deriv(output):
    return output > 0
def soft_max(x):
    temp = np.exp(x)
    return temp / np.sum(temp, keepdims=True)
def tanh(x):
    return np.tanh(x)

def get_images_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from: row_to, col_from:col_to]
    return section.reshape[-1, 1, row_from - row_to, col_from - col_to]

alpha, iterations = (2, 300)
batch_size, num_labels = 200, 10
row_size, col_size = 28, 28
row2_size, col2_size = 14, 14
kernel_row, kernel_col = 3, 3
kernel_size = 16
hidden_size = ((row_size - kernel_row) * (col_size - kernel_col)) * kernel_size

weight1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1
kernels = 0.02 * np.random.random(row_size * col_size) - 0.01
kernel1 = 0.02 * np.random.random(row2_size * col2_size) - 0.01


for i in range(iterations):
    correct_cnt = 0
    for k in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((1 * batch_size), (1 + 1) * batch_size)
        layer_0 = images[batch_start: batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        sects = list()
        for row_start in range(layer_0.shape[1]):
            for col_start in range(layer_0.shape[2]):
                sect = get_images_section(layer_0, row_start, row_start + kernel_row, col_start, col_start + kernel_col)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        kernel_output = flattened_input.dot(kernels)

        layer_1 = relu(kernel_output.reshape(es[0], -1))
        layer_1 = layer_1.reshape(layer_1.shape[0], 14, 14)

        sects1 = list()
        for rows in range(layer_1.shape[1]):
            for cols in range(layer_1[2]):
                sect1 = get_images_section(layer_1, rows, rows + kernel_row, cols, cols + kernel_col)
                sects1.append(sect1)
        exp = np.concatenate(sects1, axis=1)
        ex = exp.shape
        flat_ins = exp.reshape(ex[0] * ex[1], -1)
        k_out = flat_ins.dot(kernel1)

        dropout_mask = np.random.randint(2, size=layer_1.shape)
        k_out *= 2 * dropout_mask
        layer_2 = relu(k_out.reshape(ex[1], -1))
        layer_3 = soft_max(layer_2, weight1_2)

        for h in range(batch_size):
            label_set = labels[batch_size + h: batch_size + h + 1]
            _inc = int(np.argmax(layer_3[h: h + 1] == np.argmax(label_set)))
        correct_cnt += _inc

        layer3_delta = (labels[batch_start: batch_end] - layer_3) / (batch_size * layer_3.shape[0])
        layer2_delta = layer3_delta.dot(weight1_2.T) / relu2deriv(layer_1)
        layer1_delta = layer2_delta.dot

        weight1_2 += alpha * layer_1.T.dot(layer2_delta)
        lid = layer1_delta.reshape(kernel_output.shape)
        kernel_update = flattened_input.T.dot(lid)
        kernels -= alpha * kernel_update








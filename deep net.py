import numpy as np

def zeroes_vector(a, b):
    output = np.zeros(len(a), len(b))
    return output


def vector_multiplication(vec_a, vec_b):
    output = zeroes_vector(vec_a, vec_b)
    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            output[i][j] = vec_a[i] * vec_b[j]
    return output


weights = np.array([0.5, 0.48, -0.7])
alpha1 = 0.1
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
input1 = streetlights[0]
goal_pred = walk_vs_stop[0]
for iter in range(20):
    prediction = input1.dot(weights)
    error1 = (prediction - goal_pred) ** 2
    delta1 = prediction - goal_pred
    weights = weights - (alpha1 * (input1 * delta1))
   # print('this is the value of the prediction')
   # print(prediction)
   # print('this is the updated weight')
   # print(weights)
# print('Done!')
print()
print()
# learning the entire data set with stochastic gradient descent

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
input1 = streetlights[0]
goal_pred = walk_vs_stop[0]
weights2 = np.array([0.5, 0.48, -0.7])
for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input2 = streetlights[row_index]
        output2 = walk_vs_stop[row_index]
        prediction2 = input2.dot(weights2)
        error2 = (prediction2 - output2) ** 2
        error_for_all_lights += error2
        delta2 = prediction2 - output2
        weights2 = weights2 - (alpha1 * (input2 * delta2))
        #print('prediction: {}'.format(prediction2))
        #print('updated weights: {}'.format(weights2))
    #print('error value: {}'.format(error_for_all_lights))
# forward propagation
np.random.seed(1)


def relu(x):
    return (x > 0) * x


alpha = 0.2
hidden_size = 4
weights0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights1_2 = 2 * np.random.random((hidden_size, 1)) - 1
streetlights1 = np.array([[1, 0, 1],
                          [0, 1, 1],
                          [0, 0, 1],
                          [1, 1, 1]])
walk_vs_stop1 = np.array([[1, 1, 0, 0]]).T
layer0 = streetlights1[0]
layer1 = relu(np.dot(layer0, weights0_1))
layer_2 = np.dot(layer1, weights1_2)

# a full neural network arrangement
def relu2deriv(output):  # returns 1 for input greater than 0 and 0 for input less than 0
    return output > 0
weights0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer2_error = 0
    for i in range(len(streetlights1)):
        layer_0 = streetlights1[i: i+1]
        layer_1 = relu(np.dot(layer_0, weights0_1))
        layer2 = np.dot(layer_1, weights1_2)

        layer2_error += np.sum((layer2 - walk_vs_stop1[i:i+1])**2)
        layer2_delta = (layer2 - walk_vs_stop1[i: i+1])
        layer1_delta = layer2_delta.dot(weights1_2.T)*relu2deriv(layer_1)

        weights1_2 -= alpha * layer_1.T.dot(layer2_delta)
        weights0_1 -= alpha * layer_0.T.dot(layer1_delta)

        if (iteration % 10 == 9):
            print('the error: {}'.format(layer2_error))



import numpy as np
def weighted_sum(a, b):
    output = 0
    assert(len(a) == len(b))
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

def element_wise_multiplication(number, vector):
    output = [0, 0, 0]
    assert(len(vector) == len(output))

    for i in range(len(vector)):
        output[i] = number * vector[i]

    return output

def vector_matrix_multiply(vector, matrix):
    output = [0, 0, 0]
    assert(len(output) == len(vector))

    for i in range(len(vector)):
        output[i] = weighted_sum(vector, matrix[i])
    return output

# Gradient Descent learning with multiple inputs and outputs

def neural_networks(weight, input):
    pred = weighted_sum(input, weight)
    return pred

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

weights = [0.1, 0.2, -0.1]
alpha = 0.01
win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]
input = [toes[0], wlrec[0], nfans[0]]

pred = neural_networks(input, weights)
error = (pred - true) ** 2
delta = pred - true
w_delta = element_wise_multiplication(delta, input)

for i in range(len(weights)):
    """"once you're using alpha, when updating the weights
        use a minus sign before the equals to sign to update it """
    weights[i] -= alpha * w_delta[i]

# print('the prediction is {}'.format(pred))
# print('the weight of the nn is {}'.format(weights))

# freezing one weight
def neural_net(input, weight):
    out = 0
    for i in range(len(input)):
        out += (input[i] * weight[i])
    return out

def ele_mul(scalar, vector):
    output = [0, 0, 0]
    for i in range(len(output)):
        output[i] += vector[i] * scalar
    return output

toes1 = [8.5, 9.5, 9.9, 9.0]
wlrec1 = [0.65, 0.8, 0.8, 0.9]
nfans1 = [1.2, 1.3, 0.5, 1.0]

ins = [toes1[0], wlrec1[0], nfans1[0]]
win_or_lose_binary_1 = [1, 1, 0, 1]
true1 = win_or_lose_binary_1[0]
weights3 = [0.1, 0.2, -.1]
alpha2 = 0.3

for iter in range(3):
    pred2 = neural_net(ins, weights3)
    error3 = (pred - true1) ** 2
    delta = pred - true1
    weight3_delta = ele_mul(delta, ins)
    #print("Iteration:" + str(iter + 1))
    #print("Pred:" + str(pred2))
    #print("Error:" + str(error3))
    #print("Delta:" + str(delta))
    #print("Weights:" + str(weights3))
    #print("Weight_Deltas:")
    #print(str(weight3_delta))
    #print()

    for i in range(len(weights3)):
        weights3[i] -= (alpha2 * weight3_delta[i])

""" A neural network with multiple outputs """

hurt = [0.1, 0.0, 0.0, 0.1]
win = [ 1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

def nn(input, weights):
    out = ele_mul(input, weights)
    return out
ins1 = wlrec1[0]
weight = [0.3, 0.2, 0.9]
true2 = [hurt[0], win[0], sad[0]]
predict = nn(input =ins1, weights =weight)
error1 = [0, 0, 0]
delta = [0, 0, 0]
print(predict)

for i in range(len(true2)):
    error1[i] = (predict[i] - true2[i]) ** 2
    delta[i] = predict[i] - true2[i]

weight_delta = element_wise_multiplication(ins1, delta)
alpha1 = 0.1
for i in range(len(weight)):
    weight[i] -= (alpha1 * weight_delta[i])
print()
print()
print("Weights:" + str(weight))
print("Weight Deltas:" + str(weight_delta))

"""A neural network with multiple inputs and outputs"""
weights_ = [[0.1, 0.1, -0.3],# hurt?
[0.1, 0.2, 0.0],  # win?
[0.0, 1.3, 0.1]]  # sad?
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
hurt = [0.1, 0.0, 0.0, 0.1]
win = [ 1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]
alpha_ = 0.01

ins_ = [toes[0], wlrec[0], nfans[0]]
truth = [hurt[0], win[0], sad[0]]
def neural_net( inputs1, multi_weights):
    prediction = vector_matrix_multiply(inputs1, multi_weights)
    return prediction

predd = neural_net(ins, weights_)
print('multiple ins and outs nn {}'.format(predd))
error2 = [0, 0, 0]
delta1 = [0, 0, 0]

for i in range(len(truth)):
    error2[i] = (predd[i] - truth[i]) ** 2
    delta1 = (predd[i] - truth[i])

def zeros_matrix(a, b):
    output = np.zeros(len(a), len(b))
    return output


def outer_prod(vec_a, vec_b):
    out = zeros_matrix(len(vec_a), len(vec_b))
    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            out[i][j] = vec_a[i] * vec_b[j]
    return out

weight_delta1 = outer_prod(ins_, delta1)
weights -= weights - (alpha_ * weight_delta1)
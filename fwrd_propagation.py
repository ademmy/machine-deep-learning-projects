weight = 0.1
inputs = 8.5

def neural_network(weight, inputs):
    prediction = weight * inputs
    return prediction

print('a single neural network:', neural_network(weight, inputs))

""" a function to calculate the weighted sum of inputs """
def w_sum(a, b):
    assert(len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])
        return output

"""
the next set of code describes a nn with multiple inputs
"""

def nn(weights, inputs):
    pred= w_sum(weights, inputs)
    return round(pred, 2)

# input features


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]  # fan count is in millions

inputs = [toes[0], wlrec[0], nfans[0]]
w1 = [0.3, 0.2, 0.9]
print('a prediction of a neural network with multiple inputs:', nn(w1, inputs))

""" a nn with multiple outputs"""
def ele_mul(num, vec):
    output= [0,0,0]
    assert(len(output) == len(vec))
    for i in range(len(vec)):
        output[i]= num * vec[i]
    return output

def neural_n(inputs, weights):

    pred = ele_mul(inputs, weights)

    return pred


ins = wlrec[1]
w2 = [0.2,0.3, 0.9]
pred = neural_n(ins, w2)
print('a prediction of a neural network with multiple outputs', pred)

""" Multiple Inputs and Multiple Outputs """

weights = [[0.1, 0.1, -0.3],  # hurt?
[0.1, 0.2, 0.0],   # win?
[0.0, 1.3, 0.1]]  # sad?

inps = [wlrec[1], toes[1], nfans[1]]

def nnn(input, weights):
    pred = v_mat_mul(input, weights)
    return pred

def wsum(a, b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] *b[i])
    return output



def v_mat_mul(vector, matrix):
    assert(len(vector) == len(matrix))
    output = [0, 0, 0]
    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])
    return output


predc = nnn(inps, weights)
print("this is a prediction of a neural_net with multiple inputs & multiple outputs:", predc)

""" predicting on predictions, the idea behind hidden layers"""

ih_wgt = [[0.1, 0.2, -0.1],# hid[0]
         [-0.1,0.1, 0.9],# hid[1]
         [0.1, 0.4, 0.1]]# hid[2]

hp_wgt = [[0.3, 1.1, -0.3],  # hurt?
         [0.1, 0.2, 0.0],  # win?
         [0.0, 1.3, 0.1]]  # sad?

w4 = [ih_wgt, hp_wgt]
inps = [toes[1], wlrec[1], nfans[1]]


# the difference between element wise multiplication and vector multiplication is that:
# in this case we are multiplying a scalar with a vector


def ele_wise_mul(number,  vector):
    output = [0, 0, 0]
    assert(len(vector) == len(output))
    for i in range(len(vector)):

        output[i] = number * vector[i]
    return output

def hidden_net(inputs, weights):
    hid = v_mat_mul(inputs, weights[0])
    pred = v_mat_mul(hid, weights[1])
    return pred


pred = hidden_net(inps, w4)
print("this is a neural net with hidden nets", pred)

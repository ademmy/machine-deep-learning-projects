import numpy as np
class Activation_functions:

    def relu(x):
        return(x > 0) * x

    def sigmoid(x):
        return 1 /(1 + np.exp(-x))

    def tanh(x):
        return(np.tanh(x))

    def softmax(x):
        result = np.exp(x)
        return result / np.sum(result, keepdims = True)

acc = Activation_functions
print()

# this is an exaple for each activation function and how they are used and the result
x = [12, 3, 5, -2, 10, 0, 11, -12, -11, 34]
x = np.array(x)
y = np.random.random((3,3))
print(x, '\n')

# relu
print('the result for relu is ')
print(acc.relu(x))
print()
# sigmoid
print('the result for simoid is')
print(acc.sigmoid(y))
print()
# tanh
print('the result for tanh is')
print(acc.tanh(y))
print()
# softmax
print('the result for softmax is')
print(acc.softmax(y))

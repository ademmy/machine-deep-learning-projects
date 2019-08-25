weight = 0.5
inputs = 2
true_values = 0.8
alpha = 0.1

""""alpha is used to deal with situations when the input variable is quite large and can cause issues 
with weight updating and predictions. And alpha is multiplied with the derivative (delta)
if the error is increasing then the alpha is too large and vice versa

"""

for i in range(25):
    prediction = weight * inputs
    error = (prediction - true_values) ** 2
    delta = prediction - true_values
    weight_delta = inputs * delta
    weight = weight - (weight_delta * alpha)
    print('error: {} and prediction: {} '.format(error, prediction))


# hot and cold learning take 2
ins = 0.6
output = 0.9
weights = 0.55
step_size = 0.01

for i in range(20):

    pred = ins * weights
    errors = (pred - output)
    up_prediction = ins * (weight + step_size)
    up_error = up_prediction - output
    down_prediction = ins * (weights - step_size)
    down_error = down_prediction - output
    # find which error leads to us closer to the output
    if up_error > down_error:
        weights = weights + step_size
    else:
        weights = weights - step_size
        print(weights)


weight_1 = 0.5
input_1 = 2
alpha = 0.1
outputs = 0.8

for i in range(25):
    predictions = input_1 * weight_1
    error_2 = (predictions - outputs) ** 2
    delta = outputs - error_2
    w_delta = delta * input_1
    weight_1 = weight_1 - (w_delta * alpha)

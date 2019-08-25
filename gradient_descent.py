k_weight= 0.5
inps= 0.5
goal_pred= 0.8
pred= inps * k_weight
error= (pred - goal_pred) **2
print(round(error,4))

weight= 0.1
lr= 0.01
no_of_toes= [8.5]

def neural_network(weight, inputs):
    pred= inputs * weight
    return pred

inputs= no_of_toes[0]
win_lose= [1]
true= win_lose[0]
pred=neural_network(inputs, weight)
error= (pred-true)**2
print(round(error,4))

#hot and cold learning
""" the idea behind hot and cold learning is that the weights are adjusted
up or down and the direction that gives less errors is chosen
"""
weight = 0.5
input = 0.5
goal_prediction = 0.8
step_amount = 0.001
for i in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2

    # print("Error:" + str(error) + " Prediction:" + str(prediction))


    up_prediction = input * ( weight + step_amount)  # pred of the up side of things
    up_error = (goal_prediction - up_prediction) ** 2   # up error
    down_prediction = input * (weight - step_amount)  # pred of the down side
    down_error = (goal_prediction - down_prediction) ** 2   # down  error


    #this is how the weights are then tuned


    if(down_error < up_error):

        weight = weight - step_amount

    if (down_error > up_error):

        weight = weight + step_amount

print('this is the weight of the nn:{}'.format(round(weight,3)))

""" A LOOK AT GRADIENT DESCENT """
weight= 0.1
alpha= 0.01
def nn(inputs, weights):
    prediction= inputs * weights
    return prediction

inputs= no_of_toes[0]
win_lose= [1]
true= win_lose[0]
pred= nn(inputs, weight)
pure_error= (pred- true) **2
delta= pred- true #delta is used to show by how much you want to effect the change in your weights
weight_delta= inputs * delta #weight delta is the same as direction and amount as described in the book
weight -= weight_delta * alpha # alpha is the scaling parameter
#print(weight)

# how to update weights
weights, inps, true= [0,0.8,1.1]

for i in range(4):
    pred= weights * inps
    error = (pred - true) ** 2
    delta= pred - true
    w_delta= delta * inps
    weights = weights - w_delta
    # print("Error: " + str(error) + " Prediction: " + str(pred) + " Weights:" + str(weights))
    # print("Delta:" + str(delta) + " Weight Delta: " + str(w_delta))


# this is how you update the weights of a neural net whe using hot or cold method of learning


error = ((inputs * weight) - goal_pred) ** 2
print()

w = 0.5
ins = 0.5
truth_value = 0.8

for i in range(20):

    pred_2 = w * ins
    error2 = (pred_2 - truth_value) ** 2
    print(error2)
    # noinspection SpellCheckingInspection
    delt = pred_2 - truth_value
    w_delt = ins * delt
    w = w - w_delt
    print('Error: {} and prediction : {}'.format(error2, w))

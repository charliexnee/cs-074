# Charlie Nee
# COSC 74
# Assignment 3

import math
# Part A

def sigmoid(weights,features):
    sum = weights[1]
    for i in range(1,len(weights)):
        sum = sum + weights[i]*features[i-1]
    denom = math.pow(math.e, -sum)
    value = 1/denom
    return value

# Part B
def log_likelihood(weights,features,actual_ys):
    # is features a list of lists?
    product = 1
    for i in range(len(weights - 1)):
        sig = math.log(sigmoid(weights, features(i)))
        current_product = math.pow(sig, actual_ys(i)) * math.pow(1-sig, 1- actual_ys(i))
        product = product * current_product
    return product

# Part C
def learn_weights(actual_ys, features, num_iterations, learning_rate):
    #initialize weights
    weight = [0] * (len(features) + 1)
    for i in range(num_iterations):
        # gradient to keep track of the weights in the current row
        gradient = [0] * (len(features) + 1)
        # calculate gradientsfor all weights
        # do the intercept first
        sig = sigmoid(weight, features[i])
        gradient[0] = actual_ys[i] - sig
        # do the rest of the gradients now
        for j in range(len(features)):
            gradient[j + 1] = (actual_ys[i] - sig * features[i][j])
        # update weights (note, each weight can be updated likethis:
        for j in range(len(weight)):
            weight[j]+=(learning_rate * gradient[j])
        if i%1000==0:
            print(log_likelihood(weight,features,actual_ys))
    return weight
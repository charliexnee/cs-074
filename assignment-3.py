# Charlie Nee
# COSC 74
# Assignment 3

import math
# Part A

def sigmoid(weights,features):
    sum = weights[1]
    for i in range(1,len(weights))
        sum = sum + weights[i]*features[i-1]
    denom = math.pow(math.e, -sum)
    value = 1/denom
    return value

# Part B
def log_likelihood(weights,features,actual_ys):
    return log_likelihood
# Part

def learn_weights(actual_ys, features, num_iterations, learning_rate):
    #initialize weights
    weights = list();
    weights.append(1)
    for i in range(len(features)):
        weights.append(1)
    # for i in range(num_iterations):
        # calculate gradientsfor all weights
        # update weights (note, each weight can be updated likethis:weight[j]+=learning_rate * gradient[j])if i%1000==0:
        # print(log_likelihood(weights,features,actual_ys))
return weights
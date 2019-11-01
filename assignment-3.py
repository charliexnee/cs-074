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
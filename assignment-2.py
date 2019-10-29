# Charlie Nee (Amanda in Canvas)
# COSC 74
# Assignment 2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


# Part A, Gradient Descent
# Code was taken from my notes from lecture, so it may look extremely similar to the professor's solution
def gd(actual_xs, actual_ys, initial_m, initial_c, learning_rate, num_iterations):
    N = len(actual_xs)
    m = initial_m
    c = initial_c
    for i in range(num_iterations):
        m_gradient = 0
        c_gradient = 0

        for x, y in zip(actual_xs, actual_ys):
            y_c = m * x + c
            m_gradient += (y - y_c) * x
            c_gradient += y - y_c
        m_gradient = m_gradient * - (2 / N)
        c_gradient = c_gradient * - (2 / N)
        m = m - m_gradient * learning_rate
        c = c - c_gradient * learning_rate
    return m, c


# Part B, SSE
# Code was taken from my notes from lecture, so it may look extremely similar to the professor's solution
def calculate_SSE(actual_xs, actual_ys, m, c):
    error = 0
    for x, y in zip(actual_xs, actual_ys):
        error += ((m*x + c) - y) ** 2
    return error


# Part C, Histogram
def plot_residuals(actual_ys, predicted_ys):
    residuals = list()
    for actual_y, predicted_y in zip(actual_ys, predicted_ys):
        residuals.append(predicted_y - actual_y)
    plt.hist(residuals, bins=len(residuals))
    plt.show()


# Helper functions for the driver code

def get_data(file_name):
    x_s = list()
    y_s = list()
    f = open(file_name, "r")
    for x in f:
        data = x.split(",")
        if data is not None:
            x_s.append(float(data[0]))
            y_s.append(float(data[1].strip("\n")))
    f.close()
    return x_s, y_s


def predict(actual_xs, m, c):
    predicted_ys = list()
    for x in actual_xs:
        predicted_ys.append(m*x + c)
    return predicted_ys


# Driver code
# Load Data
x_actual_1, y_actual_1 = get_data("hw2_data1.csv")
x_actual_2, y_actual_2 = get_data("hw2_data2.csv")

# Run gradient descent
m1, c1 = gd(x_actual_1, y_actual_1, 0, 0, 0.0001, 100)
m2, c2 = gd(x_actual_2, y_actual_2, 0, 0, 0.0001, 100)

# run SSE
error1 = calculate_SSE(x_actual_1, y_actual_1, m1, c1)
error2 = calculate_SSE(x_actual_2, y_actual_2, m2, c2)

# histograms
plot_residuals(y_actual_1, predict(x_actual_1, m1, c1))
plot_residuals(y_actual_2, predict(x_actual_2, m2, c2))

# Observations: For the first one the data seems to fall into a somewhat normal distribution
# On the second one, there seems to be a few massive outliers that skews the entire graph


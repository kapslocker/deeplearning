import numpy as np
''' TODO: Check with more functions later
'''

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_inverse(y):
    return np.log(y/(1-y))

def sigmoid_grad(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0.0, x)

def relu_inverse(x):
    return x

def relu_grad(x):
    return (x > 0) * 1.0

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

import numpy as np
''' TODO: Check with more functions later
'''

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return float(x > 0.0)

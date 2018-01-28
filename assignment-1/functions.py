import numpy as np
''' TODO: Check with more functions later
'''

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

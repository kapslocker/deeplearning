import numpy as np
import random
from functions import *
class NeuralNetwork(object):
    def __init__(self, learningRate, model, minibatchsize, epochs, activation_function = 'sigmoid', activation_function_grad = 'sigmoid_grad'):
        ''' Setup a fully connected neural network represented by
            model: sizes of each layer (1D array)'''

        # Parameters to tweak
        self.learningRate = learningRate
        self.model = model
        self.num_layers = len(model)
        self.epochs = epochs
        self.rate = learningRate
        self.minibatchsize = minibatchsize
        self.activation_function = activation_function
        self.activation_function_grad = activation_function_grad
        # Setup weights
        prev_layer = model[:-1]
        next_layer = model[1:]

        # Fully connected
        self.weights = [np.zeros(1)] + [np.random.randn(nex, prev) for nex, prev in zip(next_layer, prev_layer)]

        # Represent bias for x neurons in each of the num_layers layers.
        # Start with random biases
        self.biases = [np.random.randn(x, 1) for x in model]

        # Inputs to activations
        self.z = [np.zeros((x,1)) for x in model]

        # No. of activations = No. of inputs = No. of biases
        self.activations = [np.zeros((x, 1)) for x in model]

    def predict(self, x):
        ''' Run a forward propagation to evaluate '''
        self.forwardProp(x)
        return np.argmax(self.activations[-1])

    def forwardProp(self, inp):
        self.activations[0] = inp
        for i in xrange(1, self.num_layers):
            self.z[i] = self.weights[i].dot(self.activations[i]) + self.biases[i]
            if(activation_function == 'sigmoid'):
                self.activations[i] = sigmoid(self.z[i])
            else:
                self.activations[i] = relu(self.z[i])

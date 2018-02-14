import numpy as np
import random
from functions import *
from operator import add
class NeuralNetwork(object):
    def __init__(self, learningRate, model, minibatchsize, epochs, l2_lambda = 0.0, dropout = 1.0,activation_function = 'sigmoid', activation_function_grad = 'sigmoid_grad', objective_function = 'mean_squared', drop = True):
        ''' Setup a fully connected neural network represented by
            model: sizes of each layer (1D array)'''

        # Parameters to tweak
        self.model = model
        self.num_layers = len(model)
        self.epochs = epochs
        self.rate = learningRate
        self.mini_batch_size = minibatchsize
        self.dropout = dropout
        self.drop = drop
        self.l2lambda = l2_lambda
        self.objective_function = objective_function
        if activation_function == 'sigmoid':
            self.activation_function = lambda x : sigmoid(x)
            self.activation_function_grad = lambda x : sigmoid_grad(x)
        else:
            self.activation_function = lambda x : relu(x)
            self.activation_function_grad = lambda x : relu_grad(x)
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
        self.forwardProp(x, True)
        return np.argmax(self.activations[-1])

    def forwardProp(self, x, isTest = False):
        self.activations[0] = x
        for i in xrange(1, self.num_layers):
            ''' dropout vector for training phase'''
            activations_copy = self.activations[i - 1]
            if(self.drop and not isTest):
                r = np.random.binomial(1, self.dropout, self.activations[i - 1].shape)
                activations_copy = r * self.activations[i - 1]
            self.z[i] = np.dot(self.weights[i], activations_copy) + self.biases[i]
            self.activations[i] = self.activation_function(self.z[i])

    def learn(self, training_data, test_data):
        # Test data is used only for printing progress per epoch, and not for training
        self.N = len(training_data)
        ''' Minibatch gradient descent '''
        for i in xrange(self.epochs):
    	       random.shuffle(training_data)
               ''' create batches by choosing randomly'''
               batches = [training_data[j : j + self.mini_batch_size] for j in xrange(0, self.N, self.mini_batch_size)]
               for batch in batches:
                   self.updatebatch(batch)
               print "Processed Epoch {0} ".format(i), "Test accuracy: ", self.test(test_data), "Train accuracy: ", self.test(training_data)

    def backprop(self, x, y):
        error_biases =  [np.zeros(bias.shape) for bias in self.biases]
        error_weights = [np.zeros(wt.shape) for wt in self.weights]
        ''' One forward pass followed by one backward pass '''
        self.forwardProp(x)
        if(self.objective_function == 'mean_squared'):
            outputLayerError = (self.activations[-1] - y) * self.activation_function_grad(self.z[-1])
        elif self.objective_function == 'cross_entropy':
            outputLayerError = (self.activations[-1] - y)
        error_biases[-1] = outputLayerError
        error_weights[-1] = outputLayerError.dot(self.activations[-2].transpose())
        for i in xrange(self.num_layers - 2, 0, -1):
            temp = np.dot(self.weights[i + 1].transpose(), error_biases[i + 1]) * self.activation_function_grad(self.z[i])
            error_biases[i] = temp
            error_weights[i] = np.dot(error_biases[i], np.transpose(self.activations[i - 1]))
        return error_biases, error_weights

    def updatebatch(self, batch):
        error_biases = [np.zeros(bias.shape) for bias in self.biases]
        error_weights = [np.zeros(weight.shape) for weight in self.weights]
        ''' Evaluate gradients for each image in batch '''
        for x,y in batch:
            db, dw = self.backprop(x,y)
            error_biases = map(add, error_biases, db)
            error_weights = map(add, error_weights, dw)
        ''' Update weights and bias '''
        self.biases = [bias - (self.rate * error_bias) / self.mini_batch_size for bias, error_bias in zip(self.biases, error_biases)]
        self.weights = [(1.0 - ((self.rate * self.l2lambda) / self.N)) * weight - (self.rate * error_weight) / self.mini_batch_size for weight, error_weight in zip(self.weights, error_weights)]

    def test(self, test_data):
        n = len(test_data)
        count = len(filter(lambda (x,y) : np.argmax(y) == self.predict(x), test_data))
        return float(count) / n

import numpy as np
import random
from functions import *
class NeuralNetwork(object):
    def __init__(self, learningRate, model, minibatchsize, epochs, activation_function = 'sigmoid', activation_function_grad = 'sigmoid_grad'):
        ''' Setup a fully connected neural network represented by
            model: sizes of each layer (1D array)'''

        # Parameters to tweak
        self.model = model
        self.num_layers = len(model)
        self.epochs = epochs
        self.rate = learningRate
        self.minibatchsize = minibatchsize
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

        #Batch norm parameters
        self.beta = [np.random.randn(x, 1) for x in model]
        self.gamma = [np.random.randn(x, 1) for x in model]

        # Inputs to activations
        self.z = [np.zeros((x,1)) for x in model]

        # No. of activations = No. of inputs = No. of biases
        self.activations = [np.zeros((x, 1)) for x in model]
    def predict(self, x):
        ''' Run a forward propagation to evaluate '''
        self.forwardProp(x)
        return np.argmax(self.activations[-1])

    def forwardProp(self, x):
        self.activations[0] = self.activation_function(x)
        for i in xrange(1, self.num_layers):
            self.z[i] = self.weights[i].dot(self.activations[i - 1]) + self.biases[i]
            self.activations[i] = self.activation_function(self.z[i])

    def bn_forwardProp(self, X):
        self.activations[0] = self.activation_function(X)
        mu = 1/N*np.sum(h,axis =0) # Size (H,)
        sigma2 = 1/N*np.sum((h-mu)**2,axis=0)# Size (H,)
        hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
        y = gamma*hath+beta


    def learn(self, training_data, test_data):
    	random.shuffle(training_data)
        mini_batches = [training_data[k:k + self.minibatchsize] for k in range(0, len(training_data), self.minibatchsize)]
        for i in xrange(self.epochs):
            for mini_batch in mini_batches:
                sum_b = [np.zeros(bias.shape) for bias in self.biases]
                sum_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self.forwardProp(x)
                    delta_biases, delta_weights = self.error(x, y)
                    sum_b = [b + db for b, db in zip(sum_b, delta_biases)]
                    sum_w = [w + dw for w, dw in zip(sum_w, delta_weights)]
                self.biases =  [b - (self.rate/self.minibatchsize) * d_b for b, d_b in zip(self.biases, sum_b)]
                self.weights = [w - (self.rate/self.minibatchsize) * d_w for w, d_w in zip(self.weights, sum_w)]
            print i, self.test(test_data)


    def test(self, test_data):
        n = len(test_data)
        count = len(filter(lambda (x,y) : np.argmax(y) == self.predict(x), test_data))
        return (float(count) * 100.0) / float(n)

    def error(self, x, y):
    	error_biases = [np.zeros(bias.shape) for bias in self.biases]
    	error_weights = [np.zeros(weight.shape) for weight in self.weights]
        error_biases[-1] = (self.activations[-1] - y) * self.activation_function_grad(self.z[-1])
    	error_weights[-1] = np.dot(error_biases[-1], np.transpose(self.activations[-2]))
    	for i in xrange(self.num_layers-2,0,-1):
            temp = np.dot(np.transpose(error_weights[i + 1]), error_biases[i + 1])
            error = np.multiply(temp, self.activation_function_grad(self.z[i]))
            error_biases[i] = error
            error_weights[i] = np.transpose(self.activations[i-1] * np.transpose(error))
    	return error_biases, error_weights

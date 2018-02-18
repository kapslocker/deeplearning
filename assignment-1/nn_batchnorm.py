import numpy as np
import sys
import random
from functions import *
from operator import add
class NeuralNetwork(object):
    def __init__(self, learningRate, model, minibatchsize, epochs, l1_lambda = 0.0,  l2_lambda = 0.0, dropout = 1.0,activation_function = 'sigmoid', activation_function_grad = 'sigmoid_grad', objective_function = 'mean_squared', drop = True):
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
        self.l1lambda = l1_lambda
        self.objective_function = objective_function
        self.activation_inverse = sigmoid_inverse
        self.pop_mean = [np.zeros(x) for x in model]
        self.pop_var = [np.zeros(x) for x in model]
        self.num_batches = 0
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
        self.weights = [np.zeros((1, 1))] + [np.random.randn(nex, prev) for nex, prev in zip(next_layer, prev_layer)]

        # Represent bias for x neurons in each of the num_layers layers.
        # Start with random biases
        self.biases = [np.random.randn(x, 1) for x in model]
        self.beta = [np.zeros(x).reshape((x,1)) for x in model]
        self.gamma = [np.ones(x).reshape((x,1)) for x in model]

        # Inputs to activations
        self.z = [np.zeros((x,self.mini_batch_size)) for x in model]
        self.z_inter = [np.zeros((x,self.mini_batch_size)) for x in model]
        # No. of activations = No. of inputs = No. of biases
        self.activations = [np.zeros((x, self.mini_batch_size)) for x in model]
    def predict(self, x, y):
        ''' Run a forward propagation to evaluate '''
        #self.forwardProp(x, True)
        return np.argmax(self.bnt_forward(x, y))

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
               self.pop_mean = [np.zeros(x) for x in self.model]
               self.pop_var = [np.zeros(x) for x in self.model]
               self.num_batches = len(batches) - 1
               for x in xrange(len(batches) - 1):
                   self.updatebatch(batches[x])
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

    def custom(self, A, B, op):
        if(op=='+'):
            return np.transpose(np.transpose(A)+np.transpose(B))
        elif(op=='*'):
            return np.transpose(np.transpose(A)*np.transpose(B))
        elif(op=='-'):
            return np.transpose(np.transpose(A)-np.transpose(B))

    def bn_forward(self, batch):
        #print("ff1")
        self.activations[0] = [x[0] for x in batch]
        for i in xrange(1, self.num_layers-1):
            self.z[i] = np.dot(self.weights[i], self.activations[i-1])
            self.z[i] = self.custom(self.z[i], self.biases[i], '+')
            mu = np.mean(self.z[i], axis=1)
            #print(self.pop_mean[i].shape)
            self.pop_mean[i] = self.pop_mean[i].reshape(mu.shape) + mu/self.num_batches
            #print(self.pop_mean[i].shape)
            #sys.exit()
            interm = self.custom(self.z[i],mu,'-')
            sigma2 = np.mean((interm)**2,axis=1)
            self.pop_var[i] = self.pop_var[i].reshape(sigma2.shape) + sigma2/self.num_batches
            temp = self.custom(interm,(sigma2+1e-6)**(-1./2.),'*')
            self.z_inter[i] = self.custom(self.custom(self.gamma[i],temp,'*'),self.beta[i],'+')
            self.activations[i] = self.activation_function(self.z_inter[i])
        self.z[-1] = np.dot(self.weights[-1], self.activations[-2].reshape((self.z[-2].shape[0], self.z[-2].shape[1]))) + self.biases[-1]
        self.activations[-1] = self.activation_function(self.z[-1])
        #print(np.array([x[1] for x in batch])[0])
        #print(self.activations[-1][:,0])

    def bnt_forward(self, x, y):
        activations = [np.zeros((k,1)) for k in self.model]
        z = [np.zeros((k,1)) for k in self.model]
        activations[0] = x
        #print("tt1")
        for i in xrange(1, self.num_layers-1):
            #print("weight",self.weights[i].shape)
            #print("act",activations[i-1].shape)
            #print("dot",np.dot(self.weights[i], activations[i-1]).shape)
            #print("b",np.transpose(self.biases[i]))
            z[i] = np.dot(self.weights[i], activations[i-1]) + self.biases[i]
            #print(z[i].shape)
            z[i]= (z[i] - self.pop_mean[i])*(self.pop_var[i]+1e-6)**(-1./2.)
            #print(z[i].shape, self.pop_mean[i].shape, self.pop_var[i].shape)
            #sys.exit()
            activations[i] = self.gamma[i]*z[i] + self.beta[i]
            activations[i] = self.activation_function(activations[i])
        z[-1] = np.dot(self.weights[-1], activations[-2]) + self.biases[-1]
        activations[-1] = self.activation_function(z[-1])
        #print(np.array(self.pop_mean),np.array(self.pop_var))
        #print(activations[-1], y)
        return activations[-1]

    def bn_backprop(self, batch):
        y = [x[1] for x in batch]
        error_biases =  [np.zeros((x,self.mini_batch_size)) for x in self.model]
        error_weights = [np.zeros((wt.shape[0], wt.shape[1], self.mini_batch_size)) for wt in self.weights]
        error_beta = [np.zeros(bet.shape) for bet in self.beta]
        error_gamma = [np.zeros(gam.shape) for gam in self.gamma]
        self.bn_forward(batch)
        y = np.array(y)
        y = np.transpose(y)
        y = y.reshape(self.activations[-1].shape)
        if(self.objective_function == 'mean_squared'):
            outputLayerError = (self.activations[-1] - y) * self.activation_function_grad(self.z[-1])
        elif self.objective_function == 'cross_entropy':
            outputLayerError = (self.activations[-1] - y)
        #print(self.activations[-1].shape, np.array(y).shape)
        #print(outputLayerError[:,0])
        error_biases[-1] = outputLayerError
        #print(np.transpose(self.activations[-2][:,1]).shape)
        error_weights[-1] = np.array([np.dot(outputLayerError[:,x].reshape((9,1)),np.transpose(self.activations[-2][:,x])) for x in xrange(self.mini_batch_size)])
        #print(outputLayerError)
        #print(error_weights[-1])
        for i in xrange(self.num_layers - 2, 0, -1):
            dy = np.dot(self.weights[i + 1].transpose(), error_biases[i + 1])*(self.activation_function_grad(self.z_inter[i])).reshape(error_biases[i].shape)
            h = self.z[i]
            mu = np.mean(h, axis = 1)
            var = np.mean(self.custom(h,mu,'-')**2, axis = 1)
            error_beta[i] = np.sum(dy, axis=1).reshape(self.beta[i].shape)
            error_gamma[i] = np.sum(self.custom(h,mu,'-').reshape(error_biases[i].shape) * (var + 1e-4)**(-1. / 2.) * dy, axis=1).reshape(self.gamma[i].shape)
            #print((var + 1e-4)**(-1. / 2.))

            temp = self.custom(self.custom(h,mu,'-'),(var + 1e-4)**(-1.0),'*').reshape(dy.shape) * np.sum(dy * (self.custom(h,mu,'-')).reshape(dy.shape), axis=1).reshape(mu.shape)
            #print(np.array(self.mini_batch_size * dy - np.sum(dy, axis=1).reshape(mu.shape)
            #    - temp))
            #print(dy)
            dh = (1. / self.mini_batch_size) * self.gamma[i] * (var + 1e-4)**(-1. / 2.) * (self.mini_batch_size * dy - np.sum(dy, axis=1).reshape(mu.shape)
                - temp)
            error_biases[i] = dh
            #print(error_biases[i][:,1].reshape(mu.shape), np.transpose(self.activations[i-1][1]))
            error_weights[i] = np.array([np.dot(error_biases[i][:,x].reshape(mu.shape),np.transpose(self.activations[i-1][x])) for x in xrange(self.mini_batch_size)])
            #print(error_weights[i][1][0],"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print(np.array(error_biases))
        #print("beta",error_beta[1])
        #print("gamma",error_gamma[1])
        #print("bias",error_biases[1][:,0])
        #print("weight",error_weights[1][0])
        error_biases = [np.sum(error_biases[x], axis=1).reshape(self.beta[x].shape) for x in xrange(len(error_biases))]
        error_weights =  [np.sum(error_weights[x], axis=0) for x in xrange(len(error_weights))]
        return error_biases, error_weights, error_beta, error_gamma

    def updatebatch(self, batch):
        error_biases = [np.zeros(bias.shape) for bias in self.biases]
        error_beta = [np.zeros(bet.shape) for bet in self.beta]
        error_gamma = [np.zeros(gam.shape) for gam in self.gamma]
        error_weights = [np.zeros(weight.shape) for weight in self.weights]
        ''' Evaluate gradients for each image in batch '''
        #for x,y in batch:
        #    db, dw = self.backprop(x,y)
        #    error_biases = map(add, error_biases, db)
        #    error_weights = map(add, error_weights, dw)
        error_biases, error_weights, error_beta, error_gamma = self.bn_backprop(batch)
        #print(self.weights[1])
        ''' Update weights and bias '''
        self.biases = [bias - (self.rate * error_bias)/ self.mini_batch_size for bias, error_bias in zip(self.biases, error_biases)]
        if self.l2lambda > 0:
            self.weights = [(1.0 - ((self.rate * self.l2lambda) / self.N)) * weight - (self.rate * error_weight) / self.mini_batch_size for weight, error_weight in zip(self.weights, error_weights)]
        elif self.l1lambda > 0:
            self.weights = [weight - np.sign(weight) * (self.rate * self.l1lambda) / self.N - (self.rate * error_weight) / self.mini_batch_size for weight, error_weight in zip(self.weights, error_weights)]
        else:
            self.weights = [weight - (self.rate * error_weight)/ self.mini_batch_size for weight, error_weight in zip(self.weights, error_weights)]
        self.beta = [bet - (self.rate * error_bet)/ self.mini_batch_size for bet, error_bet in zip(self.beta, error_beta)]
        self.gamma = [gam - (self.rate * error_gam)/ self.mini_batch_size for gam, error_gam in zip(self.gamma, error_gamma)]

    def test(self, test_data):
        n = len(test_data)
        count = len(filter(lambda (x,y) : np.argmax(y) == self.predict(x,y), test_data))
        return float(count) / n

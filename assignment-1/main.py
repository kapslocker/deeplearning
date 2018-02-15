from extract_data import *
from neuralnet import *

# labelstring = "LNBEDKTAY"
labelstring = "LNB"
learning_rate = 3
mini_batch_size = 36
model = [784,30,9]
model[-1] = len(labelstring)
epochs = 100
dropout_rate = 0.9
isDropout = True
l2lambda = 0.0
l1lambda = 0.0
isSoftmaxEnabled = False
objective_function = 'mean_squared'
activationFunction = 'relu'
print "learning rate = ", learning_rate
print "minibatchsize = ", mini_batch_size
print "No. of epochs = ", epochs
print "Dropout probability = ", dropout_rate
print "Is dropout = ", isDropout
print "L2 lambda = ", l2lambda
print "L1 lambda = ", l1lambda
print "Softmax Layer = ", isSoftmaxEnabled
print "objective_function = ", objective_function
print "activation_function = ", activationFunction
NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs, l1_lambda = l1lambda, l2_lambda = l2lambda, dropout = dropout_rate, activation_function = activationFunction, objective_function = objective_function, drop = isDropout, isSoftmax = isSoftmaxEnabled)
# NN = Network(model)
trainData = train_data(labelstring)
testData = test_data(labelstring)

print "Training data size: {0}, No. of Parameters: {1}".format(len(trainData), np.prod(model))

print "Training started..."
# NN.SGD(trainData, epochs, mini_batch_size, learning_rate, testData)
NN.learn(trainData, testData)
print "Testing started..."
print "Testing data size: {0}".format(len(testData))
NN.test(testData)

'''
objective_function takes two values: 'mean_squared' and 'cross_entropy'
activation_function takes two values: 'relu' and 'sigmoid'
Testing results:
For 2 classes: 15373 images, [784 * 30 * 2] parameters
learning_rate = 0.1
dropout = 0.5
Test Accuracy after 25 epochs = 98.2 %

For 9 classes: 50125 images:

[784 * 100 * 9] parameters: +L2regularization +dropout fastlearn
    objective_function = cross_entropy
    activation_function = sigmoid
    learning_rate = 3
    dropout_rate = 0.5
    l2lambda = 10.0
    Test Accuracy = '91.7%' after 70 epochs

[784 * 100 * 9] parameters: -L2regularization +dropout slowlearn
    objective_function = cross_entropy
    activation_function = sigmoid
    learning_rate = 0.5
    dropout_rate = 0.5
    l2lambda = 0.0
    Test Accuracy = '95.5 %' after 90 epochs

[784 * 100 * 9] parameters: -L2regularization -dropout fastlearn
    objective_function = cross_entropy
    activation_function = sigmoid
    learning_rate = 3
    dropout_rate = 0.9
    l2lambda = 0
    Test Accuracy = 98.03%

'''

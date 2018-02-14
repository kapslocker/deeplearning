from extract_data import *
from neuralnet import *

labelstring = "LNBEDKTAY"
# labelstring = "XU"
learning_rate = 3
mini_batch_size = 36
model = [784,30,9]
model[-1] = len(labelstring)
epochs = 100
dropout_rate = 0.9
isDropout = True

NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs, dropout = dropout_rate, objective_function = 'cross_entropy', drop = isDropout)
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
For 2 classes: 15373 images, [784 * 30 * 2] parameters
learning_rate = 0.1
dropout = 0.5
Test Accuracy after 25 epochs = 98.2 %

For 9 classes: 50125 images, [784 * 30 * 9] parameters
    learning_rate =
    dropout =
    Test Accuracy =
'''

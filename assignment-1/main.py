from extract_data import *
from neuralnet import *

labelstring = "XMOFG"
learning_rate = 0.1
mini_batch_size = 10
model = [784,100,3]
model[-1] = len(labelstring)
epochs = 5

NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs)
trainData = train_data(labelstring)
testData = test_data(labelstring)

print "Training started..."
NN.learn(trainData, testData)
print "Testing started..."
print NN.test(testData)

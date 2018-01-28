from extract_data import *
from neuralnet import *

labelstring = "abcdefghi"
learning_rate = 0.5
mini_batch_size = 32
model = [784,400,9]
epochs = 2

NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs)
trainData = train_data(labelstring)
testData = test_data(labelstring)

print "Training started..."
NN.learn(trainData, testData)
print "Testing started..."
print NN.test(testData)

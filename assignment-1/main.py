from extract_data import *
from neuralnet import *
labelstring = "XM"
learning_rate = 0.1
mini_batch_size = 36
model = [784,30,2]
epochs = 10

NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs)
# NN = Network(model)
trainData = train_data(labelstring)
testData = test_data(labelstring)


print "Training started..."
# NN.SGD(trainData, epochs, mini_batch_size, learning_rate, testData)
NN.learn(trainData, testData)
print "Testing started..."
NN.test(testData)

from extract_data import *
from neuralnet import *
from nielsennet import *
labelstring = "rocdtkghm"
learning_rate = 3
mini_batch_size = 36
model = [784,30,10]
epochs = 10

NN = NeuralNetwork(learning_rate, model, mini_batch_size, epochs)
# NN = Network(model)
trainData = train_data(labelstring)
testData = test_data(labelstring)


print "Training started..."
# NN.SGD(trainData, epochs, mini_batch_size, learning_rate, testData)
NN.learn(trainData, testData)
print "Testing started..."
NN.evaluate(testData)

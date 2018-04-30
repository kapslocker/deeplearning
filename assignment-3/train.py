from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import get_data

TRAIN_DIR = "TIMIT/TRAIN/"
TEST_DIR = "TIMIT/TEST/"

''' Audio splits are as:
*_DIR/DR*/folder/sound.wav '''



''' Performing the binary classification method'''
input_size = 1
hidden_size = 250
num_layers = 2
num_classes = 2
batch_size = 1  #SGD
num_epochs = 50
learning_rate = 0.001

class BiDirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.rand(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection
        c0 = Variable(torch.rand(self.num_layers*2, x.size(0), self.hidden_size))
        out, _ = self.lstm(x, (h0, c0)) # forward prop
        out = self.fc(out[:, -1, :])    # get hidden state
        return out

rnn = BiDirectionalLSTM(input_size, hidden_size, num_layers, num_classes)


#train

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)
min_loss = 1000
train_data = get_data.getlabels(TRAIN_DIR)
for epoch in xrange(num_epochs):
    #SGD
    for idx, (y,labels_y) in enumerate(train_data):
        seq_len = y.shape[0]
        y = torch.from_numpy(y)
        y = Variable(y.view(1, seq_len, input_size))
        labels_y = Variable(torch.from_numpy(labels_y))

        optimizer.zero_grad()
        #forward prop
        outputs = rnn(y)
        #eval loss
        loss = criterion(outputs, labels_y)
        #back prop
        loss.backward()

        #print loss
        if idx % 100 == 0:
            print('epoch %d/%d, sample %d/%d, loss: %.6f'%(epoch + 1, num_epochs, idx, len(train_data)/batch_size, loss.data[0]))
            if loss.data[0] < min_loss:
                min_loss = loss.data[0]
                torch.save(rnn, 'model_%d.pth'%(min_loss))
#TODO:
# test

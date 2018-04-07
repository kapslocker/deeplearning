''' Evaluate t-SNE representation of the dataset to visualize clusters based on what the network learns '''
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import argparse
import os
import cv2
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import itertools

def to_tensor(frame):
    ''' Convert detected face to imagenet input '''
    scaler = transforms.Resize(256)
    cropper = transforms.CenterCrop(224)
    normalizer = transforms.Normalize(mean = [0.485, 0.456, 0.456], std = [0.229, 0.224, 0.225])
    tensorizer = transforms.ToTensor()

    transformed_frame = Variable(normalizer(tensorizer(cropper(scaler(frame))))).unsqueeze(0)
    if torch.cuda.is_available():
        return transformed_frame.cuda()
    else:
        return transformed_frame

class prevModel():
    ''' Get activations of all but the last layer '''
    def __init__(self, model):
        my_modules = list(model.children())[:-1]
        self.model_prev = nn.Sequential(*my_modules)
    def forward(self, img):
        a = self.model_prev(img)
        return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, help = "Dataset location")
    parser.add_argument('tsne_file', type = str, help = 'tSNE file location')
    parser.add_argument('model', type = str, help = "Model file location")
    args = parser.parse_args()
    if(torch.cuda.is_available()):
        model_trained = torch.load(args.model)
    else:
        model_trained = torch.load(args.model, map_location = lambda storage, loc: storage)
    model_trained.train(False)
    ''' Get all but the last linear layer of the trained network. '''
    myModel = prevModel(model_trained)
    X = list()
    print args.data_dir
    for category in os.listdir(args.data_dir):
        print "category: " + category
        for item in os.listdir(args.data_dir + "/" + category):
            frame = cv2.imread(args.data_dir + "/" + category + "/" + item)
            transformed = to_tensor(Image.fromarray(frame))
            output = myModel.forward(transformed)
            output = output.data.numpy()
            ''' Get the 512 x 512 vector '''
            output_val = [output[0][i][0][0] for i in xrange(512)]
            X.append(output_val)
    print "Vectors computed. Beginning t-SNE"
    ''' Using default tSNE parameters '''
    X_embedded = TSNE().fit_transform(X)
    with open(args.tsne_file, 'w') as opfile:
        for x in X_embedded:
            opfile.write(str(x[0]) + " " + str(x[1]) + "\n")
    print "tSNE evaluated. Drawing now..."
    '''' Draw the colored tsne map '''
    counts = list()
    for category in os.listdir(args.data_dir):
        counts.append(len(os.listdir(args.data_dir + "/" + category)))
    a = list()
    ''' Create labels '''
    for i in xrange(len(counts)):
        l = list(itertools.repeat(i, counts[i]))
        a = a + l
    X = list()
    Y = list()
    with open(args.tsne_file, 'r') as inpfile:
        for line in inpfile:
            temp = line.split(' ')
            X.append(float(temp[0]))
            Y.append(float(temp[1]))
    X = np.array(X)
    Y = np.array(Y)
    a = np.array(a)
    plt.scatter(X, Y, c = a)
    plt.show()

import subprocess

import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import wave
import librosa
from sklearn import mixture

from sklearn.neural_network import MLPClassifier



mypath="/dataset1/"
train_labels=[]
N=224

path = os.getcwd()
j=0
train_data=np.array([]).reshape((0,N, N, 3))
i=0
k=0
dataset=[]
labels=[]
test_dataset=[]
test_labels=[]

mypath='/dataset1'
speaker_labels={path+mypath+'/pant':0, path+mypath+'/atul':1,path+mypath+'/sandeep':3,path+mypath+'/shailendra':4,path+mypath+'/flute':2,path+mypath+ '/sadguru':5}
for root, dirs, files in os.walk(path+mypath, topdown=True):
    for fil in files:
      # print(files)
      if files!="":
        print(fil)
        y, sr = librosa.load(root+"/"+fil,sr=48000)
        nsamples=np.array(y).shape[0]
        n=nsamples//(48000*5)
        print(n)
        for i1 in range(10,20):
            y1=[]
            if i1!=n-1:
                y1=y[i1*5*48000:(i1+1)*5*48000]
            else:
                y1=y[nsamples-5*48000:nsamples]
            labels.append(speaker_labels[root])
            arr=librosa.feature.mfcc(y=y1, sr=sr)
            dataset.append(arr)
	for i1 in range(20,30):
	    if i1!=n-1:
                y1=y[i1*5*48000:(i1+1)*5*48000]
            else:
                y1=y[nsamples-5*48000:nsamples]
            test_labels.append(speaker_labels[root])
            arr=librosa.feature.mfcc(y=y1, sr=sr)
            test_dataset.append(arr)



        i+=1
        
    j+=1


dataset=np.array(dataset)
dataset=np.reshape(dataset,(dataset.shape[0],-1))
test_dataset=np.array(test_dataset)
test_dataset=np.reshape(test_dataset,(test_dataset.shape[0],-1))
test_labels=np.array(test_labels)
print(test_dataset.shape)
print(dataset.shape)
labels=np.array(labels)
print(labels.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(50,50,), random_state=1)
clf.fit(dataset,labels)
y_predict=clf.predict(dataset)
mat=np.zeros((5,5))
for i in range(dataset.shape[0]):
    mat[labels[i],y_predict[i]]+=1

print(mat)
y_test=clf.predict(test_dataset)
mattest=np.zeros((5,5))
for i in range(test_dataset.shape[0]):
    mattest[test_labels[i],y_test[i]]+=1
print(mattest)

print(dataset.shape)
print(labels.shape)

""" short time fourier transform of audio signal """

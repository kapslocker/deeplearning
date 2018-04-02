import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

import cv2
import argparse

def get_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rois = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = [frame[y:y + h, x: x + w] for (x,y,w,h) in rois]
    return faces

def to_tensor(frame):
    ''' Convert detected face to imagenet input '''
    scaler = transforms.Scale((224,224))
    normalizer = transforms.Normalize(mean = [0.485, 0.456, 0.456], std = [0.229, 0.224, 0.225])
    tensorizer = transforms.ToTensor()

    transformed_frame = Variable(normalizer(tensorizer(scaler(img))))
    return transformed_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_file", type = str, help = "image location")
    args = parser.parse_args()
    frame = cv2.imread(args.img_file)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = models.resnet18(pretrained = False)
    model.load_state_dict(torch.load('models/saved_model.pth'))
    model.train(False)

    face = get_faces(frame)
    for face in faces:
        trans_face = to_tensor(face)
        output = model(trans_face)
        print output

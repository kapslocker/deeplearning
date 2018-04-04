import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
import argparse



def get_faces(frame):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../classifier/' + 'cascade.xml')
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_roi = face_cascade.detectMultiScale(grayframe, 1.3, 5)
    faces = [frame[y:y + h, x: x + w] for (x,y,w,h) in face_roi]
    return faces, face_roi

def to_tensor(frame):
    ''' Convert detected face to imagenet input '''
    scaler = transforms.Scale(256)
    cropper = transforms.CenterCrop(224)
    normalizer = transforms.Normalize(mean = [0.485, 0.456, 0.456], std = [0.229, 0.224, 0.225])
    tensorizer = transforms.ToTensor()

    transformed_frame = Variable(normalizer(tensorizer(cropper(scaler(frame))))).unsqueeze(0)
    if torch.cuda.is_available():
        return transformed_frame.cuda()
    else:
        return transformed_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_file", type = str, help = "Test image\\video location")
    parser.add_argument("saved_model", type = str, help = "Saved model file location")
    args = parser.parse_args()
    num_classes = 7
    labels = ['atul', 'noise', 'raman', 'sadguru', 'sandeep', 'shailendra', 'sorabh']
    if(torch.cuda.is_available()):
        model = torch.load(args.saved_model)
    else:
        model = torch.load(args.saved_model, map_location = lambda storage, loc: storage)
    model.train(False)

    cap = cv2.VideoCapture(args.img_file)
    while(True):
        code, frame = cap.read()
        # Uncomment for direct checking without face detection.
        # output = model(to_tensor(Image.fromarray(frame)))
        # temp = frame
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # idx = np.argmax(output.data.numpy())
        # cv2.putText(temp, labels[idx], (100 - 50, 100 - 50), font, 4, (255,255,255), 2, cv2.LINE_AA)
        # cv2.imshow('result', temp)
        # cv2.waitKey(1)

        # Uncomment for checking after faces are detected.
        faces, locs = get_faces(frame)
        if(len(faces) == 0):
            cv2.imshow('result', frame)
            cv2.waitKey(1)
        for (face, loc) in zip(faces, locs):
            trans_face = to_tensor(Image.fromarray(face))
            output = model(trans_face)
            (x,y,w,h) = loc
            temp = frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(temp, (x,y), (x + w, y + h), (255,0,255), 2)
            idx = np.argmax(output.data.numpy())
            cv2.putText(temp, labels[idx], (x , y), font, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('result', temp)
            cv2.waitKey(1)

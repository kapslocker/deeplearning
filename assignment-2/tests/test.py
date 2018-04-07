import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
import argparse
import os

def get_faces(frame):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_cascade = cv2.CascadeClassifier('../classifier/' + 'cascade.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_roi = face_cascade.detectMultiScale(grayframe, 1.3, 5)
    faces = [frame[y:y + h, x: x + w] for (x,y,w,h) in face_roi]
    return faces, face_roi

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

def frame_label(frame, model, label_arr):
    ''' Get label of frame predicted by model '''
    transformed = to_tensor(Image.fromarray(frame))
    output = model(transformed)
    # top_prob, top_label = torch.topk(nn.functional.softmax(output), 1)
    idx = np.argmax(output.data.numpy())
    return label_arr[idx]

def img_mode(data_dir, model, label_arr, act_labels):
    ''' Expects cropped faces as dataset '''
    #TODO: Add option to test full frame instead of cropped faces.
    i = 0
    count = 0
    for item in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, item))
        mylabel = frame_label(img, model, label_arr)
        if(mylabel == act_labels[i]):
            count += 1
        print mylabel, count, i
        if(item.split('_')[-1] == '0.jpg'):
            i += 1
    return float(count) / float(len(act_labels))

def video_mode(vid_file, model, label_arr, act_labels = None):
    ''' Create bounding boxes and display video with labels. Return accuracy in the end'''
    cap = cv2.VideoCapture(vid_file)
    i = 0
    count = 0
    while(cap.isOpened()):
        code, frame = cap.read()
        faces, locs = get_faces(frame)
        mylabel = 'j,k'
        if(len(faces) > 0):
            for (face, loc) in zip(faces, locs):
                trans_face = to_tensor(Image.fromarray(face))
                output = model(trans_face)
                (im_x, im_y,w,h) = loc
                font = cv2.FONT_HERSHEY_SIMPLEX
                ''' Get label '''
                mylabel = frame_label(frame, model, label_arr)
                ''' Draw bounding box and write label '''
                cv2.rectangle(frame, (im_x,im_y), (im_x + w, im_y + h), (255,0,255), 2)
                cv2.putText(frame, mylabel, (im_x , im_y), font, 1, (255,255,255), 2, cv2.LINE_AA)
        if(act_labels is not None):
            if(mylabel == act_labels[i]):
                count += 1
        i += 1
        cv2.imshow('result', frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return float(count) / float(len(act_labels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type = str, help = "Test images/video location")
    parser.add_argument("saved_model", type = str, help = "Saved model file location")
    parser.add_argument('label_file', type = str, help = "Labelled file")
    parser.add_argument('is_vid_mode', type = str, help = "Is Video Mode? \'y\'/\'n\'")
    args = parser.parse_args()
    labelarr = ['A,K', 'j,k', 'F,R', 'S,G', 'S,M', 'S,K', 'S,P']
    num_classes = len(labelarr)

    if(torch.cuda.is_available()):
        model = torch.load(args.saved_model)
    else:
        model = torch.load(args.saved_model, map_location = lambda storage, loc: storage)
    model.train(False)
    with open(args.label_file) as labelfile:
        labels = [line.rstrip() for line in labelfile]
    print args.is_vid_mode
    if(args.is_vid_mode == 'y'):
        accuracy = video_mode(args.data_dir, model, labelarr, labels)
    else:
        accuracy = img_mode(args.data_dir, model, labelarr, labels)
    print accuracy

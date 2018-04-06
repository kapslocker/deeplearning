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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_file", type = str, help = "Test image\\video location")
    parser.add_argument("saved_model", type = str, help = "Saved model file location")
    parser.add_argument('label_file', type = str, help = "Labelled file")
    args = parser.parse_args()
    num_classes = 7
    labelarr = ['A,K', 'j,k', 'F,R', 'S,G', 'S,M', 'S,K', 'S,P']
    if(torch.cuda.is_available()):
        model = torch.load(args.saved_model)
    else:
        model = torch.load(args.saved_model, map_location = lambda storage, loc: storage)
    model.train(False)
    with open(args.label_file) as labels:
        filedata = [line.rstrip() for line in labels]
    print filedata[0]
    cap = cv2.VideoCapture(args.img_file)
    x = 0
    count = 0
    total = len(filedata)
    while(1):
        code, frame = cap.read()
        faces, locs = get_faces(frame)
        if(len(faces) == 0):
            mylabel = labelarr[1]
        elif(len(faces) > 1):
            counts = [0 for i in xrange(7)]
            for (face, loc) in zip(faces, locs):
                trans_face = to_tensor(Image.fromarray(face))
                output = model(trans_face)
                top_prob, top_label = torch.topk(nn.functional.softmax(output), 1)
                idx = np.argmax(top_label.data.numpy())
                counts[idx] += 1
            mylabel = labelarr[np.argmax(np.array(counts))]
        else:
            for (face, loc) in zip(faces, locs):
                trans_face = to_tensor(Image.fromarray(face))
                output = model(trans_face)
                # ''' Go through a softmax '''
                # soft = nn.functional.softmax(output)
                # print soft
                # (im_x, im_y,w,h) = loc
                # temp = frame
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.rectangle(temp, (im_x,im_y), (im_x + w, im_y + h), (255,0,255), 2)
                # dat = soft.data.numpy()
                # top_prob, top_label = torch.topk(output, 1)
                idx = np.argmax(output.data.numpy())
                mylabel = labelarr[idx]
                # print labelarr[idx], filedata[x], top_label.data.numpy()
                # x = x + 1
                # cv2.putText(temp, mylabel, (im_x , im_y), font, 1, (255,255,255), 2, cv2.LINE_AA)
                # cv2.imshow('result', temp)
                # cv2.waitKey(1)
        if(mylabel == filedata[x]):
            count += 1
        print mylabel, filedata[x]
        x = x + 1
        if x % 100 == 0:
            print count, x, float(count)/ x
    print count, float(count)/ total

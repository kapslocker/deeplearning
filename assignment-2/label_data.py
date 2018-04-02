import os
import cv2
import numpy as np
import argparse

''' Generate dataset:
    1. Filter video to get frames that have 1 face. The detection is done at a rate of 1 frame per second.
    2. Faces are identified using the Viola Jones face detector in OpenCV.
    3. Save frames and generate txt with labels.'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_file", type = str, help = "video location")
    parser.add_argument("tag", type = str, help = "the category to which this video goes")
    args = parser.parse_args()
    print args.vid_file
    ROOT = "data_sadguru/"
    CROPPED_ROOT = "data_cropped/"
    folder = ROOT + args.tag + "/"
    folder2 = CROPPED_ROOT + args.tag + "/"
    noise_folder = ROOT + "noise/"
    labelfile = ROOT + "labels.txt"
    labelfile2 = CROPPED_ROOT + "label2.txt"
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder2)
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(args.vid_file)
    vidname = args.vid_file.replace(' ','').replace('-','').replace('.','').replace('/','').replace('!','').replace('&','')
    print "Video length = " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    with open(labelfile2, 'a') as labels_cropped:
        with open(labelfile, 'a') as labels:
            (x,y,w,h) = (0,0,0,0)
            while(cap.isOpened()):
                ret, frame = cap.read()
                frame_num += 1
                if(frame_num % 20 == 0):
                    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(grayframe)
                    cropped_label = "-"
                    ''' Get the center of the frame '''
                    r,c = grayframe.shape

                    # grayframe = grayframe[:, 700 : c]
                    # print grayframe.shape
                    if(len(faces) > 1):
                        N,M = grayframe.shape
                        (x1,y1,w1,h1) = faces[0]
                        (x2, y2, w2, h2) = faces[1]
                        labels_cropped.write(cropped_label)
                        path2 = folder2 + vidname + str(frame_num) + '.jpg'
                        x = (x1 + x2) / 2
                        width = max(x1, x2) - min(x1,x2) + 50 + w1 + w2
                        height = int(1.618 * width)
                        subset = frame[max(y1 - height / 2, 0) : min(y1 + height / 2, N), max(x - width/2, 0) : min(x + width / 2, M)]
                        r,c, d = subset.shape
                        # cv2.imshow('asd', subset)
                        # # print float(r) / float(c)
                        if(r > 500 or abs(c - 500) < 200):
                            # cv2.imshow('asd', subset)
                            cv2.imwrite(path2, subset)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    cap.release()

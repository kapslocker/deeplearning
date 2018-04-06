import os
import cv2
import numpy as np
import argparse

''' Generate dataset:
    1. Filter video to get frames that have 1 speaker.
    2. Save frames and generate txt with labels.'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_file", type = str, help = "video location")
    parser.add_argument("tag", type = str, help = "the category to which this video goes")
    args = parser.parse_args()
    print args.vid_file
    ROOT = "data/"
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
    face_cascade = cv2.CascadeClassifier('classifier_newer/cascade.xml')
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)
                    cropped_label = "-"
                    if(len(faces) == 1):
                        (x,y,w,h) = faces[0]
                        path = folder + vidname + str(frame_num) + ".jpg"
                        path2 = folder2 + vidname + str(frame_num) + ".jpg"
                        label = path + "\t" + args.tag + "\n"
                        cropped_label = path2 + "\t" + args.tag + "\n"
                    else:
                        path = noise_folder + vidname + "_" + str(frame_num) + ".jpg"
                        label = path + "\t" + "noise" + "\n"
                    if cropped_label is not "-":
                        labels_cropped.write(cropped_label)
                        cv2.imwrite(path2, frame[y: y + h, x : x + w])
                    # labels.write(label)
                    # cv2.imwrite(path, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    cap.release()

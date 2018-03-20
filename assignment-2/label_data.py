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

    ROOT = "data/"
    folder = ROOT + args.tag + "/"
    noise_folder = ROOT + "noise/"
    labelfile = ROOT + "labels.txt"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(args.vid_file)
    print "Video length = " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    with open(labelfile, 'a') as labels:
        while(cap.isOpened()):
            ret, frame = cap.read()
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)
            frame_num += 1
            if(len(faces) == 1):
                path = folder + str(frame_num) + ".jpg"
                label = path + "\t" + args.tag + "\n"
            else:
                path = noise_folder + args.tag + "_" + str(frame_num) + ".jpg"
                label = path + "\t" + "noise" + "\n"
            labels.write(label)
            cv2.imwrite(path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

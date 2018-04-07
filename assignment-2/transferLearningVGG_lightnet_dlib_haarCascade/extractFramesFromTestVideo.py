import os
import glob
import cv2
import numpy as np
import os
import math
videoDirectory="./"
framesDirectory="testFrames2/"

try:
    if not os.path.exists(framesDirectory):
        os.makedirs(framesDirectory)
except OSError:
    print ('Error: Creating directory of data')




try:
    if not os.path.exists(framesDirectory):
        os.makedirs(framesDirectory)
except OSError:
    print ('Error: Creating directory of data')
        
allfiles = os.listdir(videoDirectory)
mp4Files = ['test_vid_2.avi']
print("length"+str(len(mp4Files)))
for currmp4File in mp4Files:
    # print(videoDirectory+currPerson+currmp4File)
    cap = cv2.VideoCapture(videoDirectory+"/"+currmp4File)
    currentFrame = 0
    # Capture frame-by-frame
    ret, frame = cap.read()
    while(ret):
        # Saves image of the current frame in jpg file
        name = framesDirectory+"/"+str(currentFrame) + ".jpg"
        # print ('Creating...' + name)        
        cv2.imwrite(name, frame)
        ret, frame = cap.read()
        # To stop duplicate images
        currentFrame += 1
    cap.release()
    print "Done!"
    

import os
import glob
import cv2
import numpy as np
import os
import math
videoDirectory="dataset/"
framesDirectory="frameData/"

try:
    if not os.path.exists(framesDirectory):
        os.makedirs(framesDirectory)
except OSError:
    print ('Error: Creating directory of data')


dirList=os.listdir(videoDirectory)
dirList=["atul"]

for currPerson in dirList:
    try:
        if not os.path.exists(framesDirectory+currPerson):
            os.makedirs(framesDirectory+currPerson)
    except OSError:
        print ('Error: Creating directory of data')
        
    allfiles = os.listdir(videoDirectory+currPerson)
    mp4Files = [ fname for fname in allfiles if fname.endswith('.mp4')]
    print("length"+str(len(mp4Files)))
    for currmp4File in mp4Files:
        # print(videoDirectory+currPerson+currmp4File)
        cap = cv2.VideoCapture(videoDirectory+currPerson+"/"+currmp4File)
        currentFrame = 0
        # Capture frame-by-frame
        ret, frame = cap.read()
        while(ret):
            # Saves image of the current frame in jpg file
            name = framesDirectory + currPerson +"/"+currmp4File[:-4] + str(currentFrame) + ".jpg"
            # print ('Creating...' + name)
            if currentFrame%16==0:
                cv2.imwrite(name, frame)
            ret, frame = cap.read()
            # To stop duplicate images
            currentFrame += 1
        cap.release()
        print "Done!"
    

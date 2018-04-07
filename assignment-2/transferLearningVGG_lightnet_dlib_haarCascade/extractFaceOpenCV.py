import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
import os
import glob
import cv2
import numpy as np
import os
import math


sourceDestinationDirectoryList=[
("../frameData/shailendra","shailendraFace/")
]
# ("../frameData/sandeep","sandeepFace/"),


for sourceDirectory,destinationDirectory in sourceDestinationDirectoryList:
    try:
        if not os.path.exists(destinationDirectory):
            os.makedirs(destinationDirectory)
    except OSError:
        print ('Error: Creating directory of data')

    allImages = os.listdir(sourceDirectory)


    for currImage in allImages:    
        imagePath = sourceDirectory+"/"+currImage
        # WARNING : cascade XML file from this repo : https://github.com/shantnu/FaceDetect.git
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Read the image
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        print("Found {0} faces!".format(len(faces)))
        if len(faces)==1:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            plt.imshow(image)

            im = Image.open(imagePath)

            (x, y, w, h) = faces[0]
            center_x = x+w/2
            center_y = y+h/2
            b_dim = min(max(w,h)*1.2,im.width, im.height) # WARNING : this formula in incorrect
            #box = (x, y, x+w, y+h)
            box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
            # Crop Image
            crpim = im.crop(box).resize((224,224))

            plt.imshow(np.asarray(crpim))
            
            crpim.save(destinationDirectory+"/"+currImage)

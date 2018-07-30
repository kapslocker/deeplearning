import dlib
from PIL import Image
import scipy.misc
import numpy as np
import os
from skimage import io
from skimage.draw import polygon_perimeter

# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()


sdList=[
("../atul/","atulFace"),
("../flute/","fluteFace"),
("../pant/","pantFace"),
("../sandeep/","sandeepFace"),
("../shailendra/","shailendraFace"),
]

for sourceFolder,destinationFolder in sdList:
	allfiles = os.listdir(sourceFolder)

	try:
	    if not os.path.exists(destinationFolder):
	        os.makedirs(destinationFolder)
	except OSError:
	    print ('Error: Creating directory of data')

	for x in range(0,len(allfiles)):
		print(allfiles[x])
		image = scipy.misc.imread(sourceFolder+allfiles[x])
		detected_faces = face_detector(image, 1)
		fcount=0
		print (len(detected_faces))
		im = Image.open(sourceFolder+allfiles[x])
		if len(detected_faces)==1:
			face=detected_faces[0]
			box= (face.left(),face.top(),face.right(),face.bottom())
			crpim = im.crop(box).resize((224,224))    
	        crpim.save(destinationFolder+"/"+str(fcount)+allfiles[x])
			fcount+=1


import lightnet
from PIL import Image
import os
import numpy as np
import cv2


model = lightnet.load('yolo')



sdList=[
("../frameData/pant/","pantBody")
]
# ("../frameData/sandeep/","sandeepBody"),
# ("../frameData/shailendra/","shailendraBody"),
# ("../frameData/sadguru/","sadguruBody")
# ("../frameData/atul/","atulBody"),
# ("../frameData/flute/","fluteBody"),
# ("../frameData/pant/","pantBody"),



for sourceFolder,destinationFolder in sdList:
	allfiles = os.listdir(sourceFolder)
	print (len(allfiles))

	try:
		if not os.path.exists(destinationFolder):
			os.makedirs(destinationFolder)
	except OSError:
		print ('Error: Creating directory of data')
	counterV=0
	for currFrame in allfiles:
		print (counterV)
		image = lightnet.Image.from_bytes(open(sourceFolder+currFrame, 'rb').read())
		boxes = model(image)
		personbox=[currBox for currBox in boxes if currBox[1]=='person']
		# acCounter=0
		if len(personbox)==1:
			counterV+=1
			# acCounter+=1
			# print (acCounter)
			if(counterV==3000 and destinationFolder=='pantBody'):
				break
			x=personbox[0][3][0]
			y=personbox[0][3][1]
			width=personbox[0][3][2]
			height=personbox[0][3][3]
			image = Image.open(sourceFolder+currFrame)
			left=int(x-(width/2))
			top=int(y-(height/2))
			right=int(x+(width/2))
			bottom=int(y+(height/2))
			box= (left,top,right,bottom)
			crpim = image.crop(box).resize((224,224))
			crpim.save(destinationFolder+"/"+currFrame)
import cv2
import numpy as np
import time

# 
cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
(winW, winH) = (128, 128)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = gray.copy()
	for i in range(1,55):
		blur = cv2.GaussianBlur(blur,(11,11),0)

	th3 = cv2.adaptiveThreshold(gray-blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	#th3 = cv2.bitwise_not(th3)
	morph = np.zeros_like(th3)
	for i in range(1,5):
		morph = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
		morph = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

	morph = cv2.erode(morph,kernel2,iterations = 1)

	for (x, y, window) in sliding_window(morph, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
 
		# since we do not have a classifier, we'll just draw the window
		white = np.sum(window==255)
		black = np.sum(window==0)
		if(white > black):
			print('white: ', white)
			morph[y:y+winH, x:x+winW].fill(white)
		else:
			print('black: ', black)
			morph[y:y+winH, x:x+winW].fill(black)
		#cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		
		

	cv2.imshow('frame', morph)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()
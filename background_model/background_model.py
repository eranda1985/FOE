import cv2
import numpy as np
import time

# 
kernel = np.ones((5,5),np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
(winW, winH) = (64, 64)

# sliding window function
def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# return the current window as a generator
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def background_model(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = gray.copy()
	for i in range(1,55):
		blur = cv2.GaussianBlur(blur,(11,11),0)

	th = cv2.adaptiveThreshold(gray-blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	morph = np.zeros_like(th)
	for i in range(1,5):
		morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
		morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

	morph = cv2.erode(morph,kernel2,iterations = 3)
	clone = morph.copy()

	for (x, y, window) in sliding_window(morph, stepSize=32, windowSize=(winW, winH)):
		# take only windows that are of desired size
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# analyze the contents of the window
		white = np.sum(window==255)
		black = np.sum(window==0)
				
		if(white > black):
			print('white domination: ', white)
			clone[y:y+winH, x:x+winW].fill(255)
		else:
			print('black domination: ', black)
			clone[y:y+winH, x:x+winW].fill(0)

	return clone
	#cv2.imshow('frame1', clone)
	#cv2.imshow('frame2', morph)

	#if(cv2.waitKey(1) & 0xFF == ord('q')):
		#break

#cap.release()
#cv2.destroyAllWindows()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = gray.copy()
	for i in range(1,55):
		blur = cv2.GaussianBlur(blur,(11,11),0)

	th3 = cv2.adaptiveThreshold(gray-blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	morph = np.zeros_like(th3)
	for i in range(1,5):
		morph = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
		morph = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

	cv2.imshow('frame', morph)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()
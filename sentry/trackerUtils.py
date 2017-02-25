
# static method wrapper for color-based tracking to keep things clean	

import numpy as np
import cv2

def findObjects(frame,lowerBoundary,upperBoundary,kHigh,kWidth):
	mask = cv2.inRange(frame, lowerBoundary, upperBoundary) #creates mask of pixels in range of boundary
	kernel = np.ones((kHigh,kWidth),np.uint8)
	dilate = cv2.dilate(mask,kernel,iterations = 4)
	contours, hierachy = cv2.findContours(cv2.Canny(dilate,100,200),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours, dilate


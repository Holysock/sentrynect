
# static method wrapper for color-based tracking to keep things clean	

import numpy as np
import cv2

def findColoredObjects(dest,frame,lowerBoundary,upperBoundary,kHigh,kWidth,frame_context):
	mask = cv2.inRange(frame, lowerBoundary, upperBoundary) #creates mask of pixels in range of boundary
	kernel = np.ones((kHigh,kWidth),np.uint8)
	dilate = cv2.dilate(mask,kernel,iterations = 4)
	contours, _ = cv2.findContours(cv2.Canny(dilate,100,200),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return dest+[(c,frame_context) for c in contours], dilate


	
class ColorTracker():
	import numpy as np
	import cv2
	
	frame = 0
	dilate = 0
	minVal = 0
	maxVal = 0
	minLoc = 0
	center = 0

	def setNewFrame(self,frame):
		self.frame=frame

	def findObjects(self,lowerBoundary,upperBoundary,kHigh,kWidth):
		mask1 = cv2.inRange(self.frame, lowerBoundary, upperBoundary) #creates mask of pixels in range of boundary
        kernel = np.ones((kHigh,kWidth),np.uint8)
        self.dilate = cv2.dilate(mask1,kernel,iterations = 10)
		self.contours, hierarchy = cv2.findContours(cv2.Canny(dilate,100,200),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



	
class ColorTrackerUtils():
	import numpy as np
	import cv2

	frame = None
	dilate = None
	contours = None

	def setNewFrame(self,frame):
		self.frame=frame

	@staticmethod
	def findObjects(lowerBoundary,upperBoundary,kHigh,kWidth):
		mask1 = cv2.inRange(self.frame, lowerBoundary, upperBoundary) #creates mask of pixels in range of boundary
        kernel = np.ones((kHigh,kWidth),np.uint8)
        dilate = cv2.dilate(mask1,kernel,iterations = 10)
		self.contours, _ = cv2.findContours(cv2.Canny(dilate,100,200),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		return self.contours


class IdentifiedObject():
	import numpy as np
	import cv2
	import math
	from collections import deque

	past_maps = None
	past_depths = None
	past_boxes = None
	past_centers = None
	advance_flag = False
	status = "null"
	name = "Untitled"
	group = None
	speedVector = (0,0) # (radius,phi) in polar system, x = rcos(phi), y = rsin(phi)
	center = (0,0)

	def __init__(self,obj_map,box,depth,num_past,name):
		self.center = (int(box[0][0]+float(box[1][0]-box[0][0])/2),int(box[0][1]+float(box[1][1]-box[0][1])/2))		
		self.past_maps = self.deque(num_past*[self.np.zeros(obj_map.shape,dtype="uint8")],num_past)
		self.past_depths = self.deque(num_past*[0],num_past)
		self.past_boxes = self.deque(num_past*[[(-1,-1),(-1,-1)]],num_past)
		self.past_centers = self.deque(num_past*[(-1,-1)],num_past)
		self.past_maps.appendleft(obj_map)
		self.past_depths.appendleft(depth)
		self.past_boxes.appendleft(box)
		self.advance_flag = True
		self.status = "standing"
		self.name = name

	def advanceTime(self,obj_map,box,depth):
		if not self.advance_flag:
			self.center = (int(box[0][0]+float(box[1][0]-box[0][0])/2),int(box[0][1]+float(box[1][1]-box[0][1])/2))				
			self.past_maps.appendleft(obj_map)
			self.past_depths.appendleft(depth)
			self.past_boxes.appendleft(box)
			self.advance_flag = True

	def resetAdvance_flag(self):
		self.advance_flag = False

	def evaluateMovement(self,n):
		center2 = (int(self.past_boxes[n][0][0]+float(self.past_boxes[n][1][0]-self.past_boxes[n][0][0])/2),int(self.past_boxes[n][0][1]+float(self.past_boxes[n][1][1]-self.past_boxes[n][0][1])/2))
		phi = self.math.atan2((center2[1]-self.center[1]),(center2[0]-self.center[0]))
		r = 0
		if center2[0]>=0 and center2[1]>=0: r = self.math.sqrt(float((center2[0]-self.center[0]))**2+float((center2[1]-self.center[1]))**2)
		if r > 5: self.status = "moving" 
		else: self.status = "standing"
		self.speedVector = (r,phi)
		#return r,phi,self.center

	def changeStatus(self,new_status):
		self.status = new_status
		if new_status == "hiding":          #to prevent an hiding object to have a speedvector without known movement, 
			override = self.past_maps[0]   	#we assume that it did not move while hiding at all
			for m in self.past_maps:
				m = override
			self.speedVector = (0,0)

	def alreadyExists(self,obj_map): # check if already existing object and object in new Frame share same submap 
		return self.cv2.bitwise_and(obj_map,self.past_maps[0]).any()

	def checkExistenceSimple(self): # if object has not been advanced, then it does not exist anymore or went missing (simple approach)
		return self.advance_flag

	
		
	#def __eq__(self, other): TODO: finish implementation of __eq__ and __ne__
	#"""Override the default Equals behavior by checking the other objects x past and 
	#	evaluates the possibility of being an image f(x), where f is an unknown transforming function.
	#	In simple words: they must share the same past"""
	#	
	#	if 

	def getPast_maps(self):
		return self.past_maps
	def getPast_depths(self):
		return self.past_depths
	def getPast_boxes(self):
		return self.past_boxes

		
		
		
		
		

class IdentifiedObject():
	import numpy as np
	import cv2
	import math
	from collections import deque

	frame_context = None # 1 = ir, 2 = depth, 3 = color, 4 = registiered
	past_maps = None
	past_depths = None
	past_boxes = None
	past_centers = None
	past_dt = None
	advance_flag = False
	name = "Untitled"
	group = None
	center = (0,0)
	p_t = None # stores lambda expression that predicts future positions as p(t) = (x,y). Taylor up to max. 2. degree

	def __init__(self,obj_map,box,depth,num_past,name,context):
		self.frame_context = context
		self.center = (int(box[0][0]+float(box[1][0]-box[0][0])/2),int(box[0][1]+float(box[1][1]-box[0][1])/2))		
		self.past_maps = self.deque(num_past*[self.np.zeros(obj_map.shape,dtype="uint8")],num_past)
		self.past_depths = self.deque([],num_past)
		self.past_boxes = self.deque([],num_past)
		self.past_centers = self.deque([],num_past)
		self.past_dt = self.deque([],num_past-1)
		self.past_maps.appendleft(obj_map)
		self.past_depths.appendleft(depth)
		self.past_boxes.appendleft(box)
		self.advance_flag = True
		self.p_t = lambda t: self.center
		self.name = name

	def advanceTime(self,obj_map,box,depth,dt):
		if not self.advance_flag:
			self.past_centers.appendleft(self.center)
			self.center = (int(box[0][0]+float(box[1][0]-box[0][0])/2),int(box[0][1]+float(box[1][1]-box[0][1])/2))				
			self.past_maps.appendleft(obj_map)
			self.past_depths.appendleft(depth)
			self.past_boxes.appendleft(box)
			self.past_dt.appendleft(dt)
			self.advance_flag = True

	def taylorMovement(self, degree):
		(x0,y0) = self.center		
		if degree>0 and len(self.past_centers)>=1:
			vx = (x0-self.past_centers[0][0])/self.past_dt[0]
			vy = (y0-self.past_centers[0][1])/self.past_dt[0]
			if degree>1 and len(self.past_centers)>=2:
				ax = (vx-((self.past_centers[0][0]-self.past_centers[1][0])/self.past_dt[1]))/(self.past_dt[0]+self.past_dt[1])
				ay = (vy-((self.past_centers[0][1]-self.past_centers[1][1])/self.past_dt[1]))/(self.past_dt[0]+self.past_dt[1])
				self.p_t = lambda t: (int(x0 + vx*t + ax*(t**2)/2), int(y0 + vy*t + ay*(t**2)/2))
			else: self.p_t = lambda t: (int(x0 + vx*t), int(y0 + vy*t))
		else: self.p_t = lambda t: (x0,y0)

	def changeStatus(self,new_status):
		if new_status == "hiding":         	#to prevent an hiding object to have a speedvector without known movement, 
			override = self.past_maps[0]   	#we assume that it did not move while hiding at all
			for m in self.past_maps:
				m = override
			self.p_t = lambda t: self.center

	def resetAdvance_flag(self):
		self.advance_flag = False

	def alreadyExists(self,obj_map): # check if already existing object and object in new Frame share some pixels
		return self.cv2.bitwise_and(obj_map,self.past_maps[0]).any()

	def checkExistenceSimple(self): # if object has not been advanced, then it does not exist anymore or went missing (simple approach)
		return self.advance_flag

	def getPast_maps(self):
		return self.past_maps

	def getPast_depths(self):
		return self.past_depths

	def getPast_boxes(self):
		return self.past_boxes

#	def evaluateMovement(self,n):
#		if n < len(self.past_boxes): pass
#		else: n = len(self.past_boxes)-1 
#		center2 = (int(self.past_boxes[n][0][0]+float(self.past_boxes[n][1][0]-self.past_boxes[n][0][0])/2),int(self.past_boxes[n][0][1]+float(self.past_boxes[n][1][1]-self.past_boxes[n][0][1])/2))
#		phi = self.math.atan2((center2[1]-self.center[1]),(center2[0]-self.center[0]))
#		r = 0
#		if center2[0]>=0 and center2[1]>=0: r = self.math.sqrt(float((center2[0]-self.center[0]))**2+float((center2[1]-self.center[1]))**2)
#		self.speedVector = (r,phi)
#		return r,phi,self.center

#	def weightedMovement():
#		dx_list = []
#		v_list = []
#		for x in self.past_centers:
#			dx 
#		xt = 
		
		
		
		

from collections import deque
import numpy as np
import argparse
import time
import math
import copy
import cv2
import sys
#lol
import trackerUtils as tracker
import identifiedObject
import mathUtils
import kinect

# setting up and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--debugg",help="Shows various debugg-messages (currently not in use)",action="store_true")
ap.add_argument("-u","--use",required=True,help="Chooses with camera to use. Options: color, depth, both")
ap.add_argument("-t","--track",help="Set tracking method. Options: ""reflection"" tracks reflectors (like cateyes), ""pedestrian"" tracks humans via HOG")
ap.add_argument("--show",help="Usage: --show ""<allThingsToShow>"" things to show: ir, color, depth, registered, mask, dilate, target, contours")
ap.add_argument("--fps",help="Enables FPS prints",action="store_true")
args = ap.parse_args()

debugg, show_ir, show_color, show_depth, show_registered, show_mask, show_dilate, show_target, show_contours, use_color, use_depth = (False,)*11
show = args.show
use = args.use
if args.debugg: debugg = True
if show and"ir" in show: show_ir = True
if show and"color" in show: show_color = True
if show and"depth" in show: show_depth = True
if show and"registered" in show: show_registered = True
if show and"contours" in show: show_contours = True

if use == "color": use_color = True
elif use == "depth": use_depth = True
elif use == "both": use_color = True; use_depth = True
else:
	print "unknown argument for ""--use"": "+use
	sys.exit(1)

if args.fps:
	fpsCountMAX = 10
	fpsCount = 0
	fps_t = time.clock()

#########################################################################
# preps for main loop

# initializing and starting kinect
kinect = kinect.Kinect(use_color,use_depth)
if debugg: print "Debugg: Kinect initialized."
kinect.startKinect()
if debugg: print "Debugg: Kinect started."


num_past = 5

obj_list = deque([]) # contains all objects in current frame
obj_missing_list = deque([],20) # contains all objects that went missing over the last frames due to unknown circumstances
obj_hiding_list = deque([],20) # contains all objects that hide behind an obstacle or an another object

loop_time = time.clock()

# main loop
while True:
	# grabbing a new frame
	kinect.grabFrame()
	if use_color: colorFrame = kinect.getFrame("color")
	if use_depth: irFrame = kinect.getFrame("ir")
	if use_depth: depthFrame = kinect.getFrame("depth")
	if use_color and use_depth: registeredFrame = kinect.getFrame("registered")
	kinect.releaseFrame()

##############################################################################
	# here is where the real stuff is happening
	obj_contours, obj_global_map = tracker.findObjects(irFrame,230,255,3,3) #finds simple objects wich satisfy certain criteria in a given frame

	obj_list_copy = copy.copy(obj_list) 
	for obj in obj_list_copy:                     #iterates throu list of already known objects and checks existance against objects of current frame
		if not obj.checkExistenceSimple():    #checks if known object still exists
			obj_list.remove(obj)	      #no? then remove it and evalue the reason why it disappeared
			if mathUtils.checkHiding(obj,depthFrame,10,0.01,20): #is it hiding behind an obstacle?
				print "{0} is hiding!".format(obj.name)###
				obj.changeStatus("hiding")
				obj_hiding_list.appendleft(obj)      #if its hiding, save in special list for hiding objects for later use.

			elif mathUtils.checkIfNotInScreen(obj,irFrame.shape): #left the object the screen? Then kiss it goodbye.
				print "Object {0} left the screen.".format(obj.name)###

			else: 	#if object is not hiding nor left the screen then save in special list for missing objects. It might come back later.
				print "Object {0} went missing.".format(obj.name) ###
				obj.changeStatus("missing")
				obj_missing_list.appendleft(obj)
		else: obj.resetAdvance_flag() #if object still exists, reset advance flag so time can advance


	dt = time.clock()-loop_time #dt stores times between two frames
	loop_time = time.clock() 

	for obj_contour in obj_contours: #iterates throu list of new objects and tries to retrieve missing or hiding objects, also adds any complete new object to known objects list.
		obj_list_copy = copy.copy(obj_list)
		obj_missing_list_copy = copy.copy(obj_missing_list)
		obj_hiding_list_copy = copy.copy(obj_hiding_list)
		ex_flag = False
		P1,P2 = mathUtils.getBoxofContour(obj_contour)
		obj_map = np.zeros(irFrame.shape,dtype=np.uint8)
		cv2.rectangle(obj_map,P1,P2,255,-1)
		obj_map = cv2.bitwise_and(obj_map,obj_global_map)
		obj_HMdepth = mathUtils.harmonMeanRoi((P1,P2),depthFrame)
		cv2.rectangle(irFrame,P1,P2,255, 1)
		for obj in obj_list_copy: #finds already existing objects and advances their time
			if obj.alreadyExists(obj_map):
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth,dt)
				obj.taylorMovement(2)
				ex_flag = True
				#print "Found existing object. {0}".format(obj.name) ### keep commented to avoid loads of spam :)
				break
		if ex_flag: continue
		for obj in obj_missing_list_copy: #finds missing objects and advances their time
			if obj.alreadyExists(obj_map):
				obj_list.appendleft(obj)
				obj_missing_list.remove(obj)
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth,dt)
				obj.taylorMovement(2)
				ex_flag = True
				print "Found missing object {0}.".format(obj.name) ###
				break
		if ex_flag: continue
		for obj in obj_hiding_list_copy: #finds hiding objects and advances their time
			if obj.alreadyExists(obj_map):
				obj_list.appendleft(obj)
				obj_hiding_list.remove(obj)
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth,dt)
				obj.taylorMovement(2)
				ex_flag = True
				print "Found hiding object {0}.".format(obj.name) ###
				break
		if ex_flag: continue 
		# if current object is not an existing, missing or hiding known one, its a new one.
		obj_list.appendleft(identifiedObject.IdentifiedObject(obj_map,(P1,P2),obj_HMdepth,num_past,"0"))#mathUtils.giveName("simple"
		print "Found a new object. Name: {0}".format(0) ###
		cv2.rectangle(irFrame,P1,P2,255, -1) ###

	for obj in obj_list: ### Draws lines to show velocity and direction of an object
		p_t = obj.p_t
		time_step = 0.02
		for i in xrange(5):
			cv2.line(irFrame,p_t(time_step*i),p_t(time_step*(i+1)),255,1)
		for i in xrange(len(obj.past_centers)-1):
			cv2.line(irFrame,obj.past_centers[i],obj.past_centers[i+1],255,2)	

		
	#print len(obj_list)
#########################################################################################################

	if show_contours: cv2.drawContours(irFrame, obj_contours, -1, 255, 1)
	if show_color and use_color: cv2.imshow("color", cv2.resize(colorFrame,(int(1920/3),int(1080/3))))
	if show_ir and use_depth: cv2.imshow("IR", irFrame)
	if show_depth and use_depth: cv2.imshow("depth", depthFrame)
	if show_registered and use_color and use_depth: cv2.imshow("registered", registeredFrame)

	# simple fps counter. Derives loop-counts by the time : dcount/dt
	if args.fps:
		if fpsCount <= fpsCountMAX:
			fpsCount += 1
		else:
			fps_dt = time.clock()-fps_t
			print("FPS: {0}" .format(int(fpsCount/float(fps_dt))))
			fpsCount = 0
			fps_t = time.clock()

	key = cv2.waitKey(delay=1)
	if key == ord('q'):
		break

kinect.close()
sys.exit(0)

#if last_dilate == None:
	#	last_dilate = dilate
	#if rois_color:
		#print "arithmitic mean : {0} harmonic mean: {1}".format(mathUtils.arithMeanRoi(rois_color[0],depthFrame),mathUtils.harmonMeanRoi(rois_color[0],depthFrame))


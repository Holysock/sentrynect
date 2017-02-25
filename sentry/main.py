from collections import deque
import numpy as np
import argparse
import time
import math
import copy
import cv2
import sys

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
#if show and"mask" in show: show_mask = True
#if show and"dilate" in show: show_dilate = True
#if show and"target" in show: show_target = True
if use == "color": use_color = True
elif use == "depth": use_depth = True
elif use == "both": use_color = True; use_depth = True
else:
	print "unknown argument for ""--use"": "+use
	sys.exit(1)

if args.fps:
	fpsCountMAX = 10
	fpsCount = 0
	t = time.clock()

# initializing and starting kinect
kinect = kinect.Kinect(use_color,use_depth)
if debugg: print "Debugg: Kinect initialized."
kinect.startKinect()
if debugg: print "Debugg: Kinect started."

last_frame = None
last_frame1 = None
last_frame2 = None
last_dilate = None

obj_list = deque([]) # contains all objects in current frame
obj_missing_list = deque([],20) # contains all objects that went missing over the last frames due to unknown circumstances
obj_hiding_list = deque([],20) # contains all objects that hide behind an obstacle or an another object

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
	obj_contours, obj_global_map = tracker.findObjects(irFrame,230,255,3,3)
	obj_list_copy = copy.copy(obj_list)
	for obj in obj_list_copy:
		if not obj.checkExistenceSimple():
			obj_list.remove(obj)

			if mathUtils.checkHiding(obj,depthFrame,10): 
				print "{0} is hiding!".format(obj.name)###
				obj.changeStatus("hiding")
				obj_hiding_list.appendleft(obj)

			elif mathUtils.checkIfNotInScreen(obj,irFrame.shape):
				print "Object {0} left the screen.".format(obj.name)###

			else: 
				print "Object {0} went missing.".format(obj.name) ###
				obj.changeStatus("missing")
				obj_missing_list.appendleft(obj)
		else: obj.resetAdvance_flag()



	for obj_contour in obj_contours:
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
		#print obj_HMdepth ###
		for obj in obj_list_copy:
			if obj.alreadyExists(obj_map):
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth)
				obj.evaluateMovement(5)
				ex_flag = True
				break
		if ex_flag:
			#print "Found existing object. {0}".format(obj.name) ### keep commented to avoid loads of spam :)
			continue
		for obj in obj_missing_list_copy:
			if obj.alreadyExists(obj_map):
				obj_list.appendleft(obj)
				obj_missing_list.remove(obj)
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth)
				obj.evaluateMovement(5)
				ex_flag = True
		if ex_flag:
			print "Found missing object {0}.".format(obj.name) ###
			continue
		for obj in obj_hiding_list_copy:
			if obj.alreadyExists(obj_map):
				obj_list.appendleft(obj)
				obj_hiding_list.remove(obj)
				obj.advanceTime(obj_map,(P1,P2),obj_HMdepth)
				ex_flag = True
		if ex_flag:
			print "Found hiding object {0}.".format(obj.name) ###
			continue
		print "Found a new object." ###
		obj_list.appendleft(identifiedObject.IdentifiedObject(obj_map,(P1,P2),obj_HMdepth,6,"0"))
		cv2.rectangle(irFrame,P1,P2,255, -1) ###

	for obj in obj_list: ### Draws lines to show velocity and direction of an object
		(r,phi) = obj.speedVector
		point0 = obj.center
		point1 = (int(point0[0]-r*math.cos(phi)),int(point0[1]-r*math.sin(phi)))
		cv2.line(irFrame,point0,point1,255, 1)
		
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
			dt = time.clock()-t
			print("FPS: {0}" .format(int(fpsCount/float(dt))))
			fpsCount = 0
			t = time.clock()

	key = cv2.waitKey(delay=1)
	if key == ord('q'):
		break

kinect.close()
sys.exit(0)

#if last_dilate == None:
	#	last_dilate = dilate
	#if rois_color:
		#print "arithmitic mean : {0} harmonic mean: {1}".format(mathUtils.arithMeanRoi(rois_color[0],depthFrame),mathUtils.harmonMeanRoi(rois_color[0],depthFrame))


import numpy as np
import cv2
import sys
import time
import argparse
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

ap = argparse.ArgumentParser()
ap.add_argument("-p","--pipeline",help="Specifies a pipline to use. Options: ""gl"" OpenGL(default), ""cl"": OpenCL, ""cpu"": Cpu(slow) ")
ap.add_argument("-d","--debugg",help="Shows various debugg-messages (currently not in use)",action="store_true")
ap.add_argument("-t","--track",help="Set tracking method. Options: ""reflection"" tracks reflectors (like cateyes), ""pedestrian"" tracks humans via HOG")
ap.add_argument("--show",help="Usage: --show <allThingsToShow> things to show: IR, color, depth, registered, mask, dilate, target, contours")
ap.add_argument("--fps",help="Enables FPS prints",action="store_true")
args = ap.parse_args()

debugg, show_ir, show_color, show_depth, show_registered, show_mask, show_dilate, show_target, show_contours = (False,)*9
enable_rgb = False
enable_depth = True

if args.pipeline == "cpu":
	from pylibfreenect2 import CpuPacketPipeline
	pipeline = CpuPacketPipeline()
elif args.pipeline == "cl":
	from pylibfreenect2 import OpenCLPacketPipeline
	pipeline = OpenCLPacketPipeline()
else: 
	from pylibfreenect2 import OpenGLPacketPipeline
	pipeline = OpenGLPacketPipeline()
if args.debugg:
	debugg = True
show = args.show
if "IR" in show:
	show_ir = True
if "color" in show:
	show_color = True
	enable_rgb = True
if "depth" in show:
	show_depth = True
if "registered" in show:
	show_registered = True
	enable_rgb = True
if "mask" in show:
	show_mask = True
if "dilate" in show:
	show_dilate = True
if "target" in show:
	show_target = True
if "contours" in show:
	show_contours = True

if args.fps:
	fpsCountMAX = 10
	fpsCount = 0
	t = time.clock()

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)


def getROIs(frame):
	contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def getMeanDepth(roi, depthFrame): # computes the mean depth of given region of interest
	depthMap = []
	for p in roi: 
		if depthFrame[p[0][1],p[0][0]]!=0:
			depthMap.append(depthFrame[p[0][1],p[0][0]])	
		if not depthMap: return -1
	return int(np.mean(depthMap))	

while True: #main loop
    frames = listener.waitForNewFrame()

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]
    if enable_rgb and enable_depth:
        registration.apply(color, depth, undistorted, registered)
    elif enable_depth:
        registration.undistortDepth(depth, undistorted)

    if enable_depth:
        irNorm = np.array((ir.asarray() / 65535.) * 255, dtype = np.uint8) 
        #depthNorm = np.array((depth.asarray() / 4500.) * 255, dtype = np.uint8)
        undistortedNorm = np.array((undistorted.asarray(np.float32) / 4500.) * 255, dtype = np.uint8)

        mask1 = cv2.inRange(irNorm, 230, 255) #creates mask of very bright (reflecting) objects
        kernel = np.ones((5,2),np.uint8) #uses a 5x2 matrix to get a better depth perception of humans  
        dilate = cv2.dilate(mask1,kernel,iterations = 10)
        dist = cv2.distanceTransform(dilate,cv2.cv.CV_DIST_L2,cv2.cv.CV_DIST_MASK_PRECISE)#creates a distance map to find center point
        cv2.normalize(dist,dist,alpha = 0.0, beta = 1.0, norm_type=cv2.NORM_MINMAX)
        minVal,maxVal,minLoc,center = cv2.minMaxLoc(dist)
        contours = getROIs(cv2.Canny(dilate,100,200)) #creats contours wich contain regions of interests

        if show_target and not minVal == maxVal: #shows a cross on the currently choosen target
            cv2.line(irNorm,(center[0]-20,center[1]),(center[0]+20,center[1]),255,2) #cross on target
            cv2.line(irNorm,(center[0],center[1]-20),(center[0],center[1]+20),255,2)
            cv2.line(irNorm,(center[0],center[1]),(int(512/2),int(424/2)),255,2) #line to target

        for roi in contours: 
            print "Mean depth: {0}".format(getMeanDepth(roi,undistortedNorm))

        if show_contours: cv2.drawContours(irNorm, contours, -1, 255, 3)
        if show_ir: cv2.imshow("IR", irNorm)
        if show_depth: cv2.imshow("depth", undistortedNorm)
        if show_mask: cv2.imshow("dist", dist)
        if show_dilate: cv2.imshow("dilate", dilate)

    if enable_rgb:
        colorResz = cv2.resize(color.asarray(),(int(1920 / 3), int(1080 / 3)))
        if show_color: cv2.imshow("color", colorResz)
    if enable_rgb and enable_depth:
        if show_registered: cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    if args.fps:
        if fpsCount <= fpsCountMAX: # simple fps counter. Derives loop-counts by the time : dcount/dt
            fpsCount += 1
        else:
            dt = time.clock()-t
            print("FPS: {0}" .format(int(fpsCount/float(dt))))
            fpsCount = 0
            t = time.clock()

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)

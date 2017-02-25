import numpy as np
import cv2
import sys
import time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import OpenGLPacketPipeline
pipeline = OpenGLPacketPipeline()

fpsCountMAX = 10
fpsCount = 0
t = time.clock()

enable_rgb = False
enable_depth = True

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

hog = cv2.HOGDescriptor() # initialize the HOG descriptor/person detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
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
        irNorm = ir.asarray() / 65535.
        depthNorm = depth.asarray() / 4500.
        undistortedNorm = undistorted.asarray(np.float32) / 4500.
        irNorm = np.array(irNorm * 255, dtype = np.uint8)
        (rects, weights) = hog.detectMultiScale(irNorm, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
		    cv2.rectangle(irNorm, (x, y), (x + w, y + h), (255), 2)

        cv2.imshow("ir", irNorm)
        cv2.imshow("depth", depthNorm)
        #cv2.imshow("undistorted", undistortedNorm)
        #hist = cv2.calcHist([ir.asarray()],[0],None,[16],[0,256])
        #print sorted(hist)[15]
    if enable_rgb:
        colorResz = cv2.resize(color.asarray(),(int(1920 / 5), int(1080 / 5)))
        #gray_image = cv2.cvtColor(colorResz, cv2.COLOR_BGR2GRAY)
        #(rects, weights) = hog.detectMultiScale(gray_image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        #for (x, y, w, h) in rects:
		#    cv2.rectangle(colorResz, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("color", colorResz)
    if enable_rgb and enable_depth:
        cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    if fpsCount <= fpsCountMAX: # simple fps counter. Derives loop-counts by the time : dcount/dt
        fpsCount += 1
    else:
        dt = time.clock()-t
        print("FPS: {0}" .format(fpsCount/float(dt)))
        fpsCount = 0
        t = time.clock()

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)

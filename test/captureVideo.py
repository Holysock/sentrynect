import numpy as np
import cv2
import sys
import time
import skvideo.io
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import OpenGLPacketPipeline

filename = sys.argv[1]

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
if enable_rgb: writerColor = skvideo.io.FFmpegWriter("output/"+filename+"Color.mp4")
if enable_depth: writerIR = skvideo.io.FFmpegWriter("output/"+filename+"IR.mp4")
if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

while True:
    frames = listener.waitForNewFrame()

    if enable_rgb:
        color = frames["color"]
        colorNorm = np.array(cv2.resize(color.asarray(),(int(1920 / 2), int(1080 / 2))), dtype = np.uint8)
        cv2.imshow("color", colorNorm)
        writerColor.writeFrame(colorNorm)
    if enable_depth:
        ir = frames["ir"]
        irNorm = np.array((ir.asarray() / 65535.) * 255, dtype = np.uint8)
        cv2.imshow("ir", irNorm)
        writerIR.writeFrame(irNorm)
    listener.release(frames)
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

device.stop()
device.close()
outColor.release()
outIR.release()
sys.exit(0)

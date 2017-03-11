class Kinect():
	import sys
	import numpy as np
	from pylibfreenect2 import Freenect2, SyncMultiFrameListener
	from pylibfreenect2 import FrameType, Registration, Frame
	from pylibfreenect2 import OpenCLPacketPipeline

	serial = None
	device = None
	registration = None
	frames = None
	fn = None
	enable_rgb = False
	enable_depth = False
	undistorted = Frame(512, 424, 4)
	registered = Frame(512, 424, 4)
	

	def __init__(self,color,depth):
		self.fn = self.Freenect2()
		num_devices = self.fn.enumerateDevices()
		if num_devices == 0:
		    print("No device connected!")
		    self.sys.exit(1)
		self.serial = self.fn.getDeviceSerialNumber(0)
		self.device = self.fn.openDevice(self.serial, pipeline=self.OpenCLPacketPipeline())
		types = 0
		if color: types |= self.FrameType.Color; self.enable_rgb = color
		if depth: types |= (self.FrameType.Ir | self.FrameType.Depth); self.enable_depth = depth
		self.listener = self.SyncMultiFrameListener(types)
		self.device.setColorFrameListener(self.listener)
		self.device.setIrAndDepthFrameListener(self.listener)

	def startKinect(self):
		if self.enable_rgb and self.enable_depth:
		    self.device.start()
		else:
		    self.device.startStreams(rgb=self.enable_rgb, depth=self.enable_depth)
		if self.enable_depth:
			self.registration = self.Registration(self.device.getIrCameraParams(),self.device.getColorCameraParams())

	def grabFrame(self):
		self.frames = self.listener.waitForNewFrame()

	def getFrame(self,frameType):
		if frameType == "color":
			frame = self.frames["color"]
			return self.np.array(frame.asarray(), dtype = self.np.uint8)
		elif frameType == "ir":
			frame = self.frames["ir"]
			return self.np.array((frame.asarray()*(255/65535.)), dtype = self.np.uint8)
		elif frameType == "depth":
			frame = self.frames["depth"]
			return self.np.array((frame.asarray(self.np.float32)*(255/4500.)), dtype = self.np.uint8)
		elif frameType == "registered":
			self.registration.apply(self.frames["color"], self.frames["depth"], self.undistorted, self.registered)
			return self.registered

	def releaseFrame(self):
		self.listener.release(self.frames)

	def close(self):
		self.device.stop()
		self.device.close()
	
		
			
	
		

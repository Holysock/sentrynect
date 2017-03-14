import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy import stats
import numpy as np
import math

def harmonMeanRoi((P1,P2), mat):
	m = mat[P1[1]:P2[1],P1[0]:P2[0]]
	#print len(m)
	mod = SourceModule("""
	__global__ void averageWithZeroSupression(float *dataOut, float *dataIn){
		extern __shared__ float sdata[];
		const u_int id = threadIdx.x;
		const u_int zNum_adress = blockDim.x;

		sdata[id] = dataIn[id];
		if(dataIn[id]==0) atomicAdd(&sdata[zNum_adress], 1);

		__syncthreads();
		for(int s = 1; s < blockDim.x; s *= 2){
			u_int index = 2 * s *id;
			if(index+s<blockDim.x) sdata[index] += sdata[index + s];
			__syncthreads();
		}
		if(id==0){ 
			u_int n = blockDim.x-sdata[zNum_adress];
			if(n>0) dataOut[0] = sdata[0]/n; 
			else dataOut[0] = 0;
		}
	}
	""")
	avr = mod.get_function("averageWithZeroSupression")
	m_flat = m.flatten.astype(np.float32)
	m_avr = np.zeros(1).astype(np.float32)
	avr(drv.Out(m_avr),drv.In(m_flat),block=(len(m_flat),1,1),grid=(1,1), shared=m_flat.nbytes*2)
	return int(m_avr[0])
	

#	for r in m:
#		for v in r:
#			if v>0:
#				m_new.append(v)
#	if m_new == []: return -1
#	return int(stats.hmean(m_new))

def getBoxofContour(x):
	p_max = [[0,0]]
	p_min = x[0].copy()
	for p in x:
		if p[0][0]<p_min[0][0]: p_min[0][0]=p[0][0]
		if p[0][0]>p_max[0][0]: p_max[0][0]=p[0][0]
		if p[0][1]<p_min[0][1]: p_min[0][1]=p[0][1]
		if p[0][1]>p_max[0][1]: p_max[0][1]=p[0][1]
	return (p_min[0][0],p_min[0][1]),(p_max[0][0],p_max[0][1])

def checkLoop(x):
	p_last = x[0]
	p_first = x[0]
	for p in x:
		if abs(p[0][0]-p_last[0][0])>1 or abs(p[0][1]-p_last[0][1])>1:
			print "found faulty loop! {0} {1}".format(abs(p[0][0]-p_last[0][0]),abs(p[0][1]-p_last[0][1]))
			#fix_line = lineIt(p_last[0],p[0])
			#print fix_line
		p_last = p

def checkHiding(obj,depth_map,tolerance,time_step,points_num):
	p_t = obj.p_t
	depth_values = []
	for i in xrange(points_num):
		p = p_t(time_step*i)
		if not(p[0]>=0 and p[1]>=0 and p[0]<depth_map.shape[1] and p[1]<depth_map.shape[0]): continue 
		value = depth_map[p[1]][p[0]]
		if value not in depth_values and value > 0: depth_values.append(value)
	if depth_values == []: return False
	if obj.getPast_depths()[0]-stats.hmean(depth_values)>tolerance: return True
	return False

#	(r,phi) = obj.speedVector
#	pos_1 = np.array(obj.center)
#	pos_2 = np.array((int(pos_1[0]-r*math.cos(phi)),int(pos_1[1]-r*math.sin(phi))))
#	obj_depth = obj.getPast_depths()[0]
#	line = lineIt(pos_1,pos_2,depth_map.shape)
#	depth_values = []
#	#print pos_1,pos_2,line
#	for p in line:
#		if depth_map[p[1]-1][p[0]-1] > 0:
#			depth_values.append(depth_map[p[1]-1][p[0]-1])
#			#print depth_map[p[1]-1][p[0]-1]
#	if depth_values != []:
#		if obj_depth-stats.hmean(depth_values)>tolerance: return True
#		else: return False
#	return False

def checkIfNotInScreen(obj,shape):
	return False
	(r,phi) = obj.speedVector
	pos_1 = obj.center
	pos_2 = (pos_1[0]-r*math.cos(phi),pos_1[1]-r*math.sin(phi))		
	return (pos_2[0]<=0 or pos_2[1]<=0 or pos_2[0]>=shape[1] or pos_2[1]>=shape[0])

def lineIt(P1, P2,shape):
	#Thanks to mohikhsan, http://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
	P1X = P1[0]
	P1Y = P1[1]
	P2X = P2[0]
	P2Y = P2[1]

	dX = P2X - P1X
	dY = P2Y - P1Y
	dXa = np.abs(dX)
	dYa = np.abs(dY)

	itbuffer = np.empty(shape=(np.maximum(dYa,dXa),2))
	itbuffer.fill(np.nan)

	(imageH,imageW) = shape

   #Obtain coordinates along the line using a form of Bresenham's algorithm
	negY = P1Y > P2Y
	negX = P1X > P2X
	if P1X == P2X: #vertical line segment
		itbuffer[:,0] = P1X
		if negY:
			itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
		else:
			itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
	elif P1Y == P2Y: #horizontal line segment
		itbuffer[:,1] = P1Y
		if negX:
			itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
		else:
			itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
	else: #diagonal line segment
		steepSlope = dYa > dXa
		if steepSlope:
			slope = dX.astype(np.float32)/dY.astype(np.float32)
			if negY:
				itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
			else:
				itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
				itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
		else:
			slope = dY.astype(np.float32)/dX.astype(np.float32)
			if negX:
				itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
			else:
				itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
			itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
	colX = itbuffer[:,0]
	colY = itbuffer[:,1]
	itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

	return itbuffer.astype(int)


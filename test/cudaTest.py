import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

from scipy import stats

mod = SourceModule("""
__global__ void addThem(float *dest, float *a, float *b){
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	dest[index] = a[index]+b[index];
}

__global__ void reduce(int *dataOut, int *dataIn){
	extern __shared__ int sdata[];
	const int id = blockIdx.x * blockDim.x + threadIdx.x; // index global ueber alle bloecke
	const int tid = threadIdx.x; //index in block share
	sdata[tid] = dataIn[id]; // element wird von globalem speicher in blockweise shared geladen.

	__syncthreads();

	for(int s = 1; s < blockDim.x; s *= 2){
		int index = 2 * s *tid;
		if(index<blockDim.x) sdata[index] += sdata[index + s];
		__syncthreads();
	}

	if(tid==0) dataOut[blockIdx.x] = sdata[0];
}

__global__ void averageWithZeroSupression(float *dataOut, float *dataIn){
	extern __shared__ float sdata2[];
	const u_int id = threadIdx.x;
	const u_int zNum_adress = blockDim.x;

	sdata2[id] = dataIn[id];
	if(dataIn[id]==0) atomicAdd(&sdata2[zNum_adress], 1);

	__syncthreads();
	for(int s = 1; s < blockDim.x; s *= 2){
		u_int index = 2 * s *id;
		if(index+s<blockDim.x) sdata2[index] += sdata2[index + s];
		__syncthreads();
	}
	if(id==0){ 
		u_int n = blockDim.x-sdata2[zNum_adress];
		if(n>0) dataOut[0] = sdata2[0]/n; 
		else dataOut[0] = 0;
	}
}

__global__ void hMeanWithZeroSupression(float *dataOut, float *dataIn){
	extern __shared__ float sdata3[];
	const u_int id = threadIdx.x;
	const u_int zNum_adress = blockDim.x*2;

	sdata3[id] = dataIn[id];
	if(dataIn[id]==0) atomicAdd(&sdata3[zNum_adress], 1);

	__syncthreads();
	u_int first = 1;
	for(int s = 1; s < blockDim.x; s *= 2){
		u_int index = 2 * s *id;
		if(first==1 && index+s<blockDim.x){
			float x0 = 0;
			float x1 = 0;
			if(sdata3[index] > 0) x0 = 1/sdata3[index];
			if(sdata3[index+s] > 0) x1 = 1/sdata3[index+s];
			sdata3[index] = x0+x1;
			first = 0;
		}
		else if(index+s<blockDim.x) sdata3[index] += sdata3[index + s];
		__syncthreads();
	}
	if(id==0){ 
		int n = blockDim.x-sdata3[zNum_adress];
		if(n>0) dataOut[0] = n/sdata3[0]; 
		else dataOut[0] = 0;
		dataOut[1] = sdata3[0];
	}
}
""")

red = mod.get_function("reduce")


numBlocks = 1
numThreads = 101

c = np.ones(101).astype(np.int32)
c_red = np.zeros(numBlocks).astype(np.int32)

for i in xrange(len(c)):
	c[i] = i

red(drv.Out(c_red),drv.In(c),block=(numThreads,1,1),grid=(numBlocks,1), shared=c.nbytes*2/numBlocks)

print c_red

avr = mod.get_function("averageWithZeroSupression")
hm = mod.get_function("hMeanWithZeroSupression")## broken shit

d = np.zeros([7]).astype(np.float32)
d_avr = np.zeros(1).astype(np.float32)
d_hm = np.zeros(2).astype(np.float32)

for i in xrange(len(d)):
	d[i] = i

avr(drv.Out(d_avr),drv.In(d),block=(len(d),1,1),grid=(1,1), shared=d.nbytes*2)
hm(drv.Out(d_hm),drv.In(d),block=(len(d),1,1),grid=(1,1), shared=d.nbytes*2)

print d_avr,d_hm,stats.hmean(d[1:])
















/*
 * Reduces an array of n floats to blockSize elements
 * npts4mean is used in the final reduce stage 
 * to divide the result by the number of elements to get the mean
 *
 */
template <unsigned int blockSize,typename T>
__global__ void reduce(T* idata, T* odata, const unsigned n, const unsigned npts4mean) {
    extern __shared__ T mean[];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    mean[tid] = (T)0;


    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mean[tid] += idata[i] + idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { mean[tid] += mean[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { mean[tid] += mean[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { mean[tid] += mean[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { mean[tid] += mean[tid + 32]; }
        if (blockSize >=  32) { mean[tid] += mean[tid + 16]; }
        if (blockSize >=  16) { mean[tid] += mean[tid +  8]; }
        if (blockSize >=   8) { mean[tid] += mean[tid +  4]; }
        if (blockSize >=   4) { mean[tid] += mean[tid +  2]; }
        if (blockSize >=   2) { mean[tid] += mean[tid +  1]; }
    }

    // write result for this block to global mem 
    if (tid == 0 && npts4mean) odata[blockIdx.x] = mean[0]/(float)npts4mean;
    if (tid == 0 && npts4mean==0) odata[blockIdx.x] = mean[0];


}

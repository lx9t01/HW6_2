// gillespie_cuda.cuh Haixiang Xu


#ifndef CUDA_FFT_CONVOLVE_CUH
#define CUDA_FFT_CONVOLVE_CUH

void cudaCallGillKernel(const int blocks,
    const int threadsPerBlock,
    float* dev_points, 
    float* dev_points_2, 
    float* state,
    float* X, 
    float* dev_timestep,
    float* dev_accu_time,
    const int N);

void cudaCallFindMinKernel(const int blocks,
    const int threadsPerBlock,
    float* dev_accu_time,
    float* dev_min_time,
    const int N);

void cudaCallResampleKernel(const int blocks, 
    const int threadsPerBlock, 
    float* dev_resample_X, 
    float* dev_X, 
    float* dev_accu_time, 
    const int N, 
    const int T);



#endif
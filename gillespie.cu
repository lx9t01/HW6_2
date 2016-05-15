// gillespie.cu Haixiang Xu
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include "gillespie_cuda.cuh"

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
/*
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
*/


// a single iteration of the Gillespie algorithm on 
// the given system using an array of random numbers 
// given as an argument.
__global__
void cudaGillKernel(float* dev_points,
    float* dev_points_2,
    float* state,
    float* X, 
    float* dev_timestep,
    float* dev_accu_time,
    const int N) {

    const float kon = 0.1;
    const float koff = 0.9;
    const float b = 10.0;
    const float g = 1.0;

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < N) {

        if (state[idx] < 0.5){
            float lamda = kon + X[idx] * g;
            dev_timestep[idx] = -logf(dev_points[idx]) / lamda;
            dev_accu_time[idx] += dev_timestep[idx];
            if (dev_points_2[idx] > kon / lamda) { // if X--
                X[idx]--;
            } else { // if OFF --> ON
                state[idx] = 1;
            }
        } else {
            float lamda = koff + b + X[idx] * g;
            dev_timestep[idx] = -logf(dev_points[idx]) / lamda;
            dev_accu_time[idx] += dev_timestep[idx];
            if (dev_points_2[idx] <= koff / lamda) { // ON --> OFF
                state[idx] = 0;
            } else if (dev_points_2[idx] <= (koff + b) / lamda) { // X++
                X[idx]++;
            } else { // X--
                X[idx]--;
            }
        }
        __syncthreads();
        idx += blockDim.x * gridDim.x;
    }
}

// a kernel to use reduction to find minimum
__global__
void cudaFindMinKernel (
    float* dev_timestep,
    float* min_timestep, // NEED TO ALLOCATE
    const int N) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float data[64]; // rememeber to update this !!!!!!!

    for (int i = 0; i < 64; ++i){
        data[i] = 99999;
    }
    __syncthreads();

    while (idx < N) {
        {
            atomicMin(&data[threadIdx.x], dev_timestep[idx]);
        }
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    int l = blockDim.x;
    while (l >= 1) {
        l /= 2;
        if (threadIdx.x < l) {
            data[threadIdx.x] = (data[threadIdx.x]<data[threadIdx.x + l])? data[threadIdx.x]:data[threadIdx.x + l];
        }
        __syncthreads();
    }
    *min_timestep = data[0];
    // printf("%f\n", min_timestep);


}

__global__
void cudaResampleKernel(
    float* dev_resample_X, 
    float* dev_X, 
    float* dev_accu_time, 
    const int N, 
    const int T) {
    // TODO
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < N) {
        int i = floorf(dev_accu_time[idx] * 10);
        for (int j = 0; j < i && j < T; ++j) {
            if (dev_resample_X[idx * T + j] < 0) {
                dev_resample_X[idx * T + j] = dev_X[idx];
            }
        }
        // while (dev_is_resampled[idx * T + i] == 0 && i >= 0 && i < T) {
        //     dev_is_resampled[idx * T + i] = 1;
        //     dev_resample_X[idx * T + i] = dev_X[idx];
        //     i--;
        // }
        idx += blockDim.x * gridDim.x;
    }
}





void cudaCallGillKernel(const int blocks,
    const int threadsPerBlock,
    float* dev_points, 
    float* dev_points_2, 
    float* state,
    float* X, 
    float* dev_timestep,
    float* dev_accu_time,
    const int N) {
    cudaGillKernel<<<blocks, threadsPerBlock>>>(dev_points, dev_points_2, state, X, dev_timestep, dev_accu_time, N);
}

void cudaCallFindMinKernel(const int blocks, 
    const int threadsPerBlock,
    float* dev_accu_time,
    float* dev_min_time,
    const int N) {
    cudaFindMinKernel<<<blocks, threadsPerBlock>>>(dev_accu_time, dev_min_time, N);
}


void cudaCallResampleKernel(const int blocks, 
    const int threadsPerBlock, 
    float* dev_resample_X, 
    float* dev_X, 
    float* dev_accu_time, 
    const int N, 
    const int T) {
    cudaResampleKernel<<<blocks, threadsPerBlock>>>(dev_resample_X, dev_X, dev_accu_time, N, T);
}














// gillespie.cc

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>

#include "gillespie_cuda.cuh"


using std::cerr;
using std::cout;
using std::endl;

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);            \
    return EXIT_FAILURE;}} while(0)



void check_args(int argc, char **argv){
    if (argc != 3){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "./gillespie <threads per block> <max number of blocks>\n";
        exit(EXIT_FAILURE);
    }
}


int main (int argc, char** argv) {

    check_args(argc, argv);
    const int threadsPerBlock = atoi(argv[1]);
    const int blocks = atoi(argv[2]);

    float* dev_points; // to determine the timestep
    float* dev_points_2; // to determine the reaction

    const int N = 100; // each iteration there is N simulations running

    cudaMalloc((void**)&dev_points, N * sizeof(float));
    cudaMalloc((void**)&dev_points_2, N * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    float* state;
    float* dev_concentration;
    float* dev_timestep;

    cudaMalloc((void**) &state, N * sizeof(float));
    cudaMalloc((void**) &dev_concentration, N * sizeof(float));
    cudaMalloc((void**) &dev_timestep, N * sizeof(float));

    cudaMemset(state, 0, N * sizeof(float));
    cudaMemset(dev_concentration, 0, N * sizeof(float));
    cudaMemset(dev_timestep, 0, N * sizeof(float));

    float* dev_accu_time;
    cudaMalloc((void**) &dev_accu_time, N * sizeof(float));
    cudaMemset(dev_accu_time, 0, N * sizeof(float));
    float* host_min_time = (float*)malloc(1 * sizeof(float));
    memset(host_min_time, 0, 1 * sizeof(float));

    float* dev_min_time;
    cudaMalloc((void**)&dev_min_time, 1 * sizeof(float));

    // std::vector< std::vector<float> > vector_X;
    // std::vector< std::vector<float> > vector_accu_time;


    // resampling the data in vectors
    const int T = 1000; // the total time interval after resampling

    // the matrix for resampled data
    float* resamp_X = (float*)malloc(N * T * sizeof(float));

    float* dev_resample_X;
    cudaMalloc((void**)&dev_resample_X, N * T * sizeof(float));
    cudaMemset(dev_resample_X, -1.0, N * T * sizeof(float));
    // the matrix to mark if a time point has been ipdated
    // int* is_resampled = new int[N * T]();
    // int* dev_is_resampled;
    // cudaMalloc((void**)&dev_is_resampled, N * T * sizeof(int));
    // cudaMemset(dev_is_resampled, 0, N * T * sizeof(int));

    const float final_time = 100;
    curandSetPseudoRandomGeneratorSeed(gen, 1234);
    cudaError err; 

    float* test = (float*)malloc(N * sizeof(float));
    float* test_accu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(test, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("before kernel, X: %f\n", test[0]);
    while (*host_min_time <= final_time) {
        curandGenerateUniform(gen, dev_points, N * sizeof(float));
        curandGenerateUniform(gen, dev_points_2, N * sizeof(float));
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "curand No kernel error detected" << endl;
        }
        cudaMemcpy(test, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("before kernel, X: %f\n", test[0]);
        // for each iteration, call a kernel
        // calculates state, X concentration, timestep, accumulate time
        cudaCallGillKernel(blocks, threadsPerBlock, dev_points, dev_points_2, state, dev_concentration, dev_timestep, dev_accu_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "gill No kernel error detected" << endl;
        }
        cudaMemcpy(test, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(test_accu, dev_accu_time, N * sizeof(float), cudaMemcpyDeviceToHost);

        printf("after kernel, X: %f\n", test[0]);
            // printf("accu time step: %f\n", test_accu[0]);
        // printf("Gill kernel called\n");

        // float* host_state = new float[N]();
        // float* host_X = new float[N](); // TODO destruct!!!!!!!!!!!!!!!
        // // float* host_timestep = new float[N]();
        // // cudaMemcpy(host_state, state, N * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(host_X, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
        // // cudaMemcpy(host_timestep, dev_timestep, N * sizeof(float), cudaMemcpyDeviceToHost);
        // float* host_accu_time = new float[N]();
        // cudaMemcpy(host_accu_time, dev_accu_time, N * sizeof(float), cudaMemcpyDeviceToHost);

        // cudaCallResampleKernel(blocks, threadsPerBlock, dev_resample_X, dev_concentration, dev_accu_time, N, T);
        
        // std::vector<float> v_X(std::begin(host_X), std::end(host_X)); // c++ 11
        // std::vector<float> v_accu_time(std::begin(host_accu_time), std::end(host_accu_time));

        // vector_X.push_back(v_X);
        // vector_accu_time.push_back(v_accu_time);

        // run a reduction kernel to find the minimum accumulate time       
        cudaCallFindMinKernel(blocks, threadsPerBlock, dev_accu_time, dev_min_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "resemple No kernel error detected" << endl;
        }
        gpuErrchk(cudaMemcpy(host_min_time, dev_min_time, 1 * sizeof(float), cudaMemcpyDeviceToHost));

        // float new_min = 99999;
        // for (int i = 0; i < N; ++i) {
        //     if (test_accu[i] < new_min) {
        //         new_min = test_accu[i];
        //     }
        // }
        // *host_min_time = new_min;

        printf("min get ");
        printf("this min: %f\n", *host_min_time);
        getchar();
    }
    free(test);
    // for (int i = 0; i < N; ++i) {
    //     printf("%f\n", test_accu[i]);
    // }
    free(test_accu);

    cudaMemcpy(resamp_X, dev_resample_X, N * T * sizeof(float), cudaMemcpyDeviceToHost);
    FILE *total_resample_file = fopen("resample.txt", "w");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < T; ++j) {
            fprintf(total_resample_file, "%f ", resamp_X[i * T + j]);
        }
        fprintf(total_resample_file, "\n");
    }


    fclose(total_resample_file);

    // find the mean and var
    float* mean = new float[T]();
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < N; ++j) {
            mean[i] += resamp_X[j * T + i];
        }
        mean[i] /= N;
    }





    FILE *outputFile = fopen("output.txt", "w");
    for (int i = 0; i < T; ++i) {
        fprintf(outputFile, "%f ",mean[i]);
    }
    fclose(outputFile);


    delete mean;
    free(host_min_time);
    cudaFree(state);
    cudaFree(dev_concentration);
    cudaFree(dev_points);
    cudaFree(dev_points_2);
    cudaFree(dev_timestep);
    cudaFree(dev_accu_time);
    cudaFree(dev_min_time);
    free(resamp_X);


    return EXIT_SUCCESS;
}

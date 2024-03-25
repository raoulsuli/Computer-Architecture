#include <stdio.h>
#include "utils/utils.h"

__global__ void kernel_parity_id(int *a, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N) {
        a[i] %= 2;
    }
}

__global__ void kernel_block_id(int *a, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        a[i] = blockIdx.x;
    }
}

__global__ void kernel_thread_id(int *a, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N) {
        a[i] = threadIdx.x;
    }
}

int main(void) {
    int nDevices;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&prop, i);

        printf("Device number: %d\nDevice name: %s\nTotal memory: %zu\nMemory Clock Rate (KHz): %d\nMemory Bus Width (bits): %d\n"
        , i, prop.name, prop.totalGlobalMem, prop.memoryClockRate, prop.memoryBusWidth);
    }

    cudaSetDevice(0);

    int *host_array = (int *)calloc(sizeof(int), 16);
    fill_array_int(host_array, 16);
    int *device_array;
    cudaMalloc(&device_array, 16 * sizeof(int));
    cudaMemcpy(device_array, host_array, 16 * sizeof(int), cudaMemcpyHostToDevice);
  
    kernel_parity_id<<<4, 4>>>(device_array, 16);
    cudaDeviceSynchronize();
    cudaMemcpy(host_array, device_array, 16 * sizeof(int), cudaMemcpyDeviceToHost);


    check_task_1(3, host_array);

    kernel_block_id<<<4, 4>>>(device_array, 16);
    cudaDeviceSynchronize();
    cudaMemcpy(host_array, device_array, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    check_task_1(4, host_array);

    kernel_thread_id<<<16 / 4, 4>>>(device_array, 16);
    cudaDeviceSynchronize();
    cudaMemcpy(host_array, device_array, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    check_task_1(5, host_array);

    free(host_array);
    cudaFree(device_array);

    return 0;
}
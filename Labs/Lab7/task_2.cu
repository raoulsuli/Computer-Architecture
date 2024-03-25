#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;
    cudaError_t err;

    float *host_array_a = 0;
    float *host_array_b = 0;
    float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    host_array_a = (float *)calloc(sizeof(float), N);
    host_array_b = (float *)calloc(sizeof(float), N);
    host_array_c = (float *)calloc(sizeof(float), N);

    DIE(host_array_a == NULL, "malloc(host_array_a)");
    DIE(host_array_b == NULL, "malloc(host_array_a)");
    DIE(host_array_c == NULL, "malloc(host_array_a)");

    err = cudaMalloc(&device_array_a, N * sizeof(float));
    DIE(err != cudaSuccess || device_array_a == NULL,
		"cudaMalloc(device_array_a)");

    err = cudaMalloc(&device_array_b, N * sizeof(float));
    DIE(err != cudaSuccess || device_array_a == NULL,
		"cudaMalloc(device_array_a)");

    err = cudaMalloc(&device_array_c, N * sizeof(float));
    DIE(err != cudaSuccess || device_array_a == NULL,
		"cudaMalloc(device_array_a)");

    fill_array_float(host_array_a, N);
    fill_array_random(host_array_b, N);

    err = cudaMemcpy(device_array_a, host_array_a, N * sizeof(float), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy(host_array_a)");

	err = cudaMemcpy(device_array_b, host_array_b, N * sizeof(float), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy(host_array_b)");

    const int block_size = 256;
    int num_blocks = N / block_size;

    if (N % block_size) num_blocks++;

    add_arrays<<<num_blocks, block_size>>>(device_array_a, device_array_b, device_array_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_array_c, device_array_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_task_2(host_array_a, host_array_b, host_array_c, N);

    free(host_array_a);
    free(host_array_b);
    free(host_array_c);

    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);
    return 0;
}
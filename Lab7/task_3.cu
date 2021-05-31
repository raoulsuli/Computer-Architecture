#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

#define BUF_2M		(2 * 1024 * 1024)
#define BUF_32M		(32 * 1024 * 1024)

int main(void) {
    cudaSetDevice(0);

    int *host_array_a = 0;
    int *host_array_b = 0;

    int *device_array_a = 0;
    int *device_array_b = 0;
    int *device_array_c = 0;

    host_array_a = (int *)calloc(sizeof(int), BUF_32M);
    host_array_b = (int *)calloc(sizeof(int), BUF_32M);

    cudaMalloc(&device_array_a, sizeof(int) * BUF_32M);
    cudaMalloc(&device_array_b, sizeof(int) * BUF_32M);
    cudaMalloc(&device_array_c, sizeof(int) * BUF_2M);

    // Check for allocation errors
    if (host_array_a == 0 || host_array_b == 0 || 
        device_array_a == 0 || device_array_b == 0 || 
        device_array_c == 0) {
        printf("[*] Error!\n");
        return 1;
    }

    for (int i = 0; i < BUF_32M; ++i) {
        host_array_a[i] = i % 32;
        host_array_b[i] = i % 2;
    }

    printf("Before swap:\n");
    printf("a[i]\tb[i]\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%d\n", host_array_a[i], host_array_b[i]);
    }

    cudaMemcpy(device_array_a, host_array_a, sizeof(int) * BUF_32M, cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_b, host_array_b, sizeof(int) * BUF_32M, cudaMemcpyHostToDevice);

    for (int i = 0; i < BUF_32M; i += BUF_2M) {
        cudaMemcpy(device_array_c, device_array_b + i, BUF_2M * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(device_array_b + i, device_array_a + i, BUF_2M * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(device_array_a + i, device_array_c, BUF_2M * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(host_array_a, device_array_a, sizeof(int) * BUF_32M, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_array_b, device_array_b, sizeof(int) * BUF_32M, cudaMemcpyDeviceToHost);

    printf("\nAfter swap:\n");
    printf("a[i]\tb[i]\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%d\n", host_array_a[i], host_array_b[i]);
    }

    free(host_array_a);
    free(host_array_b);

    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);
    return 0;
}
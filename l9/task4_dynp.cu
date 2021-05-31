#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEM    128

using namespace std;

__global__ void worker(int *data, int *result, int id)
{
    atomicAdd(&result[id], data[threadIdx.x]);
}

__global__ void master(int *data, int *result)
{
    worker<<<1, data[threadIdx.x]>>>(data, result, threadIdx.x);
}

void generateData(int *data, int num) {
    srand(time(0));
    
    for(int i = 0; i < num; i++) {
        data[i] = rand() % 8 + 2;
    }
}

void print(int *data, int num) {
    for(int i = 0; i < num; i++) {
        cout << data[i] << " ";
    }
    cout << endl;
}

// TASK check
// each element result[i] should be sum of first data[i] elements of data[i]
bool checkResult(int *data, int num, int *result) {

    for(int i = 0; i < num; i++) {
        
        int sum = 0;
        for(int j = 0; j < data[i]; j++) {
            sum += data[j];
        }
        
        if(result[i] != sum) {
            cout << "Error at " << i << ", requested sum of first " 
                << data[i] << " elem, got " << result[i] << endl;
            return false;
        }
    }
    
    return true;
}

int main(int argc, char *argv[])
{
    int *data = NULL;
    cudaMallocManaged(&data, NUM_ELEM * sizeof(int));

    int *result = NULL;
    cudaMallocManaged(&result, NUM_ELEM * sizeof(int));
    
    generateData(data, NUM_ELEM);
    
    master<<< 1, NUM_ELEM >>>(data, result);
    cudaDeviceSynchronize();
    
    print(data, NUM_ELEM);
    print(result, NUM_ELEM);
    
    if(checkResult(data, NUM_ELEM, result)) {
        cout << "Result OK" << endl;
    } else { 
        cout << "Result ERR" << endl;
    }
    
    cudaFree(data);
    cudaFree(result);

    return 0;
}
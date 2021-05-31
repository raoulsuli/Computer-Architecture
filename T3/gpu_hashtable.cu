#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

#define K_PRIME 46649
#define P_PRIME 84323297
#define BLOCK_SIZE 1024

__device__ int hash_function (int key, int capacity) {
	return abs(K_PRIME * key % P_PRIME) % capacity;
}

__global__ void get_batch(hash_pair *hashtable, int *device_keys, int numKeys, int *values, int capacity) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys || device_keys[i] == KEY_INVALID) return;

	int hash = hash_function(device_keys[i], capacity);

	while (true) {
		if (hashtable[hash].key == device_keys[i]) {
			values[i] = hashtable[hash].value;
			return;
		} 

		hash = (hash + 1) % capacity;
	}
}

__global__ void insert_batch(hash_pair *hashtable, int *device_keys, int *device_values, int *no_inserts, int numKeys, int capacity) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= numKeys || device_keys[i] == KEY_INVALID || device_values[i] == KEY_INVALID) return;

	int hash = hash_function(device_keys[i], capacity);
	int curr_key = 0;

	while (true) {
		curr_key = atomicCAS(&hashtable[hash].key, KEY_INVALID, device_keys[i]);
		if (curr_key == KEY_INVALID) {
			hashtable[hash].value = device_values[i];
			atomicAdd(no_inserts, 1);
			return;
		} else if (curr_key == device_keys[i]) {
			hashtable[hash].value = device_values[i];
			return;
		}
		hash = (hash + 1) % capacity;
	}
}

__global__ void reshape_hashtable(hash_pair *new_hashtable, hash_pair *hashtable, int capacity, int numBucketsReshape) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= capacity || hashtable[i].key == KEY_INVALID || hashtable[i].value == KEY_INVALID) return;

	int hash = hash_function(hashtable[i].key, numBucketsReshape);

	while (true) {
		if (!atomicCAS(&new_hashtable[hash].key, KEY_INVALID, hashtable[i].key)) {
			new_hashtable[hash].value = hashtable[i].value;
			return;
		}
		hash = (hash + 1) % numBucketsReshape;
	}
}

GpuHashTable::GpuHashTable(int size) {
	capacity = size;
	size = 0;

	glbGpuAllocator->_cudaMalloc((void **) &hashtable, capacity * sizeof(hash_pair));
	cudaCheckError();

	cudaMemset(hashtable, KEY_INVALID, capacity * sizeof(int));
	cudaCheckError();
}

GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(hashtable);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	hash_pair *new_hashtable;

	glbGpuAllocator->_cudaMalloc((void **) &new_hashtable, numBucketsReshape * sizeof(hash_pair));
	cudaCheckError();

	cudaMemset(new_hashtable, KEY_INVALID, numBucketsReshape * sizeof(int));
	cudaCheckError();
	
	int num_blocks = capacity / BLOCK_SIZE;
	if (capacity % BLOCK_SIZE) num_blocks++;

	reshape_hashtable<<<num_blocks, BLOCK_SIZE>>>(new_hashtable, hashtable, capacity, numBucketsReshape);

	cudaDeviceSynchronize();
	cudaCheckError();
	
	glbGpuAllocator->_cudaFree(hashtable);
	cudaCheckError();

	hashtable = new_hashtable;
	capacity = numBucketsReshape;
}

bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys;
	int *device_values;
	int *no_inserts;

	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaCheckError();
	glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaCheckError();
	glbGpuAllocator->_cudaMallocManaged((void **) &no_inserts, sizeof(int));
	cudaCheckError();

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	if ((load_factor() + numKeys / float(capacity)) >= 0.85f) reshape((size + numKeys) / 0.7f);

	int num_blocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) num_blocks++;

	insert_batch<<<num_blocks, BLOCK_SIZE>>>(hashtable, device_keys, device_values, no_inserts, numKeys, capacity);

	cudaDeviceSynchronize();
	cudaCheckError();

	size += *no_inserts;

	glbGpuAllocator->_cudaFree(device_keys);
	cudaCheckError();

	glbGpuAllocator->_cudaFree(device_values);
	cudaCheckError();

	glbGpuAllocator->_cudaFree(no_inserts);
	cudaCheckError();

	return true;
}

int *GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *values;

	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaCheckError();

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	glbGpuAllocator->_cudaMallocManaged((void **) &values, numKeys * sizeof(int));
	cudaCheckError();

	cudaMemset(values, KEY_INVALID, numKeys * sizeof(int));
	cudaCheckError();

	int num_blocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) num_blocks++;

	get_batch<<<num_blocks, BLOCK_SIZE>>>(hashtable, device_keys, numKeys, values, capacity);

	cudaDeviceSynchronize();
	cudaCheckError();

	glbGpuAllocator->_cudaFree(device_keys);
	cudaCheckError();

	return values;
}

float GpuHashTable::load_factor() {
	return (float) size / capacity;
}

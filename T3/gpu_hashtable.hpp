#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

typedef struct {
	int key;
	int value;
} hash_pair;

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	private:
		int size = 0;
		int capacity;
		hash_pair *hashtable;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		float load_factor();

		~GpuHashTable();
};

#endif


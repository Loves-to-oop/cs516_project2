
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

const int N = 2;

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

printf("i: %d = %d * %d + %d", i, blockIdx.x, blockDim.x, threadIdx.x);

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    

float A[N][N];
float B[N][N];
float C[N][N];

for(int i = 0; i <= N - 1; i ++){
	for(int j = 0; j <= N - 1; j ++)
	{
	
	A[i][j] = i * j;
	B[i][j] = i * j;
	//C[i][j] = i * j;

	
	}//end for j
}//end for i 
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    
}

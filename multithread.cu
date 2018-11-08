
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
#include <bits/stdc++.h>

/*
#include "cuPrintf.cu"`
 */


using namespace std;

inline void __cudaSafeCall( cudaError err,
		const char *file, const int line ) 

{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
	do
	{
		if ( cudaSuccess != err )
		{

			fprintf( stderr,
					"cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString( err ) );
			exit( -1 );

		}
	} while ( 0 );



#pragma warning( pop ) 
#endif
	// CUDA_CHECK_ERROR

	return;
}//end function

inline void __cudaCheckError( const char *file, const int line ) {
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
	do
	{

		cudaError_t err = cudaGetLastError();	
		if( cudaSuccess != err )
		{
			fprintf( stderr,
					"cudaCheckError() with sync failed at %s:%i : %s.\n", 
					file, line, cudaGetErrorString( err ) );
			exit( -1 );


		}

		err = cudaThreadSynchronize();
		if( cudaSuccess != err )
		{

			if ( cudaSuccess != err )
				fprintf( stderr,
						"cudaCheckError() failed at %s:%i : %s.\n",
						file, line, cudaGetErrorString( err ) );
			exit( -1 );

		}
	} while ( 0 );


	// More careful checking. However, this will affect performance. // Comment if not needed

#pragma warning( pop )
#endif // CUDA_CHECK_ERROR

	return;

}

void bubble_sort(int * array, int size)
{


	for(int i = 0; i <= size - 1; i ++)
	{

		for(int j = 1; j <= size - 1; j ++)
		{


			if(array[j] <  array[j - 1])
			{

				//printf("%d %d\n", array[j - 1], array[j]);

				int c = array[j - 1];

				array[j - 1] = array[j];

				array[j] = c;

				//printf("%d %d\n\n", array[j - 1], array[j]);

			}//end if




		}//end for j

	}//end for i


}//end function

void print_array(int * array, int size)
{



	for(int i = 0; i <= size - 1; i ++)
	{

		printf("%d, ", array[i]);

	}//end for i

	printf("\n");



}//end function


int * makeRandArray( const int size, const int seed ) {
	srand( seed );
	int * array = new int[ size ];
	for( int i = 0; i < size; i ++ ) {
		array[i] = std::rand() % 1000000;
	}
	return array; }


	/*

	   Kernel is fuction to run on GPU.

	 */

	__global__ void matavgKernel(int * array, int size, int blocks_on_a_side ) {

//printf("blockdim.x: %d\n", blockDim.x);


		//i is what number, j is what digit to sort, then sort based on digit..

		int i = threadIdx.x + blockDim.x * blockIdx.x;
		int j = threadIdx.y + blockDim.y * blockIdx.y;

int threads_on_a_side = (blockDim.x * blocks_on_a_side);

int current = i + (j * threads_on_a_side);

printf("%d = %d + (%d * %d)\n", current, i, j, threads_on_a_side);


	}//end function


void print_array_(int * host_array, int size)
{

		for(int i = 0; i <= size - 1; i ++)
		{

			printf("%d, ", host_array[i]);

		}//end for i

		printf("\n");

}//end function

int main( int argc, char* argv[] ) {
	int * array; // the poitner to the array of rands 
	int size, seed; // values for the size of the array 
	bool printSorted = false;
	// and the seed for generating
	// random numbers
	// check the command line args
	if( argc < 3 ){
		std::cerr << "usage: "
			<< argv[0]
			<< " [amount of random nums to generate] [seed value for rand]" << " [1 to print sorted array, 0 otherwise]"
			<< std::endl;
		exit( -1 ); }
	// convert cstrings to ints
	{
		std::stringstream ss1( argv[1] );
		ss1 >> size;
	} {
		std::stringstream ss1( argv[2] ); 
		ss1 >> seed; }
	/*
	   {
	   int sortPrint;
	   std::stringstream ss1( argv[2] ); 
	   ss1 >> sortPrint;
	   if( sortPrint == 1 )
	   printSorted = true;
	   }
	 */
	// get the random numbers

	array = makeRandArray( size, seed );

	int * host_array = (int*)malloc(size * 4);

	for(int i =0; i <= size - 1; i ++)
	{

		host_array[i] = array[i];

	}//end for i

	print_array(array, size);

	printf("host_array\n");

	print_array(host_array, size);

	cudaEvent_t startTotal, stopTotal; float timeTotal; cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal); cudaEventRecord( startTotal, 0 );

	/////////////////////////////////////////////////////////////////////
	///////////////////////  YOUR CODE HERE       ///////////////////////
	/////////////////////////////////////////////////////////////////////

	//curandState* devRandomGeneratorStateArray;
	//  cudaMalloc ( &devRandomGeneratorStateArray, 1*sizeof( curandState ) );

	//bubble_sort(array, size);

	//    thrust::host_vector<int> hostCounts(1,  0);
	//  thrust::device_vector<int> deviceCounts(hostCounts);

	int * cuda_array;

	cudaMalloc(&cuda_array, size * 4);

	cudaMemcpy(cuda_array, host_array, size * 4, cudaMemcpyHostToDevice);

	//matavgKernel <<< 1, 1 >>> (array, size); 

	int max = 0;

	for(int i = 0; i <= size - 1; i ++)
	{

	//https://stackoverflow.com/questions/35858264/c-finding-most-significant-bit-of-a-binary-number

		bitset<32> base2 = array[i];

		int j = 32;

		while(base2[j] != 1)
			j--;

		j++;

		printf("num: %d, most sig: %d\n", array[i], j);

		if(j > max)
		{

			max = j;

		}//end if

	}//end for i

	printf("max sig: %d\n", max);

/*

   This is saying the threadid within a small block, plus the current block
   in the x direction * the size of each block in the x direction to give
   the address in the larger x direction.

//threadIdx.x + (blockDim.x * blockIdx.x);
  
//divide the array into buckets based on thread number.



*/
	
//int numBlocks = 1;

int total_threads = (size / 10);

//if(total_threads == 0)
//	total_threads = 1;

int diameter = sqrt(total_threads) + 1;

printf("total threads: %d, diameter: %d\n", total_threads, diameter);

	int number_of_digits = 32;

	int threads_on_a_side = diameter / 2;

	printf("threads_on_a_side: %d\n", threads_on_a_side);

int blocks_on_a_side = (diameter / threads_on_a_side) + 1;

printf("blocks_on_a_side: %d\n", blocks_on_a_side);

int number_of_threads = pow(blocks_on_a_side * threads_on_a_side, 2);

printf("number of threads: %d\n", number_of_threads);

	dim3 threadsPerBlock(threads_on_a_side, threads_on_a_side);

	dim3 numBlocks(blocks_on_a_side, blocks_on_a_side);

	matavgKernel <<< numBlocks, threadsPerBlock >>> (cuda_array, size, blocks_on_a_side); 

	cudaMemcpy(host_array, cuda_array, size * 4, cudaMemcpyDeviceToHost);

	cudaFree(cuda_array);

	//https://stackoverflow.com/questions/6419700/way-to-verify-kernel-was-executed-in-cuda
	/*
	   cudaError_t err = cudaGetLastError();
	   if (err != cudaSuccess) 
	   printf("Error: %s\n", cudaGetErrorString(err));

	//thrust::reduce(deviceCounts.begin(), deviceCounts.end(), 0, thrust::plus<int>());;
	 */
	//matavgKerenel(array, size);

	/***********************************
	 *
	 Stop and destroy the cuda timer
	 **********************************/
	cudaEventRecord( stopTotal, 0 );
	cudaEventSynchronize( stopTotal );
	cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
	cudaEventDestroy( startTotal );
	cudaEventDestroy( stopTotal );
	/***********************************
	  end of cuda timer destruction
	 **********************************/
	std::cerr << "Total time in seconds: "
		<< timeTotal / 1000.0 << std::endl;
	printSorted = true;

	if( printSorted ){

print_array_(host_array, size);

		///////////////////////////////////////////////
		/// Your code to print the sorted array here //
		///////////////////////////////////////////////
	}//end if 


}//end main

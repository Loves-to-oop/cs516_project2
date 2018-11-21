
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

/*

   Working on using a 1D array to store the 2D buckets.
   Working in the kernel on calculating the start and finish
   in the 1D array to pass into the bubble sort function
   for each bucket.

Figure out why bubble sort is not working on subarrays.

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

__device__ void bubble_sort(int * array, int size, int start, int finish)
{

if((finish - start) == 1 && array[finish] < array[start])
{

printf("swap: %d, %d\n", array[start], array[finish]);

int d = array[start];

array[start] = array[finish];

array[finish] = d;

}//end if

	if((finish - start) > 1)
	{

printf("%d - %d > 1 \n", finish, start);

		for(int i = start; i <= finish; i ++)
		{

printf("i: %d\n", i);

			for(int j = start + 1; j <= finish; j ++)
			{


				if(array[j] <  array[j - 1])
				{

					printf("swap bubble: %d %d\n", array[j - 1], array[j]);

					int c = array[j - 1];

					array[j - 1] = array[j];

					array[j] = c;

					//printf("%d %d\n\n", array[j - 1], array[j]);

				}//end if




			}//end for j

		}//end for i

	}//end if

}//end function

void print_array(int * array, int size)
{



	for(int i = 0; i <= size - 1; i ++)
	{

		printf("%d, ", array[i]);

	}//end for i

	printf("\n");



}//end function


__device__ void print_array_device(int * array, int size)
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



	__global__ void matavgKernel(int * array, int size, int blocks_on_a_side, 
			int number_of_threads, int *array_of_buckets, int array_size, int * bucket_counts) {

		//printf("blockdim.x: %d\n", blockDim.x);


		//i is what number, j is what digit to sort, then sort based on digit..

		int i = threadIdx.x + blockDim.x * blockIdx.x;
		int j = threadIdx.y + blockDim.y * blockIdx.y;

		int threads_on_a_side = (blockDim.x * blocks_on_a_side);

		int current = i + (j * threads_on_a_side);

		printf("%d = %d + (%d * %d)\n", current, i, j, threads_on_a_side);

		printf("current bucket size: %d\n", bucket_counts[current]);

		int bucket = 0;

		int start = 0;

		int finish = 0;

		bool start_set = false;

		bool finish_set = false;

		for(int i = 0; i <= array_size - 1; i++)
		{


			//printf("%d, ", array_of_buckets[i]);

			if(array_of_buckets[i] == -1)
				bucket ++;

			if(bucket == current && start_set == false)
			{
				start = i;
				if(start != 0)
					start ++;

				start_set = true;
			}//end if

			if(bucket == current + 1 && finish_set == false)
			{

				finish = i - 1;

				finish_set = true;


			}//end if

		}//end for i

		//finish = finish - 2;

		printf("current: %d, start: %d, finish: %d\n", current, start, finish);

		if(start < finish)
		{
			if(current == 1)
			{printf("before: ");
				print_array_device(array_of_buckets, array_size);
			}
			bubble_sort(array_of_buckets, size, start, finish);
			if(current == 1)
			{printf("after: ");
				print_array_device(array_of_buckets, array_size);
			}}//end if

		//print_array_device(array_of_buckets, array_size);


		//print_array_device(array_of_buckets[current], bucket_counts[current]);

		//bubble_sort(array_of_buckets[current], bucket_counts[current]);



	}//end function


int find_max_significant_digit(int * array, int size)
{






	return 0;

}//end function


void print_array_(int * host_array, int size)
{

	for(int i = 0; i <= size - 1; i ++)
	{

		printf("%d, ", host_array[i]);

	}//end for i

	printf("\n");

}//end function

void unit_test()
{

int *array;

int size;

int seed;

int bucket;

int max_value;

int * host_array;

int ** array_of_buckets;

int *bucket_counts;

int *array_of_buckets_1D;

for(int z = 0; z <= 10 - 1; z ++)
{

printf("z: %d\n", z);

//int *array;

size = 10;

seed = 10;

	array = makeRandArray( size, seed );
printf("array created\n");

	host_array = (int*)malloc(size * 4);

	for(int i =0; i <= size - 1; i ++)
	{

		host_array[i] = array[i];

	}//end for i

	print_array(array, size);

	printf("host_array\n");

	print_array(host_array, size);

	cudaEvent_t startTotal, stopTotal; float timeTotal; cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal); cudaEventRecord( startTotal, 0 );

	int * cuda_array;

	cudaMalloc(&cuda_array, size * 4);

	cudaMemcpy(cuda_array, host_array, size * 4, cudaMemcpyHostToDevice);
	
	
	int total_threads = (size / 10);

if(total_threads > 48)
{

	total_threads = 48;

}//end if

	int diameter = sqrt(total_threads) + 1;

	printf("total threads: %d, diameter: %d\n", total_threads, diameter);

	int number_of_digits = 32;

	int threads_on_a_side = diameter / 2;

	printf("threads_on_a_side: %d\n", threads_on_a_side);

	int blocks_on_a_side = (diameter / threads_on_a_side) + 1;

	printf("blocks_on_a_side: %d\n", blocks_on_a_side);

	int number_of_threads = pow(blocks_on_a_side * threads_on_a_side, 2);
	int number_of_buckets = number_of_threads;

	printf("number of threads: %d, buckets: %d\n", number_of_threads, number_of_buckets);

	dim3 threadsPerBlock(threads_on_a_side, threads_on_a_side);

	dim3 numBlocks(blocks_on_a_side, blocks_on_a_side);
	
	array_of_buckets = new int*[number_of_buckets];

	bucket_counts = new int[number_of_buckets];


	int bucket_memory = 100000;

	for(int i = 0; i <= number_of_buckets - 1; i ++)
	{

		array_of_buckets[i] = new int[bucket_memory];


	}//end for i

	max_value = 0;


	for(int i = 0; i <= size - 1; i ++)
	{


		if(array[i] > max_value)
			max_value = array[i];


	}//end for i

	printf("max: %d\n", max_value);

	for(int i = 0; i <= size - 1; i ++)
	{

printf("i: %d\n", i);

printf("array[i]: %d\n", array[i]);

printf("bucket: %d\n", bucket);

printf("number of buckets: %d\n", number_of_buckets);

printf("value: %.2f\n", ((double)array[i] / (double)(max_value + 1)) * number_of_buckets);

		bucket = ((double)array[i] / (double)(max_value + 1)) * number_of_buckets;

		printf("array[i]: %d, bucket: %d, ", array[i], bucket);

		printf("array[i] / max_value: %f, ", (double)array[i] / (double)(max_value + 1)); 

		array_of_buckets[bucket][bucket_counts[bucket]] = array[i]; 

		printf("value_in_array: %d, ", array_of_buckets[bucket][bucket_counts[bucket]]);

		bucket_counts[bucket] ++;

		printf("bucket count: %d, %d\n", 
				bucket_counts[bucket], 
				array_of_buckets[bucket][bucket_counts[bucket] - 1]);


	}//end for i


	int * cuda_bucket_counts;

	cudaMalloc(&cuda_bucket_counts, number_of_buckets * 4);

	cudaMemcpy(cuda_bucket_counts, bucket_counts, number_of_buckets * 4, cudaMemcpyHostToDevice);

size_t array_of_buckets_1D_size = size * 2;

	array_of_buckets_1D = new int[array_of_buckets_1D_size];

	int iter = 0;

	for(int i = 0; i <= number_of_buckets - 1; i++)
	{

		for(int j = 0; j <= bucket_counts[i] - 1; j++)
		{


			array_of_buckets_1D[iter] = array_of_buckets[i][j];

			iter ++;

		}//end for j


		array_of_buckets_1D[iter] = -1;

		iter ++;


	}//end for i

	for(int i = 0; i <= iter - 1; i ++)
	{

		printf("%d, ", array_of_buckets_1D[i]);


	}//end for i


	int * cuda_array_of_buckets;


	cudaMalloc(&cuda_array_of_buckets, size * 2 * 4);

	cudaMemcpy(cuda_array_of_buckets, array_of_buckets_1D, size * 2 * 4
			, cudaMemcpyHostToDevice);

	matavgKernel <<< numBlocks, threadsPerBlock >>> 
		(cuda_array, size, blocks_on_a_side, 
		 number_of_threads, cuda_array_of_buckets, iter, cuda_bucket_counts); 

cudaMemcpy(array_of_buckets_1D, cuda_array_of_buckets, size * 2 * 4, cudaMemcpyDeviceToHost);

cudaFree(cuda_array_of_buckets);

cudaFree(cuda_bucket_counts);

printf("after sort(unit test):\n");

print_array(array_of_buckets_1D, iter);

	cudaMemcpy(host_array, cuda_array, size * 4, cudaMemcpyDeviceToHost);

	cudaFree(cuda_array);

int j = 0;

for(int i = 0; i <= iter - 1; i ++)
{

if(array_of_buckets_1D[i] != -1)
{

	host_array[j] = array_of_buckets_1D[i];

	j++;

}//end if

}//end for i

	cudaEventRecord( stopTotal, 0 );
	cudaEventSynchronize( stopTotal );
	cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
	cudaEventDestroy( startTotal );
	cudaEventDestroy( stopTotal );

		//print_array_(host_array, size);

printf("final array: ");

		print_array_(host_array, size);

free(array);

free(host_array);

fflush(stdout);

//free(array_of_buckets);

//free(bucket_counts);

free(array_of_buckets_1D);

		}//end for

//free(array);

//free(host_array);

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

//unit_test();

	array = makeRandArray( size, seed );

	int * host_array = (int*)malloc(size * 10);

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

	int * cuda_array;

	cudaMalloc(&cuda_array, size * 4);

	cudaMemcpy(cuda_array, host_array, size * 4, cudaMemcpyHostToDevice);

	int total_threads = (size / 10);

if(total_threads > 48)
total_threads = 48;

	int diameter = sqrt(total_threads) + 1;

	printf("total threads: %d, diameter: %d\n", total_threads, diameter);

	int number_of_digits = 32;

	int threads_on_a_side = diameter / 2;

	printf("threads_on_a_side: %d\n", threads_on_a_side);

	int blocks_on_a_side = (diameter / threads_on_a_side) + 1;

	printf("blocks_on_a_side: %d\n", blocks_on_a_side);

	int number_of_threads = pow(blocks_on_a_side * threads_on_a_side, 2);
	int number_of_buckets = number_of_threads;

	printf("number of threads: %d, buckets: %d\n", number_of_threads, number_of_buckets);

	dim3 threadsPerBlock(threads_on_a_side, threads_on_a_side);

	dim3 numBlocks(blocks_on_a_side, blocks_on_a_side);

	int ** array_of_buckets = new int*[number_of_buckets];

	int *bucket_counts = new int[number_of_buckets];


	int bucket_memory = 10000;

	for(int i = 0; i <= number_of_buckets - 1; i ++)
	{

		array_of_buckets[i] = new int[bucket_memory];


	}//end for i

	int max_value = 0;


	for(int i = 0; i <= size - 1; i ++)
	{


		if(array[i] > max_value)
			max_value = array[i];


	}//end for i

	printf("max: %d\n", max_value);

	for(int i = 0; i <= size - 1; i ++)
	{

		int bucket = ((double)array[i] / (double)(max_value + 1)) * number_of_buckets;

		printf("array[i]: %d, bucket: %d, ", array[i], bucket);

		printf("array[i] / max_value: %f, ", (double)array[i] / (double)(max_value + 1)); 

		array_of_buckets[bucket][bucket_counts[bucket]] = array[i]; 

		printf("value_in_array: %d, ", array_of_buckets[bucket][bucket_counts[bucket]]);

		bucket_counts[bucket] ++;

		printf("bucket count: %d, %d\n", 
				bucket_counts[bucket], 
				array_of_buckets[bucket][bucket_counts[bucket] - 1]);


	}//end for i


	int * cuda_bucket_counts;

	cudaMalloc(&cuda_bucket_counts, number_of_buckets * 4);

	cudaMemcpy(cuda_bucket_counts, bucket_counts, number_of_buckets * 4, cudaMemcpyHostToDevice);

size_t array_of_buckets_1D_size = size * 10;

	int * array_of_buckets_1D = new int[array_of_buckets_1D_size];

	int iter = 0;

	for(int i = 0; i <= number_of_buckets - 1; i++)
	{

		for(int j = 0; j <= bucket_counts[i] - 1; j++)
		{


			array_of_buckets_1D[iter] = array_of_buckets[i][j];

			iter ++;

		}//end for j


		array_of_buckets_1D[iter] = -1;

		iter ++;


	}//end for i

	for(int i = 0; i <= iter - 1; i ++)
	{

		printf("%d, ", array_of_buckets_1D[i]);


	}//end for i


	int * cuda_array_of_buckets;


	cudaMalloc(&cuda_array_of_buckets, array_of_buckets_1D_size);

	cudaMemcpy(cuda_array_of_buckets, array_of_buckets_1D, array_of_buckets_1D_size
			, cudaMemcpyHostToDevice);

	matavgKernel <<< numBlocks, threadsPerBlock >>> 
		(cuda_array, size, blocks_on_a_side, 
		 number_of_threads, cuda_array_of_buckets, iter, cuda_bucket_counts); 

cudaMemcpy(array_of_buckets_1D, cuda_array_of_buckets, array_of_buckets_1D_size, cudaMemcpyDeviceToHost);

cudaFree(cuda_array_of_buckets);

printf("after sort:\n");

print_array(array_of_buckets_1D, iter);

	cudaMemcpy(host_array, cuda_array, size * 4, cudaMemcpyDeviceToHost);

	cudaFree(cuda_array);

int j = 0;

for(int i = 0; i <= iter - 1; i ++)
{

if(array_of_buckets_1D[i] != -1)
{

	host_array[j] = array_of_buckets_1D[i];

	j++;

}//end if

}//end for i

	//https://stackoverflow.com/questions/6419700/way-to-verify-kernel-was-executed-in-cuda

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

for(int i = 1; i <= size - 1; i ++)
{

assert(host_array[i] > host_array[i - 1]); 

}//end for i

for(int i = 0; i <= size - 1; i ++)
{
	int missing_number = 1;

printf("checking: %d, ", array[i]);

	for(int j = 0; j <= size - 1; j ++)
	{
	
	if(array[i] == host_array[j])
	{
	
printf("FOUND\n");

	missing_number = 0;
	
	}//end if
	
	}//end for j

	assert(missing_number == 0);

}//end for i


	if( printSorted ){

		print_array_(host_array, size);

		///////////////////////////////////////////////
		/// Your code to print the sorted array here //
		///////////////////////////////////////////////
	}//end if 


}//end main

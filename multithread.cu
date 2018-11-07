
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



	__device__ void bubble_sort_cuda(int * array, int size ) {

		//array[0] = 5;
		for(int i = 0; i <= size - 1; i ++)
		{


			//cuPrintf(“Value is: %d\n”, i);

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


		//return array;

	}//end function

__global__ void matavgKernel(int * array, int size ) {


	bubble_sort_cuda(array, size);


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

	matavgKernel <<< 1, 1 >>> (cuda_array, size); 

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


		for(int i = 0; i <= size - 1; i ++)
		{

			printf("%d, ", host_array[i]);

		}//end for i

		printf("\n");

		///////////////////////////////////////////////
		/// Your code to print the sorted array here //
		///////////////////////////////////////////////
	} }

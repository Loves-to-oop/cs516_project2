
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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


int * makeRandArray( const int size, const int seed ) {
	srand( seed );
	int * array = new int[ size ];
	for( int i = 0; i < size; i ++ ) {
		array[i] = std::rand() % 1000000;
	}
	return array; }


	__global__ void matavgKernel( ) {
	}


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
	
	// get the random numbers
	array = makeRandArray( size, seed );

	cudaEvent_t startTotal, stopTotal; float timeTotal; cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal); cudaEventRecord( startTotal, 0 );

	/////////////////////////////////////////////////////////////////////
	///////////////////////  YOUR CODE HERE       ///////////////////////
	/////////////////////////////////////////////////////////////////////

	// bl bf switch between buffers
	// tabe filename create new tab tabn (tab next) tabp (tab previous)

	thrust::sort(array, array + size);

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
	
	if(argc == 3)
	{
	
	std::cerr << "Total time in seconds: "
		<< timeTotal / 1000.0 << std::endl;
	
	
	printSorted = true;
	
	if( printSorted ){

		///////////////////////////////////////////////
		/// Your code to print the sorted array here //
		///////////////////////////////////////////////
	}//end if 

}//end if

}//end main

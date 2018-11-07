all:
	nvcc thrust.cu -o thrust
	nvcc singlethread.cu -o singlethread
	nvcc multithread.cu -o multithread
	nvcc bucket_sort.cu -o bucket_sort


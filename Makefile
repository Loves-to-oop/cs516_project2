all:
	nvcc template.cu -o template
	nvcc thrust.cu -o thrust
	nvcc singlethread.cu -o singlethread
	nvcc multithread.cu -o multithread


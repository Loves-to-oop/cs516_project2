all:
	nvcc template.cu -o template
	nvcc thrust.cu -o thrust

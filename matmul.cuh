#ifndef MATMUL
#define MATMUL

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using namespace std;


void ReportError(bool cond, const char *message);


template <typename T>
__global__ void kernel_matmul(T* a, T* b, T* c, int sizeX, int sizeY);


template <typename T>
cudaError cudaMatmul(T* a, T* b, T* c, int sizeX, int sizeY);


#endif // !MATMUL


void ReportError(bool cond, const char *message) {
	if (cond) {
		cout << message << "\n";
	}
}

template <typename T>
__global__ void kernel_matmul(T* a, T* b, T* c, int sizeX, int sizeY) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < sizeX * sizeY) {
		int x = i / sizeY;
		int y = i % sizeY;
		for (int k = 0; k < sizeY; k++) {
			c[i] += a[x * sizeY + k] * b[k * sizeX + y];
		}
	}
}


template <typename T>
cudaError cudaMatmul(T* a, T* b, T* c, int sizeX, int sizeY) {
	T* dev_a;
	T* dev_b;
	T* dev_c;
	cudaError cudaStatus;

	cudaStatus = cudaSetDevice(0);
	ReportError(cudaStatus != cudaSuccess, "cudaSerDevice error!");

	cudaStatus = cudaMalloc((T**)&dev_a, sizeX * sizeY * sizeof(T));
	ReportError(cudaStatus != cudaSuccess, "cudaMalloc error for vector dev_a!");

	cudaStatus = cudaMalloc((T**)&dev_b, sizeX * sizeY * sizeof(T));
	ReportError(cudaStatus != cudaSuccess, "cudaMalloc error for vector dev_b!");

	cudaStatus = cudaMalloc((T**)&dev_c, sizeX * sizeY * sizeof(T));
	ReportError(cudaStatus != cudaSuccess, "cudaMalloc error for vector dev_c!");

	cudaStatus = cudaMemcpy(dev_a, a, sizeX * sizeY * sizeof(T), cudaMemcpyHostToDevice);
	ReportError(cudaStatus != cudaSuccess, "cudaMemcpy error for vector dev_a (a -> dev_a)!");

	cudaStatus = cudaMemcpy(dev_b, b, sizeX * sizeY * sizeof(T), cudaMemcpyHostToDevice);
	ReportError(cudaStatus != cudaSuccess, "cudaMemcpy error for vector dev_b (b -> dev_b)!");

	cudaStatus = cudaMemcpy(dev_c, c, sizeX * sizeY * sizeof(T), cudaMemcpyHostToDevice);
	ReportError(cudaStatus != cudaSuccess, "cudaMemcpy error for vector dev_c (c -> dev_c)!");

	int threads = 1024;
	int blocks = (((sizeX * sizeY) - 1) / threads) + 1;
	cout << "threads:" << threads << "\n";
	cout << "blocks:" << blocks << "\n";
	kernel_matmul << <blocks, threads >> > (dev_a, dev_b, dev_c, sizeX, sizeY);

	cudaStatus = cudaGetLastError();
	ReportError(cudaStatus != cudaSuccess, "kernel_matmul error!");

	cudaStatus = cudaMemcpy(c, dev_c, sizeX * sizeY * sizeof(T), cudaMemcpyDeviceToHost);
	ReportError(cudaStatus != cudaSuccess, "cudaMemcpy error (dev_c -> c)");

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return cudaStatus;
}


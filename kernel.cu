
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <iostream>
#include "matmul.cuh"
#include "queue.h"
#include <chrono>

const int sizeX = 100;
const int sizeY = 100;

template <typename T>
void showResultsCuda(T* a, T* b, T*c, int sizeX, int sizeY, bool showMatrices) {
	auto start = chrono::high_resolution_clock::now();
	cudaError cudaStatus = cudaMatmul(a, b, c, sizeX, sizeY);
	ReportError(cudaStatus != cudaSuccess, "Error of running cudaMatmul!");
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = end - start;
	if (showMatrices) {
		cout << "matrix a:" << "\n\n";
		showMatrix(a, sizeX, sizeY);
		cout << "matrix b:" << "\n\n";
		showMatrix(b, sizeX, sizeY);
		cout << "\n\n" << "cuda time of calculating: " << duration.count() << "\n";
		cout << "matrix c_cuda:" << "\n\n";
		showMatrix(c, sizeX, sizeY);
	}
	else {
		cout << "\n\n" << "cuda time of calculating: " << duration.count() << "\n";
	}
}

int main() {
	double mean = 0;
	double sd = 1;
	double* a = new double[sizeX * sizeY];
	double* b = new double[sizeX * sizeY];
	double* c_cuda = new double[sizeX * sizeY];
	double* c_queue = new double[sizeX * sizeY];
	for (int i = 0; i < sizeX; i++) {
		for (int j = 0; j < sizeY; j++) {
			a[i * sizeY + j] = random(mean, sd);
			b[i * sizeY + j] = random(mean, sd);
			c_cuda[i * sizeY + j] = 0;
			c_queue[i * sizeY + j] = 0;
		}
	}
	// code for parallel calculating matrix multiplication ... 
	showResultsCuda(a, b, c_cuda, sizeX, sizeY, false);
	queueMatmul(a, b, c_queue, sizeX, sizeY, false);
	if (checker(c_cuda, c_queue, sizeX, sizeY)) {
		cout << "\n" << "c_cuda equal c_queue" << "\n";
	}
	else {
		cout << "\n" << "c_cuda not equal c_queue" << "\n";
	}
	return 0;
}
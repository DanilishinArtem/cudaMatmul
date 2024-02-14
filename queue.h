#pragma once
#ifndef QUEUE
#define QUEUE
#include <chrono>
#include <random>
#include <iostream>
using namespace std;

double random(double mean, double sd);


template <typename T>
void showMatrix(T* matrix, int sizeX, int sizeY) {
	for (int i = 0; i < sizeX; i++) {
		for (int j = 0; j < sizeY; j++) {
			cout << matrix[i * sizeY + j] << "  ";
		}
		cout << "\n";
	}
}


template <typename T>
void queueMatmul(T* a, T* b, T* c, int sizeX, int sizeY, bool showMatrix_info) {
	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < sizeY; i++) {
		for (int j = 0; j < sizeX; j++) {
			for (int k = 0; k < sizeY; k++) {
				c[i * sizeY + j] += a[i * sizeY + k] * b[k * sizeY + j];
			}
		}
	}
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = end - start;
	cout << "\n\n" << "queue time of calculating: " << duration.count() << "\n";
	if (showMatrix_info) {
		cout << "matrix c_queue:" << "\n\n";
		showMatrix(c, sizeX, sizeY);
	}
}


template <typename T>
bool checker(T* c_cuda, T* c_queue, int sizeX, int sizeY) {
	for (int i{ 0 }; i < sizeY; i++) {
		for (int j{ 0 }; j < sizeX; j++) {
			if (c_cuda[i * sizeY + j] != c_queue[i * sizeY + j]) {
				return false;
			}
		}
	}
}

#endif 
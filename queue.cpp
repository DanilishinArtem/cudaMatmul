#include "queue.h"


double random(double mean, double sd) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dist(mean, sd);
	return dist(gen);
}
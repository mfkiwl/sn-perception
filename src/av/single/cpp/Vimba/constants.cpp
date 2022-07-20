#include "constants.h"

// global variables
int NUM_FRAMES = 500;
double SIMILARITY = 0.0;

// Accuracy parameters:
double minArea = 700.0;
double rratio = 1.05;

void Set_NUM_FRAMES(int value)
{
	NUM_FRAMES = value;
}

void Set_SIMILARITY(double value)
{
	SIMILARITY = value;
}

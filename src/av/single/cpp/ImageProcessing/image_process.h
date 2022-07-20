#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core_c.h>

using namespace std;

string GetMatType(const cv::Mat& mat);

string GetMatDepth(const cv::Mat& mat);

/**********************************************/
// Basic functions to open and display Images:
/*********************************************/

void open_image(cv::Mat mat, string name);

void open_images(cv::Mat mat1, cv::Mat mat2, string name);

/*********************************************************/
// Functions to find centers and centroids. 
/*********************************************************/

cv::Point find_center(cv::Mat& img);

cv::Point find_centroid(cv::Mat& img);

/*****************************************************************************/
// Functions to operate and handle contours of an image
// Contours are found with OpenCV findcontours function, 
//then processed with the helper functions below
/*****************************************************************************/

int getMaxAreaContourId(vector <vector<cv::Point>> contours);

double getMaxContoursArea(vector<vector<cv::Point>> contours);

bool QuickContourCheck(cv::Mat Mask, double smallest_area);

void drawBiggestContour(cv::Mat& Mask, cv::Mat Image, string name);

void drawAllContours(cv::Mat& Mask, string name);

/***************************************************************************************/
// Functions to create and handle Masks. The mask can be easily found using the AbsDiff
// function from OpenCV. Then use the functions below to filter, process, and handle it.
/***************************************************************************************/

void threshold_3C(cv::Mat& foregroundMask, cv::Mat& diffImage, double threshold);

void compareMasks(cv::Mat mask1, cv::Mat mask2, cv::Mat& mask);

double findMax(cv::Mat diffImage);

double getSimilarity(const cv::Mat A, const cv::Mat B);

double findHistSimilarity(cv::Mat A, cv::Mat B);

/****************************************************************************************/
// Main Image processing functions. They use a combination of the functions above to
// calibrate and detect objects.
/****************************************************************************************/

void calibrate(cv::Mat image1, cv::Mat image2);

void check_scene(cv::Mat image1, cv::Mat image2);


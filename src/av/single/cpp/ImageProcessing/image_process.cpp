#include "image_process.h"
#include "../Vimba/constants.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

extern double rratio;
extern double minArea;
extern double SIMILARITY;

string GetMatDepth(const cv::Mat& mat)
{
	const int depth = mat.depth();

	switch (depth)
	{
	case CV_8U:  return "CV_8U";
	case CV_8S:  return "CV_8S";
	case CV_16U: return "CV_16U";
	case CV_16S: return "CV_16S";
	case CV_32S: return "CV_32S";
	case CV_32F: return "CV_32F";
	case CV_64F: return "CV_64F";
	default:
		return "Invalid depth type of matrix!";
	}
}

string GetMatType(const cv::Mat& mat)
{
	const int mtype = mat.type();

	switch (mtype)
	{
	case CV_8UC1:  return "CV_8UC1";
	case CV_8UC2:  return "CV_8UC2";
	case CV_8UC3:  return "CV_8UC3";
	case CV_8UC4:  return "CV_8UC4";

	case CV_8SC1:  return "CV_8SC1";
	case CV_8SC2:  return "CV_8SC2";
	case CV_8SC3:  return "CV_8SC3";
	case CV_8SC4:  return "CV_8SC4";

	case CV_16UC1: return "CV_16UC1";
	case CV_16UC2: return "CV_16UC2";
	case CV_16UC3: return "CV_16UC3";
	case CV_16UC4: return "CV_16UC4";

	case CV_16SC1: return "CV_16SC1";
	case CV_16SC2: return "CV_16SC2";
	case CV_16SC3: return "CV_16SC3";
	case CV_16SC4: return "CV_16SC4";

	case CV_32SC1: return "CV_32SC1";
	case CV_32SC2: return "CV_32SC2";
	case CV_32SC3: return "CV_32SC3";
	case CV_32SC4: return "CV_32SC4";

	case CV_32FC1: return "CV_32FC1";
	case CV_32FC2: return "CV_32FC2";
	case CV_32FC3: return "CV_32FC3";
	case CV_32FC4: return "CV_32FC4";

	case CV_64FC1: return "CV_64FC1";
	case CV_64FC2: return "CV_64FC2";
	case CV_64FC3: return "CV_64FC3";
	case CV_64FC4: return "CV_64FC4";

	default:
		return "Invalid type of matrix!";
	}
}

/**********************************************/
// Basic functions to open and display Images:
/*********************************************/


// opens one image and waits for keystroke to continue
void open_image(Mat mat, string name)
{
	namedWindow(name, WINDOW_NORMAL);

	imshow(name, mat);
	waitKey();
	destroyWindow(name);
}

// open two images in parallel and waits for keystroke
void open_images(Mat mat1, Mat mat2, string name)
{
	Mat mat;
	namedWindow(name, WINDOW_NORMAL);
	hconcat(mat1, mat2, mat);
	imshow(name, mat);
	waitKey();
	destroyWindow(name);
}

/*********************************************************/
// Functions to find centers and centroids. 
/*********************************************************/

// Finds the center of any image (does not matter size or content)
Point find_center(Mat& img)
{
	Mat blank = Mat::ones(img.rows, img.cols, CV_8UC1);
	Moments m = moments(blank, true);
	Point center(m.m10 / m.m00, m.m01 / m.m00);
	return center;
}

// Finds the centroid of the image. It can be used to find the centroid of an object if
// a mask of this one is passed.
Point find_centroid(Mat& img)
{
	Moments m = moments(img, true);
	Point centroid(m.m10 / m.m00, m.m01 / m.m00);
	return centroid;
}

/*****************************************************************************/
// Functions to operate and handle contours of an image
// Contours are found with OpenCV findcontours function, 
//then processed with the helper functions below
/*****************************************************************************/

// From a set of contours it finds the ID of the biggest one
int getMaxAreaContourId(vector <vector<cv::Point>> contours)
{
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = contourArea(contours.at(j));
		//cout << newArea << endl;
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		}
	}
	return maxAreaContourId;
}

// From a set of contours it finds the biggest the area
double getMaxContoursArea(vector<vector<cv::Point>> contours)
{
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = contourArea(contours.at(j));
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		}
	}
	return maxArea;
}

// This function does a quick check to make sure if any contour is bigger than a minimun area threshold
bool QuickContourCheck(Mat Mask, double smallest_area)
{
	// find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	findContours(Mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	Mat drawing = Mat::zeros(Mask.size(), CV_8UC3);

	// get Biggest countour area
	double area = getMaxContoursArea(contours);
	cout << "Biggest area: " << area << endl;
	return (area > smallest_area);
}

// It finds all contours of an image or mask and draws the biggest one
void drawBiggestContour(Mat& Mask, Mat Image, string name)
{
	//dilate
	Mat dilate_out;
	dilate(Mask, dilate_out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	//open_image(dilate_out, name + "-dilate");

	// find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	// Morphological operations to bound the object as much as possible
	morphologyEx(dilate_out, dilate_out, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9)));
	findContours(dilate_out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	// no significant contours found
	if (contours.empty()) return;

	// Draw bounding rectangle
	Rect rect = boundingRect(contours[getMaxAreaContourId(contours)]);

	//check area size
	if (getMaxContoursArea(contours) > minArea)
	{
		// draw biggest contour
		//Mat drawing = Mat::zeros(dilate_out.size(), CV_8UC3);
		rectangle(Image, rect, Scalar((0, 255, 255)));
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(Image, contours, getMaxAreaContourId(contours), color, 2, LINE_8, hierarchy, 0);
		imshow("feed", Image);
		waitKey(1);
		//open_image(Image, name + "-biggest-contour");
	}
}

// Finds and draws all contours of an image or mask. It uses dilate and erode to fill the gaps and 
// output the most accurate object representation.
void drawAllContours(Mat& Mask, string name)
{
	//canny edge detection to find contours
	Mat canny_out;
	Canny(Mask, canny_out, 10, 30, 3);
	//open_image(canny_out, name + "-canny");

	//dilate
	Mat dilate_out;
	dilate(canny_out, dilate_out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	//open_image(dilate_out, name + "-dilate");

	// find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	findContours(dilate_out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	Mat drawing = Mat::zeros(canny_out.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	open_image(drawing, name + "-contours");
}



/***************************************************************************************/
// Functions to create and handle Masks. The mask can be easily found using the AbsDiff
// function from OpenCV. Then use the functions below to filter, process, and handle it.
/***************************************************************************************/

/* This function applies a binary threshold to a mask. This means that every pixel with
a higher value than the threshold will be set to 255 (white), and everything lower
will be left as it is. Normally foregroundMask is a mask of zeros and diffImage is the
image that you want to apply the threshold to.

The pixel value is calculated finding the distance of each channel value. If you need
to do this for a 1 channel image, just use the threshold openCV function.
*/
void threshold_3C(Mat& foregroundMask, Mat& diffImage, double threshold)
{
	double dist;
	for (int j = 0; j < diffImage.rows; j++)
		for (int i = 0; i < diffImage.cols; i++)
		{
			Vec3b pix = diffImage.at<Vec3b>(j, i);

			dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
			dist = sqrt(dist);

			if (dist > threshold)
			{
				foregroundMask.at<unsigned char>(j, i) = 255;
			}
		}
	return;
}

/*Very useful function to filter a mask using a background one. The way it works is that if
mask1 is set 255 and mask2 is 0, then the output mask will be set to 255.

Parameters:
- Mask1: mask to be filtered
- Mask2: filter calculated during calibration
- Mask: output mask, it should be passed in as a mask of zeros to work best.
*/
void compareMasks(Mat mask1, Mat mask2, Mat& mask)
{
	for (int j = 0; j < mask1.rows; j++)
		for (int i = 0; i < mask1.cols; i++)
		{
			unsigned char value1 = mask1.at<unsigned char>(j, i);
			unsigned char value2 = mask2.at<unsigned char>(j, i);
			if (value1 == 255 && value2 == 0)
			{
				mask.at<unsigned char>(j, i) = 255;
			}
		}
}

// Finds max value in a Mat (does not need to be a mask or image, since it gets reshaped.
double findMax(Mat diffImage)
{
	Mat temp = diffImage;
	temp.reshape(1);
	double minVal;
	double maxVal;
	minMaxIdx(temp, &minVal, &maxVal);
	return maxVal;
}

/*Computes the similarity between two masks. And returns a double value.

It is done by calculate the norm between both images. There are different norms
that can be used for this purpose. I personally find CV_L1 to yield the best results,
although in the past I have also used CV_L2.
*/
double getSimilarity(const Mat A, const Mat B)
{
	if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
		// Calculate the L2 relative error between images.
		double errorL2 = norm(A, B, CV_L1);
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / (double)(A.rows * A.cols);
		return similarity;
	}
	else {
		//Images have a different size
		return 100000000.0;  // Return a bad value
	}
}

// Calculate the histogram of two images, normalizes it, and calculates the similarity between them. 
// for greyscale images I find is not accurate enough to be relied on. However, for HSV or RGB should
// work better.
double findHistSimilarity(Mat A, Mat B)
{
	double similarity;
	int channels[] = { 0 };
	int bins = 256;
	int histSize[] = { 256 };
	float lranges[] = { 0,256 };
	const float* ranges[] = { lranges };
	Mat histA;
	Mat histB;
	calcHist(&A, 1, channels, Mat(), histA, 1, histSize, ranges, true, false);
	normalize(histA, histA, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&B, 1, channels, Mat(), histB, 1, histSize, ranges, true, false);
	normalize(histB, histB, 0, 1, NORM_MINMAX, -1, Mat());
	similarity = compareHist(histA, histB, 1);
	cout << "Hist Similarity: " << similarity << endl;
	return similarity;
}

/***************************************************************************************/
//								Main Functions
/***************************************************************************************/


// Compares similarities between images and stores the biggest value
void calibrate(Mat image1, Mat image2)
{
	double similarity = getSimilarity(image1, image2);
	if (similarity > SIMILARITY)
	{
		cout << "Old similarity was: " << SIMILARITY << endl;
		cout << "Similarity update to: " << similarity << endl;
		Set_SIMILARITY(similarity);
	}
}

// Checks the scene for new objects and moving ones
void check_scene(Mat image1, Mat image2)
{
	//First we create a copy of images since mats are basically pointers, and we do not want to modify image2 as it will be used for the next iteration
	Mat current;
	image2.copyTo(current);
	// Quick similarity check to avoid any further processing if it is obvious that no new object has came in. This is very fast
	double similarity = getSimilarity(image1, image2);
	cout << "Similarity check: " << similarity << endl;
	if (similarity <= rratio*SIMILARITY) return; //
	// If similarity was not good enough to determine whether there was an object
	// or not, we need to find contours and check their sizes.
	// First find the absdifference between both images
	Mat diff;
	absdiff(image1, image2, diff);
	// Do a blur to reduce noise (depends on the camera and the quality of the image)
	GaussianBlur(diff, diff, Size(5, 5), 0);
	// Create a mask of the difference between images to remove all noise and isolate the new object/s detected
	Mat thresh;
	threshold(diff, thresh, 10, 255, THRESH_BINARY);
	//From the mask, find the biggest countour and bound it. If you are confident with the mask results you could just draw all contours to get a better
	// idea of the object. For a first approach, I decided to just draw the biggest contour to show functionality, but this could be improved
	drawBiggestContour(thresh,current, "Object");
}
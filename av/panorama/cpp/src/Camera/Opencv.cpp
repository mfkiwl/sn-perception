#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>


// Linter setup (Clang-Tidy): https://stackoverflow.com/questions/67946896/vs-code-c-linter
// TODO: Check what to use from here
// TODO: Adding packages for intellisense info
// TODO: Download - https://www.oreilly.com/library/view/effective-modern-c/9781491908419/

// argc is the number of arguments, the exe itself is 1, others are 2, 3 etc.

// The flags and the correspondings meanings: from OpenCV imgcodecs.hpp (imread: -1)
// https://stackoverflow.com/questions/56063512/what-does-flag-1-in-imread-of-opencv-mean

// OpenCV C++ basics: https://subscription.packtpub.com/book/application-development/9781786469717/1/ch01lvl1sec11/exploring-the-cv-mat-data-structure
// return cv::Mat{}; can be return {}; : https://stackoverflow.com/questions/39487065/what-does-return-statement-mean-in-c11
// error returning when void: https://stackoverflow.com/questions/20943380/how-to-handle-error-conditions-in-a-void-function

// go trough cv::Mat data structure - put it into fireship + algoexpert data structure thoughts + Tensor data structure notes (+ PyTorch documentation)
// go trough basic loop writing (size +1 -1 etc. - also python)

// what libraries does the c++ algo use to replace numpy (cylindrical panorama)
// CHECK ROS Publisher!
// automate arguments for testing

cv::Mat readImage(const std::string& img_path) {
    cv::Mat img;
    img = cv::imread(img_path, -1);
    if (!img.data) {
        std::cout << "[ERROR] No image data! \n";
        return cv::Mat{}; //can be {} 
    }
    return img;
}

void showImage(const std::string& win_name, const cv::Mat& img) {
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(win_name, img);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "[ERROR] Proper usage: ./executable <window_name> <image_path> \n";
        return 1;
    }
    std::string win_name = argv[1];
    std::string img_path = argv[2];
    
    cv::Mat img = readImage(img_path);
    showImage(win_name, img);
}

#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// https://github.com/KEDIARAHUL135/PanoramaStitchingP2/blob/master/main.cpp


class PanoramaStitcher {
  public:
    PanoramaStitcher();
    
    int readImages(std::string& imgDirPath, std::vector<cv::Mat>& imgs);

    
  private:
    // two private variables could be created to store the state (xt, yt)
    std::tuple<int, int>& unrollXY(int x, int y, int center);

    // Calculates cylinder projection values and saves them into class variables. 
    // Called in the constructor.
    void calculateCylinderValues();

    cv::Mat ProjectOntoCylinder(cv::Mat imgInit);



};


int PanoramaStitcher::readImages(std::string& imgDirPath, std::vector<cv::Mat>& imgs) { 

  if (fs::is_directory(fs::status(imgDirPath))) {
    
  }

};


int main() {

    std::vector<cv::Mat> imgs;

    std::cout << "[INFO] Images read!" << std::endl;

}


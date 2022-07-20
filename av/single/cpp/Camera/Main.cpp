#include <iostream>
#include <sstream>
#include <string>

#include <VimbaCPP.h>
// #include <AsynchronousGrab/Qt/Source/ApiController.h>
#include "../Vimba/FrameObserver.h"
#include "../Vimba/ApiController.h"
#include "../Vimba/constants.h"
#include "../ImageProcessing/image_process.h"
#include "Common/StreamSystemInfo.h"
#include "Common/ErrorCodeToMessage.h"

// usleep timer - uses microseconds
#include <unistd.h>

// #include <opencv2/opencv.hpp>

using namespace std;
using namespace AVT;
using namespace AVT::VmbAPI;
using namespace AVT::VmbAPI::Examples;

// global constants
// const int NUM_FRAMES = 500;
// const double SIMILARITY = 0.0;
extern int NUM_FRAMES;
extern double SIMILARITY;

void PrintCamInfo(const CameraPtr& camera)
{
  std::string strID;
  std::string strName;
  std::string strModelName;
  std::string strSerialNumber;
  std::string strInterfaceID;
  
  std::ostringstream ErrorStream;
  
  VmbErrorType err = camera->GetID(strID);
  if (VmbErrorSuccess != err)
  {
      ErrorStream << "[Could not get camera ID. Error code: " << err << "(" << ErrorCodeToMessage(err) << ")" << "]";
      strID = ErrorStream.str();
  }
  
  err = camera->GetName(strName);
  if (VmbErrorSuccess != err)
  {
      ErrorStream << "[Could not get camera name. Error code: " << err << "(" << ErrorCodeToMessage(err) << ")" << "]";
      strName = ErrorStream.str();
  }
  
  err = camera->GetModel(strModelName);
  if (VmbErrorSuccess != err)
  {
      ErrorStream << "[Could not get camera mode name. Error code: " << err << "(" << ErrorCodeToMessage(err) << ")" << "]";
      strModelName = ErrorStream.str();
  }
  
  err = camera->GetSerialNumber(strSerialNumber);
  if (VmbErrorSuccess != err)
  {
      ErrorStream << "[Could not get camera serial number. Error code: " << err << "(" << ErrorCodeToMessage(err) << ")" << "]";
      strSerialNumber = ErrorStream.str();
  }
  
  err = camera->GetInterfaceID(strInterfaceID);
  if (VmbErrorSuccess != err)
  {
      ErrorStream << "[Could not get interface ID. Error code: " << err << "(" << ErrorCodeToMessage(err) << ")" << "]";
      strInterfaceID = ErrorStream.str();
  }
  
  std::cout << "/// Camera Name    : " << strName << "\n"
      << "/// Model Name     : " << strModelName << "\n"
      << "/// Camera ID      : " << strID << "\n"
      << "/// Serial Number  : " << strSerialNumber << "\n"
      << "/// @ Interface ID : " << strInterfaceID << "\n\n";
}


inline VmbErrorType SetValueIntMod2(const CameraPtr& camera, const std::string& featureName, VmbInt64_t& storage)
{
	VmbErrorType    res;
	FeaturePtr      pFeature;
	res = SP_ACCESS(camera)->GetFeatureByName(featureName.c_str(), pFeature);
	if (VmbErrorSuccess == res)
	{
		VmbInt64_t minValue, maxValue;
		res = SP_ACCESS(pFeature)->GetRange(minValue, maxValue);
		if (VmbErrorSuccess == res)
		{
			maxValue = (maxValue >> 1) << 1; // mod 2 dividable
			res = SP_ACCESS(pFeature)->SetValue(maxValue);
			if (VmbErrorSuccess == res)
			{
				storage = maxValue;
			}
		}
	}
	return res;
}


int main() {

  VmbInt64_t nPLS; // Payload size
  FeaturePtr pFeature; // Feature pointer used to control and execute camera properties
  VmbErrorType err = VmbErrorSuccess;
  VimbaSystem& system = VimbaSystem::GetInstance();
  CameraPtrVector cams;
  CameraPtr cam;
  ApiController API;
  FramePtrVector frames(NUM_FRAMES); // Frame array
  IFrameObserverPtr pFrameObserver;

  // version check
  std::cout << "Vimba Version: " << system << std::endl;
  std::cout << NUM_FRAMES << std::endl;

  // Start the Vimba Engine and API
  err = system.Startup();
  // Check for errors
  if (VmbErrorSuccess != err)
  {
      std::cout << ErrorCodeToMessage(err) << std::endl;
      std::cout << "Something went wrong starting the Vimba engine and API, exiting..." << std::endl;
      system.Shutdown();
      return 0;
  }
  
  // Check for cameras
  std::cout << "\nTrying to find a camera..." << std::endl;
  err = system.GetCameras(cams);
  if (VmbErrorSuccess != err)
  {
      std::cout << ErrorCodeToMessage(err) << std::endl;
      std::cout << "Something went wrong looking for the camera(s), exiting..." << std::endl;
      system.Shutdown();
      return 0;
  }
  
  else if (cams.size() < 1)
  {
      std::cout << "No cameras found, ending the program." << std::endl;
      system.Shutdown();
      return 0;
  }
  
  
  // If successful we connect to the camera
  cam = cams[0];
  std::cout << cams.size() << std::endl;
  PrintCamInfo(cam); // print the model information
  err = cam.get()->Open(VmbAccessModeFull); // connect to the camera
  
  if (VmbErrorSuccess != err)
  {
      std::cout << ErrorCodeToMessage(err) << std::endl;
      std::cout << "Something went wrong connecting to the camera/s, exiting..." << endl;
      system.Shutdown();
      return 0;
  }

  // Next we get the image size for the buffer
  // Then we need to allocate the memory for the frames, and register them to the camera
  cam.get()->GetFeatureByName("PayloadSize", pFeature);
  pFeature.get()->GetValue(nPLS);
  SP_SET(pFrameObserver, new FrameObserver(cam));
  for (FramePtrVector::iterator iter = frames.begin(); frames.end() != iter; ++iter)
  {
      (*iter).reset(new Frame(nPLS));
      (*iter)->RegisterObserver(pFrameObserver);
      cam.get()->AnnounceFrame(*iter);
  }

  // Start the Capture Engine to queue the frames
  err = cam.get()->StartCapture();
  for (FramePtrVector::iterator iter = frames.begin(); frames.end() != iter; ++iter)
  {
      // Put the frame in the queue
      cam.get()->QueueFrame(*iter);
  }

  // Creat OpenCV matrix (allocate array data)
  cv::Mat image;
  VmbInt64_t width;
  VmbInt64_t height;
  
  err = SetValueIntMod2(cam, "Width", width);
  if (err == VmbErrorSuccess)
  {
      std::cout << "[INFO] width set: " << width << std::endl;
      err = SetValueIntMod2(cam, "Height", height);
      if (err == VmbErrorSuccess)
      {
          std::cout << "[INFO] height set: " << height << std::endl;
          image.create(height, width, CV_8UC1);
      }
  }

  // Now we are ready to start acquiring images
  // Start acquisition engine
  cam.get()->GetFeatureByName("AcquisitionStart", pFeature);
  err = pFeature.get()->RunCommand();

  // runtime code here
  usleep(250000);

  // Find out pixel values
  //VmbPixelFormatType pixel_format;
  
  std::cout << "[INFO - PIXEL FORMAT] The pixel format is: " << API.GetPixelFormat() << std::endl;
  
  
 FeaturePtr pixelFeature;


  err = cam->GetFeatureByName("PixelFormat", pixelFeature);
  if (VmbErrorSuccess == err) {
    std::cout << "[INFO] Pixel format feature captured! " << std::endl; 
    err = pixelFeature->SetValue(VmbPixelFormatMono8);
    if (VmbErrorSuccess == err) {
      std::cout << "[INFO] Pixel format set! " << std::endl;    
    } else if (VmbErrorSuccess != err) {
        std::cout << "[INFO] Pixel format NOT set! " << std::endl;
    }
  }

  
  
//   err = SP_ACCESS(cam)->GetFeatureByName("PixelFormat", pFeature);
//   if (VmbErrorSuccess == err) {
      
//       std::cout << "[INFO - PIXEL FORMAT] Feature Accessed!" << std::endl;
//       const char* pixel_format;
//       err = pFeature.get()->GetEntry(pixel_format);
      
//       if (VmbErrorSuccess == err) {
//         std::cout << "[INFO - PIXEL FORMAT] The camera's current pixel format: " << pixel_format << std::endl;
//       }

//   }

  //

  std::string name = "Vimba";
  std::cout << "[INFO] Streaming started!" << std::endl;


  while (1) {
    if (API.GetFrame(image, cam, pFrameObserver) == VmbFrameStatusComplete) {
      try {
        // open image

        cv::namedWindow(name);
        cv::imshow(name, image);
        cv::waitKey(10);  
      } catch (cv::Exception& e) {
        cerr << e.msg << std::endl; 
      }
    }
  }
  
  // stop the streaming
  cam.get()->GetFeatureByName("AcquisitionStop", pFeature);
  err = pFeature.get()->RunCommand();

//   if (VmbErrorSuccess != err)
//   {
//       std::cout << ErrorCodeToMessage(err) << std::endl;
//       std::cout << "Error starting the stream with the camera, ending..." << std::endl;
//       cam.get()->Close();
//       system.Shutdown();
//       return 0;
//   }
  
  // We need to wait for the buffer to start loading
  
  
//   if (VmbErrorSuccess != err)
//   {
//       std::cout << ErrorCodeToMessage(err) << std::endl;
//       std::cout << "Error starting the stream with the camera, ending..." << std::endl;
//       cam.get()->Close();
//       system.Shutdown();
//       return 0;
//   }

  

//   confirm acquisition
//   if (VmbErrorSuccess == pFeature.get()->RunCommand()) {
//       std::cout << " Acquisition started " << std :: endl;
//   }

  // new code

  // cv::namedWindow(name, cv::WINDOW_NORMAL);

  // cv::imshow(name, image);
  // cv::waitKey(0);
  // cv::destroyWindow(name);

  //open_image(image, "Reference");


//   // While images are being taken we can start processing them
  
//   // First a short calibration to get the similarity
//   //  NOTE: in the future this should include creating the homography between the LIDAR and the camera
//   // maybe some tweaks to the camera saturation and exposure configuration as well (currently done manual)
//   int niter = 0;
//   for (int i = 0; i < 30; i++)
//   {
//       std::cout << "Calibrating Frame: " << i << std::endl;
//       if (niter == 0)
//       {
//           if (API.GetFrame(image, cam, pFrameObserver) == VmbFrameStatusComplete)
//           {
//               niter = 1;
//               open_image(image, "Reference");
//           }
//       }
//       else if (niter == 1)
//       {
//           if (API.GetFrame(image2, cam, pFrameObserver) == VmbFrameStatusComplete)
//           {
//               try
//               {
//                   calibrate(image, image2);
//               }
//               catch (cv::Exception& e)
//               {
//                   cerr << e.msg << std::endl;
//               }
//           }
//       }
//   }
  
//   std::cout << "Calibration finished, max similairy value found was: " << SIMILARITY << std::endl;
//   // We stop the acquisition to start the true checking
//   cam.get()->GetFeatureByName("AcquisitionStop", pFeature);
//   err = pFeature.get()->RunCommand();
  
//   if (VmbErrorSuccess != err)
//   {
//       std::cout << ErrorCodeToMessage(err) << std::endl;
//       std::cout << "Error stopping the stream with the camera, ending..." << std::endl;
//       cam.get()->Close();
//       system.Shutdown();
//       return 0;
//   }


  
//   // While loop to run until ESC is pressed, this way we reuse the allocated frames
//   while (1) //(!GetAsyncKeyState(VK_ESCAPE))
//   {
//       // we restart from the beginning 
//       cam.get()->GetFeatureByName("AcquisitionStart", pFeature);
//       err = pFeature.get()->RunCommand();
  
//       // We need to wait for the buffer to start loading
//       usleep(250000);
  
//       niter = 0;
//       for (int i = 0; i < NUM_FRAMES; i++)
//       {
//           std::cout << "Processing Frame: " << i << std::endl;
//           if (niter == 0) // first frame to be used as reference
//           {
//               if (API.GetFrame(image, cam, pFrameObserver) == VmbFrameStatusComplete)
//               {
//                   niter = 1;
//                   //open_image(image, "First frame");
//               }
//           }
//           else if (niter == 1)
//           {
//               if (API.GetFrame(image2, cam, pFrameObserver) == VmbFrameStatusComplete)
//               {
//                   try
//                   {
//                       cv::namedWindow("feed", cv::WINDOW_NORMAL); // window to show the results
//                       check_scene(image, image2); // look for any new or moving object
//                       image = image2; // save the second image as reference for the next iteration
//                   }
//                   catch (cv::Exception& e)
//                   {
//                       cerr << e.msg << std::endl;
//                   }
//               }
//           }
//       }
  
//       // stop the streaming
//       cam.get()->GetFeatureByName("AcquisitionStop", pFeature);
//       err = pFeature.get()->RunCommand();
  
//   }
  
  // clean up
  cv::destroyAllWindows();
  cam.get()->EndCapture();
  cam.get()->FlushQueue();
  cam.get()->RevokeAllFrames();
  for (FramePtrVector::iterator iter = frames.begin(); frames.end() != iter; ++iter)
  {
      // Unregister the observer
      (*iter)->UnregisterObserver();
  }
  
  cam.get()->Close(); // close the communication

  // end of code 
  system.Shutdown();
  
  return 0;

}
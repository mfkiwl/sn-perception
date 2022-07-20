#include "ApiController.h"
#include <iostream>
#include <sstream>
// usleep timer - uses microseconds
#include <unistd.h>

#include <Common/StreamSystemInfo.h>
#include <Common/ErrorCodeToMessage.h>


extern int NUM_FRAMES;

namespace AVT {
 namespace VmbAPI {
  namespace Examples {

	ApiController::ApiController()
		// Get a reference to the Vimba singleton
		: m_system(VimbaSystem::GetInstance())
	{
	}


	// Overload
	ApiController::ApiController(CameraPtr pCamera, IFrameObserverPtr pFrameObserver)
		// Get a reference to the Vimba singleton
		: m_system(VimbaSystem::GetInstance())
	{
		m_pCamera = pCamera;
		m_pFrameObserver = pFrameObserver;
	}

	ApiController::~ApiController()
	{
	}

	// Translates Vimba error codes to readable error messages
	std::string ApiController::ErrorCodeToMessage(VmbErrorType eErr) const
	{
		return AVT::VmbAPI::Examples::ErrorCodeToMessage(eErr);
	}

	VmbErrorType ApiController::StartUp()
	{
		VmbErrorType res;

		// Start Vimba
		res = m_system.Startup();
		if (VmbErrorSuccess == res)
		{
			// This will be wrapped in a shared_ptr so we don't delete it
			SP_SET(m_pCameraObserver, new CameraObserver());
			// Register an observer whose callback routine gets triggered whenever a camera is plugged in or out
			res = m_system.RegisterCameraListObserver(m_pCameraObserver);
		}

		return res;
	}

	void ApiController::ShutDown()
	{
		// Release Vimba
		m_system.Shutdown();
	}
	/*** helper function to set image size to a value that is dividable by modulo 2.
	\note this is needed because AVTImageTransform does not support odd values for some input formats
	*/
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

	VmbErrorType ApiController::StartContinuousImageAcquisition(const std::string& rStrCameraID)
	{
		// Open the desired camera by its ID
		VmbErrorType res = m_system.OpenCameraByID(rStrCameraID.c_str(), VmbAccessModeFull, m_pCamera);
		if (VmbErrorSuccess == res)
		{
			// Set the GeV packet size to the highest possible value
			// (In this example we do not test whether this cam actually is a GigE cam)
			FeaturePtr pCommandFeature;
			if (VmbErrorSuccess == SP_ACCESS(m_pCamera)->GetFeatureByName("GVSPAdjustPacketSize", pCommandFeature))
			{
				if (VmbErrorSuccess == SP_ACCESS(pCommandFeature)->RunCommand())
				{
					bool bIsCommandDone = false;
					do
					{
						if (VmbErrorSuccess != SP_ACCESS(pCommandFeature)->IsCommandDone(bIsCommandDone))
						{
							break;
						}
					} while (false == bIsCommandDone);
				}
			}
			res = SetValueIntMod2(m_pCamera, "Width", m_nWidth);
			if (VmbErrorSuccess == res)
			{
				res = SetValueIntMod2(m_pCamera, "Height", m_nHeight);
				if (VmbErrorSuccess == res)
				{
					FeaturePtr pFormatFeature;
					// Set pixel format. For the sake of simplicity we only support Mono and RGB in this example.
					res = SP_ACCESS(m_pCamera)->GetFeatureByName("PixelFormat", pFormatFeature);
					if (VmbErrorSuccess == res)
					{
						// Read back the currently selected pixel format
						SP_ACCESS(pFormatFeature)->GetValue(m_nPixelFormat);

						if (VmbErrorSuccess == res)
						{
							// Create a frame observer for this camera (This will be wrapped in a shared_ptr so we don't delete it)
							SP_SET(m_pFrameObserver, new FrameObserver(m_pCamera));
							// Start streaming
							res = SP_ACCESS(m_pCamera)->StartContinuousImageAcquisition(NUM_FRAMES, m_pFrameObserver);
							if (VmbErrorSuccess == res) std::cout << "SUCCESS starting continuous Image Acquisition " << std::endl;
							else  std::cout << ErrorCodeToMessage(res) << std::endl;
						}
					}
				}
			}
		}

		return res;
	}

	VmbErrorType ApiController::StopContinuousImageAcquisition()
	{
		// Stop streaming
		m_pCamera->StopContinuousImageAcquisition();

		// Close camera
		return  m_pCamera->Close();
	}

	CameraPtrVector ApiController::GetCameraList()
	{
		CameraPtrVector cameras;
		// Get all known cameras
		if (VmbErrorSuccess == m_system.GetCameras(cameras))
		{
			// And return them
			return cameras;
		}
		return CameraPtrVector();
	}

	int ApiController::GetWidth()
	{
		return (int)m_nWidth;
	}

	int ApiController::GetHeight()
	{
		return (int)m_nHeight;
	}

	VmbPixelFormatType ApiController::GetPixelFormat()
	{
		return (VmbPixelFormatType)m_nPixelFormat;
	}

	// Returns the oldest frame that has not been picked up yet
	FramePtr ApiController::GetFrame()
	{
		return SP_DYN_CAST(m_pFrameObserver, FrameObserver)->GetFrame();
	}

	// Returns the oldest frame that has not been picked up yet
	FramePtr ApiController::GetFrame(IFrameObserverPtr pFrameObserver)
	{
		return SP_DYN_CAST(pFrameObserver, FrameObserver)->GetFrame();
	}

	// Get the oldest frame and encode data in the given Mat
	VmbFrameStatusType ApiController::GetFrame(cv::Mat& m)
	{
		FramePtr frame = SP_DYN_CAST(m_pFrameObserver, FrameObserver)->GetFrame();
		// Make sure the frame transfer is completed and successful
		VmbFrameStatusType status = VmbFrameStatusIncomplete;
		VmbErrorType err = VmbErrorIncomplete;
		if (SP_ISNULL(frame))
		{
			SP_ACCESS(m_pCamera)->QueueFrame(frame);
			return status;
		}
		err = SP_ACCESS(frame)->GetReceiveStatus(status);
		// Successful check
		if (err != VmbErrorSuccess)
		{
			std::cout << "Problems reading status, skipping frame..." << std::endl;
			std::cout << ErrorCodeToMessage(VmbErrorSuccess) << std::endl;
			status = VmbFrameStatusIncomplete;
		}

		// See if it is not corrupt
		if (VmbFrameStatusComplete == status)
		{
			unsigned char* buffer;
			err = SP_ACCESS(frame)->GetImage(buffer);

			if (err == VmbErrorSuccess)
			{
				std::cout << "Image successfully acquired..." << std::endl;
				try
				{
					m.create(m_nHeight, m_nWidth, CV_8UC1);
					m.data = buffer;
					//SP_ACCESS(m_pCamera)->QueueFrame(frame);
				}
				catch (cv::Exception& e)
				{
					std::cerr << e.msg << std::endl;
				}
			}
			else
			{
				std::cout << "Could not get the image, skipping it." << std::endl;
				status = VmbFrameStatusInvalid;
			}
		}
		else std::cout << "Corrupted Frame Skipping..." << std::endl;

		// done copying the frame, give it back to the camera
		SP_ACCESS(m_pCamera)->QueueFrame(frame);

		return status;
	}

	// Get the oldest frame and encode data in the given Mat - Overload
	VmbFrameStatusType ApiController::GetFrame(cv::Mat& m,CameraPtr pCamera, IFrameObserverPtr pFrameObserver)
	{
		// Dave - You may need a timeout error here!
		if (!FrameAvailable(pFrameObserver))
		{
			std::cout << "Waiting for Frame to arrive" << std::endl;
			while (!FrameAvailable(pFrameObserver)) continue;
			usleep(100000); // frame needs to get copied before we can read it. Currently there is no way to check when this happened.
		}
		FramePtr frame = SP_DYN_CAST(pFrameObserver, FrameObserver)->GetFrame();
		// Make sure the frame transfer is completed and successful
		VmbFrameStatusType status = VmbFrameStatusIncomplete;
		VmbErrorType err = VmbErrorIncomplete;
		if (SP_ISNULL(frame))
		{
			SP_ACCESS(pCamera)->QueueFrame(frame);
			return status;
		}
		err = SP_ACCESS(frame)->GetReceiveStatus(status);
		// Successful check
		if (err != VmbErrorSuccess)
		{
			std::cout << "Problems reading status, skipping frame..." << std::endl;
			std::cout << ErrorCodeToMessage(VmbErrorSuccess) << std::endl;
			status = VmbFrameStatusIncomplete;
		}

		// See if it is not corrupt
		if (VmbFrameStatusComplete == status)
		{
			unsigned char* buffer;
			err = SP_ACCESS(frame)->GetImage(buffer);


			if (err == VmbErrorSuccess)
			{
				std::cout << "Image successfully acquired..." << std::endl;
				try
				{
					//m.create(m_nHeight, m_nWidth, CV_8UC1);
					m.data = buffer;
					//SP_ACCESS(pCamera)->QueueFrame(frame);
				}
				catch (cv::Exception& e)
				{
					std::cerr << e.msg << std::endl;
				}
			}
			else
			{
				std::cout << "Could not get the image, skipping it." << std::endl;
				status = VmbFrameStatusInvalid;
			}
		}
		else std::cout << "Corrupted Frame Skipping..." << std::endl;

		// done copying the frame, give it back to the camera
		SP_ACCESS(pCamera)->QueueFrame(frame);

		return status;
	}

	// Clears all remaining frames that have not been picked up
	void ApiController::ClearFrameQueue()
	{
		SP_DYN_CAST(m_pFrameObserver, FrameObserver)->ClearFrameQueue();
	}

	bool ApiController::FrameAvailable()
	{
		return SP_DYN_CAST(m_pFrameObserver, FrameObserver)->FrameAvailable();
	}

	// Overload
	bool ApiController::FrameAvailable(IFrameObserverPtr pFrameObserver)
	{
		return SP_DYN_CAST(pFrameObserver, FrameObserver)->FrameAvailable();
	}

	unsigned int ApiController::GetQueueFrameSize()
	{
		return SP_DYN_CAST(m_pFrameObserver, FrameObserver)->GetQueueFrameSize();
	}

	// Queues a frame to continue streaming
	VmbErrorType ApiController::QueueFrame(FramePtr pFrame)
	{
		return SP_ACCESS(m_pCamera)->QueueFrame(pFrame);
	}

	// Returns the camera observer as QObjects to connect their signals to the view's sots
	CameraObserver* ApiController::GetCameraObserver()
	{
		return SP_DYN_CAST(m_pCameraObserver, CameraObserver).get();
	}

	// Returns the frame observer as QObjects to connect their signals to the view's sots
	FrameObserver* ApiController::GetFrameObserver()
	{
		return SP_DYN_CAST(m_pFrameObserver, FrameObserver).get();
	}

	std::string ApiController::GetVersion() const
	{
		std::ostringstream os;
		os << m_system;
		return os.str();
	}

  }
 }
} 
#pragma once
#include <string>

#include "opencv2/opencv.hpp"
#include <Common/ErrorCodeToMessage.h>

#include "CameraObserver.h"
#include "FrameObserver.h"
#include "constants.h"

namespace AVT {
 namespace VmbAPI {
  namespace Examples {
	class ApiController
	{
	public:
		ApiController();
		ApiController(CameraPtr pCamera, IFrameObserverPtr pFrameObserver);
		~ApiController();

		VmbErrorType        StartUp();
		void                ShutDown();

		VmbErrorType        StartContinuousImageAcquisition(const std::string& rStrCameraID);
		VmbErrorType        StopContinuousImageAcquisition();

		int                 GetWidth();
		int                 GetHeight();
		VmbPixelFormatType  GetPixelFormat();
		CameraPtrVector     GetCameraList();
		FramePtr            GetFrame();
		FramePtr            GetFrame(IFrameObserverPtr pFrameObserver);
		VmbFrameStatusType GetFrame(cv::Mat& m);
		VmbFrameStatusType GetFrame(cv::Mat& m, CameraPtr pCamera, IFrameObserverPtr pFrameObserver);
		bool FrameAvailable();
		bool FrameAvailable(IFrameObserverPtr pFrameObserver);
		unsigned int GetQueueFrameSize();
		VmbErrorType        QueueFrame(FramePtr pFrame);
		void                ClearFrameQueue();

		CameraObserver *GetCameraObserver();
		FrameObserver* GetFrameObserver();

		std::string         ErrorCodeToMessage(VmbErrorType eErr) const;
		std::string         GetVersion() const;

	private:
		// A reference to our Vimba singleton
		VimbaSystem& m_system;
		// The currently streaming camera
		CameraPtr                   m_pCamera;
		// Every camera has its own frame observer
		IFrameObserverPtr           m_pFrameObserver;
		// Our camera observer
		ICameraListObserverPtr      m_pCameraObserver;
		// The current pixel format
		VmbInt64_t                  m_nPixelFormat;
		// The current width
		VmbInt64_t                  m_nWidth;
		// The current height
		VmbInt64_t                  m_nHeight;
	};
  }
 }
} 



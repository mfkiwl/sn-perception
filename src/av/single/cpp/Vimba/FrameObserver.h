#pragma once
#include <VimbaCPP.h>
#include <queue>
#include <mutex>

namespace AVT {
	namespace VmbAPI {

		class FrameObserver : virtual public IFrameObserver
		{
			//    Q_OBJECT

		public:
			// We pass the camera that will deliver the frames to the constructor
			FrameObserver(CameraPtr pCamera) : IFrameObserver(pCamera) { ; }

			// This is our callback routine that will be executed on every received frame
			virtual void FrameReceived(const FramePtr pFrame);

			// After the view has been notified about a new frame it can pick it up
			FramePtr GetFrame();

			bool FrameAvailable();
			unsigned int GetQueueFrameSize();

			// Clears the double buffer frame queue
			void ClearFrameQueue();

		private:
			// Since a Qt signal cannot contain a whole frame
			// the frame observer stores all FramePtr
			std::queue<FramePtr> m_Frames;
			std::mutex m_FramesMutex;
			//    QMutex m_FramesMutex;

			//  signals:
				// The frame received event that passes the frame directly
			void FrameReceivedSignal(int status);

		};

	}
}
#include <iostream>

#include "FrameObserver.h"


namespace AVT {
	namespace VmbAPI {

		void FrameObserver::FrameReceived(const FramePtr pFrame)
		{
			bool bQueueDirectly = true;
			VmbFrameStatusType eReceiveStatus;

			//    if( pFrame->GetReceiveStatus( eReceiveStatus ) == VmbFrameStatusComplete )
			if (pFrame->GetReceiveStatus(eReceiveStatus) == VmbErrorSuccess)
			{
				// Lock the frame queue
				m_FramesMutex.lock();
				// Add frame to queue
				m_Frames.push(pFrame);
				// Unlock frame queue
				m_FramesMutex.unlock();
				// Emit the frame received signal
		//        emit FrameReceivedSignal( eReceiveStatus );
				bQueueDirectly = false;

				//	std::cout << "Received a frame, pushing to queue, size " << m_Frames.size() << std::endl;
			}

			// If any error occurred we queue the frame without notification
			if (true == bQueueDirectly)
			{
				m_pCamera->QueueFrame(pFrame);
			}
		}

		// Returns the oldest frame that has not been picked up yet
		FramePtr FrameObserver::GetFrame()
		{
			// Lock the frame queue
			m_FramesMutex.lock();

			FramePtr res;
			if (!m_Frames.empty())
			{
				// Pop frame from queue
				res = m_Frames.front();
				m_Frames.pop();
			}

			else std::cout << "Frames empty!" << std::endl;
			// Unlock frame queue
			m_FramesMutex.unlock();
			return res;
		}

		bool FrameObserver::FrameAvailable()
		{
			return (m_Frames.size() > 0);
		}

		unsigned int FrameObserver::GetQueueFrameSize()
		{
			return m_Frames.size();
		}

		void FrameObserver::ClearFrameQueue()
		{
			// Lock the frame queue
			m_FramesMutex.lock();
			// Clear the frame queue and release the memory
			std::queue<FramePtr> empty;
			std::swap(m_Frames, empty);
			// Unlock the frame queue
			m_FramesMutex.unlock();
		}

	}
} 
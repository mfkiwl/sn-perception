#pragma once
#include <VimbaCPP.h>

namespace AVT {
	namespace VmbAPI {

		class CameraObserver : public ICameraListObserver
		{

		public:
			// This is our callback routine that will be executed every time a camera was plugged in or out
			virtual void CameraListChanged(CameraPtr pCamera, UpdateTriggerType reason);

			//  signals:
				// The camera list changed signal that passes the new camera and the its state directly
			void CameraListChangedSignal(int reason);
		};

	}
} 
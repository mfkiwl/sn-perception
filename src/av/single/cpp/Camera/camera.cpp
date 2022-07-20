// #include <iostream>

// # include "Vimba.h"

// namespace AVT {
//   namespace VmbAPI {

//   /* Observer that reacts on new frames. */
//   class FrameObserver : public IFrameObserver {
//    public:
//     FrameObserver(CameraPtr pCamera) : IFrameObserver(pCamera);
//     void FrameReceived(const FramePtr pFrame);
    
//   };


//   /* Constructor for the FrameObserver class. In your contructor call the constructor
//    of the base class and pass a camera object */
//   FrameObserver::FrameObserver(CameraPtr pCamera) : IFrameObserver(pCamera) {
//     // init code goes here
//   }

//   /* Frame callback notifies about incoming frames (send notification 
//   to working thread). Do not apply image processing within this callback (performance).
//   When the frame has been processed, requeue it. */
//   void FrameObserver::FrameReceived(const FramePtr pFrame) {

//     VmbFrameStatusType eReceiveStatus;
    

//     if (VmbErrorSuccess == pFrame->GetReceiveStatus(eReceiveStatus)) {
//     if (VmbFrameStatusComplete == eReiveStatus) {
//       // Put your code here to react on a successfully received frame.
//     } else {
//         // Put your code here to react on an unsuccessfully received frame.
//       }
//     }

//     // When you are finished copying the frame, re-queue it.
//     pCamera_->QueueFrame(pFrame);
//   }

//   // block scope
//   {
//     VmbErrorType res;
//     FramePtr pFrame;
//     CameraPtr pCamera;

//     // After defining the observer, register the observer before queuing the frame.
//     res = pFrame.RegisterObserver( IFrameObserverPtr( new FrameObserver( pCamera )));
//   }


// }
/** ****************************************************************************
 *  @file    faces_framework_test.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/05
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <trace.hpp>
#include <Viewer.hpp>
#include <FaceAnnotation.hpp>
#include <FaceComposite.hpp>
#include <utils.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
main
  (
  int argc,
  char **argv
  )
{
  // Read sample annotations
  UPM_PRINT("Processing from a video file ...");
  cv::VideoCapture capture;
  capture.open("test/000909960.avi");
  if (not capture.isOpened())
  {
    UPM_ERROR("Could not grab images from video");
    return EXIT_FAILURE;
  }
  cv::Mat frame;
  std::vector<upm::FaceAnnotation> faces;
  upm::FaceAnnotation ann;

  /// Load face components
  boost::shared_ptr<upm::FaceComposite> composite(new upm::FaceComposite());

//  boost::shared_ptr<upm::Viewer> viewer(new upm::Viewer);
//  viewer->init(0, 0, "faces_framework_test");
  for (;;)
  {
    if (not capture.grab())
      break;

    capture.retrieve(frame);
    if (frame.empty())
      break;

    // Process frame
    double ticks = processFrame(frame, composite, faces, ann);
    UPM_PRINT("FPS = " << cv::getTickFrequency()/ticks);

    // Draw results
//    showResults(viewer, ticks, 20, frame, composite, faces, ann);
  }

  UPM_PRINT("End of faces_framework_test");
  return EXIT_SUCCESS;
};

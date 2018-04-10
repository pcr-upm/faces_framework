/** ****************************************************************************
 *  @file    utils.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef UTILS_HPP
#define UTILS_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceComposite.hpp>
#include <FaceAlignment.hpp>
#include <FaceAnnotation.hpp>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

static const std::vector<float> HP_LABELS = {-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90};
extern std::map< FacePartLabel,std::vector<int> > DB_PARTS;
extern std::vector<unsigned int> DB_LANDMARKS;

double
processFrame
  (
  cv::Mat frame,
  boost::shared_ptr<FaceComposite> composite,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  );

void
showResults
  (
  const boost::shared_ptr<Viewer> &viewer,
  double ticks,
  int delay,
  cv::Mat frame,
  boost::shared_ptr<FaceComposite> composite,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  );

cv::Rect
intersection
  (
  const cv::Rect_<float> r1,
  const cv::Rect_<float> r2
  );

int
getHeadposeIdx
  (
  float label
  );

cv::Rect_<float>
getBbox
  (
  const FaceAnnotation &ann
  );

void
setPointsToComputeProjectionMatrix
  (
  const std::string &path,
  const FaceAnnotation &ann,
  std::vector<cv::Point2f> &image_pts,
  std::vector<cv::Point3f> &world_pts
  );

cv::Point3f
getHeadpose
  (
  const FaceAnnotation &ann
  );

void
getNormalizedErrors
  (
  const FaceAnnotation &face,
  const FaceAnnotation &ann,
  const ErrorMeasure &measure,
  std::vector<unsigned int> &indices,
  std::vector<float> &errors
  );

} // namespace upm

#endif /* UTILS_HPP */

/** ****************************************************************************
 *  @file    FaceDetector.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <FaceDetector.hpp>

namespace upm {

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceDetector::show
  (
  const boost::shared_ptr<upm::Viewer> &viewer,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  // Ground truth
  cv::Scalar cyan_color(255,122,0);
  int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  viewer->rectangle(ann.bbox.pos.x, ann.bbox.pos.y, ann.bbox.pos.width, ann.bbox.pos.height, thickness, cyan_color);

  // Detected bounding boxes
  cv::Scalar green_color(0,255,0);
  for (const FaceAnnotation &face : faces)
    viewer->rectangle(face.bbox.pos.x, face.bbox.pos.y, face.bbox.pos.width, face.bbox.pos.height, thickness, green_color);
};

float
areaIntersection
  (
  const cv::Rect &ann_rect,
  const cv::Rect &face_rect
  )
{
  const int max_x = MAX(ann_rect.x, face_rect.x);
  const int max_y = MAX(ann_rect.y, face_rect.y);
  const int min_w = MIN(ann_rect.x+ann_rect.width, face_rect.x+face_rect.width) - max_x;
  const int min_h = MIN(ann_rect.y+ann_rect.height, face_rect.y+face_rect.height) - max_y;
  return ((min_w < 0) or (min_h < 0)) ? 0 :  min_w*min_h;
};

float
areaUnion
  (
  const cv::Rect &ann_rect,
  const cv::Rect &face_rect
  )
{
  return (ann_rect.width*ann_rect.height) + (face_rect.width*face_rect.height) - areaIntersection(ann_rect, face_rect);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceDetector::evaluate
  (
  boost::shared_ptr<std::ostream> output,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  // Ratio of intersected areas
  for (const FaceAnnotation &face : faces)
  {
    float ratio = areaIntersection(ann.bbox.pos, face.bbox.pos) / areaUnion(ann.bbox.pos, face.bbox.pos);
    *output << getComponentClass() << " " << ann.filename << " " << ann.bbox.pos << " " << face.bbox.pos << " " << face.bbox.detector_idx << " " << ratio << std::endl;
  }
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceDetector::save
  (
  const std::string dirpath,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  // Save images with ratio of intersected areas greater than threshold
  const float threshold = 0.5f;
  int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  cv::Scalar cyan_color(255,122,0), green_color(0,255,0), red_color(0,0,255);
  float max_ratio = FLT_MIN;
  cv::Mat image = cv::imread(ann.filename, CV_LOAD_IMAGE_COLOR);
  cv::rectangle(image, ann.bbox.pos.tl(), ann.bbox.pos.br(), cyan_color, thickness);
  for (const FaceAnnotation &face : faces)
  {
    cv::rectangle(image, face.bbox.pos.tl(), face.bbox.pos.br(), green_color, thickness);
    // Ratio of intersected areas
    float ratio = areaIntersection(ann.bbox.pos, face.bbox.pos) / areaUnion(ann.bbox.pos, face.bbox.pos);
    if (ratio > max_ratio)
      max_ratio = ratio;
  }
  std::string text = std::to_string(max_ratio);
  cv::putText(image, text, cv::Point(10, image.rows-10), cv::FONT_HERSHEY_SIMPLEX, 1, red_color);
  if (max_ratio < threshold)
  {
    std::size_t found = ann.filename.find_last_of("/");
    std::string filepath = dirpath + ann.filename.substr(found+1);
    cv::imwrite(filepath, image);
  }
};

} // namespace upm

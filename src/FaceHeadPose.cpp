/** ****************************************************************************
 *  @file    FaceHeadPose.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <ModernPosit.h>
#include <boost/filesystem.hpp>

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
cv::Mat
projectAxis
  (
  const cv::Point3f &headpose
  )
{
  cv::Mat axis = cv::Mat::eye(3,3,cv::DataType<float>::type); // [yaw (blue), pitch (green), roll (red)]
  cv::Mat ann_axis, rot_matrix = ModernPosit::eulerToRotationMatrix(headpose);
  rot_matrix = (cv::Mat_<float>(3,3) <<  rot_matrix.at<float>(1,2), rot_matrix.at<float>(1,1),-rot_matrix.at<float>(1,0),
                                        -rot_matrix.at<float>(0,2),-rot_matrix.at<float>(0,1), rot_matrix.at<float>(0,0),
                                         rot_matrix.at<float>(2,2), rot_matrix.at<float>(2,1),-rot_matrix.at<float>(2,0));
  ann_axis = rot_matrix*axis;
  return ann_axis;
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
FaceHeadPose::show
  (
  const boost::shared_ptr<upm::Viewer> &viewer,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  // Ground truth
  cv::Scalar blue_color(255,0,0), green_color(0,255,0), red_color(0,0,255);
  double length = static_cast<int>(roundf(ann.bbox.pos.height)*0.5f);
  int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  cv::Mat ann_axis = projectAxis(ann.headpose) * length;
  cv::Point mid = (ann.bbox.pos.tl() + ann.bbox.pos.br()) * 0.5;
  viewer->line(mid.x, mid.y, mid.x+ann_axis.at<float>(1,0), mid.y-ann_axis.at<float>(0,0), thickness, blue_color);
  viewer->line(mid.x, mid.y, mid.x+ann_axis.at<float>(1,1), mid.y-ann_axis.at<float>(0,1), thickness, green_color);
  viewer->line(mid.x, mid.y, mid.x+ann_axis.at<float>(1,2), mid.y-ann_axis.at<float>(0,2), thickness, red_color);

  // Estimated head-pose
  cv::Scalar cyan_color(122,0,0), lime_color(0,122,0), salmon_color(0,0,122);
  for (const FaceAnnotation &face : faces)
  {
    length = static_cast<int>(roundf(face.bbox.pos.height)*0.5f);
    thickness = MAX(static_cast<int>(roundf(face.bbox.pos.height*0.01f)), 3);
    cv::Mat face_axis = projectAxis(face.headpose) * length;
    mid = (face.bbox.pos.tl() + face.bbox.pos.br()) * 0.5;
    viewer->line(mid.x, mid.y, mid.x+face_axis.at<float>(1,0), mid.y-face_axis.at<float>(0,0), thickness, cyan_color);
    viewer->line(mid.x, mid.y, mid.x+face_axis.at<float>(1,1), mid.y-face_axis.at<float>(0,1), thickness, lime_color);
    viewer->line(mid.x, mid.y, mid.x+face_axis.at<float>(1,2), mid.y-face_axis.at<float>(0,2), thickness, salmon_color);
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
FaceHeadPose::evaluate
  (
  boost::shared_ptr<std::ostream> output,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  for (const FaceAnnotation &face : faces)
    *output << getComponentClass() << " " << ann.filename << " " << ann.headpose << " " << face.headpose << std::endl;
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
FaceHeadPose::save
  (
  const std::string dirpath,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  // Save images with mean error greater than threshold
  const float threshold = 25.0f;
  cv::Scalar blue_color(255,0,0), green_color(0,255,0), red_color(0,0,255), cyan_color(122,0,0), lime_color(0,122,0), salmon_color(0,0,122);
  double length = static_cast<int>(roundf(ann.bbox.pos.height)*0.5f);
  int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  for (const FaceAnnotation &face : faces)
  {
    cv::Mat image = cv::imread(face.filename, cv::IMREAD_COLOR);
    cv::Mat ann_axis = projectAxis(ann.headpose) * length;
    cv::Point mid = (ann.bbox.pos.tl() + ann.bbox.pos.br()) * 0.5;
    cv::line(image, mid, cv::Point2f(mid.x+ann_axis.at<float>(1,0), mid.y-ann_axis.at<float>(0,0)), blue_color, thickness);
    cv::line(image, mid, cv::Point2f(mid.x+ann_axis.at<float>(1,1), mid.y-ann_axis.at<float>(0,1)), green_color, thickness);
    cv::line(image, mid, cv::Point2f(mid.x+ann_axis.at<float>(1,2), mid.y-ann_axis.at<float>(0,2)), red_color, thickness);

    length = static_cast<int>(roundf(face.bbox.pos.height)*0.5f);
    thickness = MAX(static_cast<int>(roundf(face.bbox.pos.height*0.01f)), 3);
    cv::Mat face_axis = projectAxis(face.headpose) * length;
    mid = (face.bbox.pos.tl() + face.bbox.pos.br()) * 0.5;
    cv::line(image, mid, cv::Point2f(mid.x+face_axis.at<float>(1,0), mid.y-face_axis.at<float>(0,0)), cyan_color, thickness);
    cv::line(image, mid, cv::Point2f(mid.x+face_axis.at<float>(1,1), mid.y-face_axis.at<float>(0,1)), lime_color, thickness);
    cv::line(image, mid, cv::Point2f(mid.x+face_axis.at<float>(1,2), mid.y-face_axis.at<float>(0,2)), salmon_color, thickness);

    // Absolute head-pose error
    float error = static_cast<float>(cv::sum(cv::abs(cv::Mat(ann.headpose-face.headpose).t()))[0]);
    std::string text = std::to_string(error);
    cv::putText(image, text, cv::Point(10, image.rows-10), cv::FONT_HERSHEY_SIMPLEX, 1, red_color);
    if (error > threshold)
    {
      std::size_t found = face.filename.find_last_of('/');
      std::string filepath;
      unsigned int num = 0;
      do
      {
        filepath = dirpath + std::to_string(num) + "_" + face.filename.substr(found+1);
        num++;
      }
      while (boost::filesystem::exists(filepath));
      cv::imwrite(filepath, image);
    }
  }
};

} // namespace upm

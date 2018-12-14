/** ****************************************************************************
 *  @file    utils.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <utils.hpp>
#include <trace.hpp>
#include <ModernPosit.h>
#include <iomanip>
#include <boost/algorithm/cxx11/iota.hpp>

namespace upm {

std::map< FacePartLabel,std::vector<int> > DB_PARTS;
std::vector<unsigned int> DB_LANDMARKS;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
double
processFrame
  (
  cv::Mat frame,
  boost::shared_ptr<FaceComposite> composite,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  double ticks = static_cast<double>(cv::getTickCount());
  composite->process(frame, faces, ann);
  return static_cast<double>(cv::getTickCount()) - ticks;
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
showResults
  (
  const boost::shared_ptr<Viewer> &viewer,
  double ticks,
  int delay,
  cv::Mat frame,
  boost::shared_ptr<FaceComposite> composite,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  viewer->resizeCanvas(frame.cols, frame.rows);
  viewer->beginDrawing();
  viewer->image(frame, 0, 0, frame.cols, frame.rows);
  std::ostringstream outs;
  outs << "FPS =" << std::setprecision(3);
  outs << static_cast<double>(cv::getTickFrequency())/ticks << std::ends;
  viewer->text(outs.str(), 20, frame.rows-20, cv::Scalar(255,0,255), 0.5);
  composite->show(viewer, faces, ann);
  viewer->endDrawing(delay);
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
cv::Rect
intersection
  (
  const cv::Rect_<float> r1,
  const cv::Rect_<float> r2
  )
{
  /// Find overlapping region
  cv::Rect_<float> intersect;
  intersect.x = std::max(r1.x, r2.x);
  intersect.y = std::max(r1.y, r2.y);
  intersect.width  = std::min(r1.x+r1.width, r2.x+r2.width) - intersect.x;
  intersect.height = std::min(r1.y+r1.height, r2.y+r2.height) - intersect.y;

  /// Check for non-overlapping regions
  if (intersect.width <= 0.0f or intersect.height <= 0.0f)
    intersect = cv::Rect_<float>(0.0f, 0.0f, 0.0f, 0.0f);

  return intersect;
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
int
getHeadposeIdx
  (
  float label
  )
{
  cv::Mat diff = cv::abs(cv::Mat(HP_LABELS)-label);
  float minimum = *std::min_element(diff.begin<float>(), diff.end<float>());
  /// In case of draw use lower angle interval (i.e. if [0, 30] are 15 then 0)
  const int num_headposes = static_cast<int>(HP_LABELS.size());
  const int half = (num_headposes-1) / 2;
  std::vector<int> indices(num_headposes);
  boost::algorithm::iota(indices.begin(), indices.end(), -half);
  int best_idx, best_min = INT_MAX;
  for (auto it = diff.begin<float>(); it < diff.end<float>(); it++)
  {
    int idx = static_cast<int>(std::distance(diff.begin<float>(),it));
    if (((*it) == minimum) and (abs(indices[idx]) < best_min))
    {
      best_idx = idx;
      best_min = abs(indices[idx]);
    }
  }
  return best_idx;
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
cv::Rect_<float>
getBbox
  (
  const FaceAnnotation &ann
  )
{
  /// Default annotation or smallest feature-based bounding box enclosing
  if (ann.bbox.pos == upm::FaceAnnotation().bbox.pos)
  {
    std::vector<cv::Point2f> points;
    for (const FacePart &ann_part : ann.parts)
      for (const FaceLandmark &ann_landmark : ann_part.landmarks)
        points.push_back(ann_landmark.pos);
    return cv::boundingRect(points);
  }
  return ann.bbox.pos;
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
cv::Point3f
getHeadpose
  (
  const FaceAnnotation &ann
  )
{
  /// Default annotation or modern POSIT feature-based algorithm
  if ((ann.headpose == upm::FaceAnnotation().headpose) and (ann.parts != upm::FaceAnnotation().parts))
  {
    /// Load 3D face shape
    std::vector<cv::Point3f> world_all;
    std::vector<unsigned int> index_all;
    ModernPosit::loadWorldShape("faces_framework/headpose/posit/data/", DB_LANDMARKS, world_all, index_all);
    /// Robust correspondences
    std::vector<cv::Point3f> world_pts;
    std::vector<cv::Point2f> image_pts;
    const std::vector<unsigned int> mask = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    ModernPosit::setCorrespondences(world_all, index_all, ann, mask, world_pts, image_pts);

    /// Intrinsic parameters (image -> camera)
    const float BBOX_SCALE = 0.3f;
    cv::Point2f shift(ann.bbox.pos.width*BBOX_SCALE, ann.bbox.pos.height*BBOX_SCALE);
    cv::Rect_<float> bbox_enlarged = cv::Rect_<float>(ann.bbox.pos.x-shift.x, ann.bbox.pos.y-shift.y, ann.bbox.pos.width+(shift.x*2), ann.bbox.pos.height+(shift.y*2));
    bbox_enlarged.x = bbox_enlarged.x+(bbox_enlarged.width*0.5f)-(bbox_enlarged.height*0.5f);
    bbox_enlarged.width = bbox_enlarged.height;
    double focal_length = static_cast<double>(bbox_enlarged.width) * 1.5;
    cv::Point2f face_center = (bbox_enlarged.tl() + bbox_enlarged.br()) * 0.5f;
    cv::Mat cam_matrix;
    cam_matrix = (cv::Mat_<float>(3,3) << focal_length,0,face_center.x, 0,focal_length,face_center.y, 0,0,1);
    /// Extrinsic parameters (camera -> 3D world)
    cv::Mat rot_matrix, trl_matrix;
    ModernPosit::run(world_pts, image_pts, cam_matrix, 100, rot_matrix, trl_matrix);
//    cv::Mat rot_matrix1 = (cv::Mat_<float>(3,4) << rot_matrix.at<float>(0,0),rot_matrix.at<float>(0,1),rot_matrix.at<float>(0,2),trl_matrix.at<float>(0), rot_matrix.at<float>(1,0),rot_matrix.at<float>(1,1),rot_matrix.at<float>(1,2),trl_matrix.at<float>(1), rot_matrix.at<float>(2,0),rot_matrix.at<float>(2,1),rot_matrix.at<float>(2,2),trl_matrix.at<float>(2));
//    std::cout << rot_matrix1 << std::endl;
//    cv::Mat rmat2, rvec2, tvec2;
//    cv::solvePnP(world_pts, image_pts, cam_matrix, cv::Mat(), rvec2, tvec2, false, cv::SOLVEPNP_ITERATIVE);
//    cv::Rodrigues(rvec2, rmat2);
//    cv::Mat rot_matrix2 = (cv::Mat_<float>(3,4) << rmat2.at<float>(0,0),rmat2.at<float>(0,1),rmat2.at<float>(0,2),tvec2.at<float>(0), rmat2.at<float>(1,0),rmat2.at<float>(1,1),rmat2.at<float>(1,2),tvec2.at<float>(1), rmat2.at<float>(2,0),rmat2.at<float>(2,1),rmat2.at<float>(2,2),tvec2.at<float>(2));
//    std::cout << rot_matrix2 << std::endl;
//    cv::Mat rmat3, rvec3, tvec3, inliers;
//    cv::solvePnPRansac(world_pts, image_pts, cam_matrix, cv::Mat(), rvec3, tvec3, false, 100, 15.0f, 0.995, inliers, cv::SOLVEPNP_ITERATIVE);
//    cv::Rodrigues(rvec3, rmat3);
//    cv::Mat rot_matrix3 = (cv::Mat_<float>(3,4) << rmat3.at<float>(0,0),rmat3.at<float>(0,1),rmat3.at<float>(0,2),tvec3.at<float>(0), rmat3.at<float>(1,0),rmat3.at<float>(1,1),rmat3.at<float>(1,2),tvec3.at<float>(1), rmat3.at<float>(2,0),rmat3.at<float>(2,1),rmat3.at<float>(2,2),tvec3.at<float>(2));
//    std::cout << rot_matrix3 << std::endl;

    /// Decomposition of a rotation matrix into three Euler angles
    return ModernPosit::rotationMatrixToEuler(rot_matrix);
  }
  return ann.headpose;
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
getNormalizedErrors
  (
  const FaceAnnotation &face,
  const FaceAnnotation &ann,
  const ErrorMeasure &measure,
  std::vector<unsigned int> &indices,
  std::vector<float> &errors
  )
{
  float normalization;
  switch (measure)
  {
    case ErrorMeasure::pupils:
    {
      /// Pupil distance normalization
      std::vector<FaceLandmark> lpts = ann.parts[FacePartLabel::leye].landmarks, rpts = ann.parts[FacePartLabel::reye].landmarks;
      cv::Point2f lpupil, rpupil;
      for (const FaceLandmark &pt : lpts)
        lpupil += pt.pos;
      for (const FaceLandmark &pt : rpts)
        rpupil += pt.pos;
      normalization = static_cast<float>(cv::norm(lpupil*(1.0f/lpts.size())-rpupil*(1.0f/rpts.size())));
      break;
    }
    case ErrorMeasure::corners:
    {
      /// Outer corners of the eyes normalization
      std::vector<FaceLandmark> lpts = ann.parts[FacePartLabel::leye].landmarks, rpts = ann.parts[FacePartLabel::reye].landmarks;
      auto lcorner = std::find_if(lpts.begin(), lpts.end(), [](const FaceLandmark &obj){return obj.feature_idx == 7;});
      auto rcorner = std::find_if(rpts.begin(), rpts.end(), [](const FaceLandmark &obj){return obj.feature_idx == 12;});
      normalization = static_cast<float>(cv::norm((*lcorner).pos-(*rcorner).pos));
      break;
    }
    case ErrorMeasure::height:
    {
      /// Bounding box size normalization
      cv::Rect_<float> bbox = getBbox(ann);
      normalization = bbox.height;
      break;
    }
    default:
    {
      /// Diagonal of the bounding box
      cv::Rect_<float> bbox = getBbox(ann);
      normalization = cv::sqrt((bbox.width*bbox.width)+(bbox.height*bbox.height));
      break;
    }
  }
  /// Estimate normalized error for each labelled landmark
  for (const FacePart &ann_part : ann.parts)
    for (const FaceLandmark &ann_landmark : ann_part.landmarks)
    {
      unsigned int idx = ann_landmark.feature_idx;
      for (const FacePart &face_part : face.parts)
      {
        auto found = std::find_if(face_part.landmarks.begin(), face_part.landmarks.end(), [&idx](const FaceLandmark &obj){return obj.feature_idx == idx;});
        if (found != face_part.landmarks.end())
        {
          indices.push_back(idx);
          errors.push_back(static_cast<float>(cv::norm((*found).pos-ann_landmark.pos)/normalization)*100.0f);
        }
      }
    }
};

} // namespace upm

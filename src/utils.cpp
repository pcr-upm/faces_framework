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
#include <MeanFace3DModel.hpp>
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
  // Find overlapping region
  cv::Rect_<float> intersect;
  intersect.x = std::max(r1.x, r2.x);
  intersect.y = std::max(r1.y, r2.y);
  intersect.width  = std::min(r1.x+r1.width, r2.x+r2.width) - intersect.x;
  intersect.height = std::min(r1.y+r1.height, r2.y+r2.height) - intersect.y;

  // Check for non-overlapping regions
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
  // In case of draw use lower angle interval (i.e. if [0, 30] are 15 then 0)
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
  // Default annotation or smallest feature-based bounding box enclosing
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
void
setPointsToComputeProjectionMatrix
  (
  const std::string &path,
  const FaceAnnotation &ann,
  std::vector<cv::Point2f> &image_pts,
  std::vector<cv::Point3f> &world_pts
  )
{
  // Load 3D mean face coordinates
  MeanFace3DModel mean_face_3D;
  std::vector<int> db_landmarks;
  unsigned int num_landmarks = static_cast<int>(DB_LANDMARKS.size());
  switch (num_landmarks)
  {
    case 21: {
      mean_face_3D.load(path + "mean_face_3D_21.txt");
      db_landmarks = {1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24};
      break;
    }
    case 29: {
      mean_face_3D.load(path + "mean_face_3D_29.txt");
      db_landmarks = {1, 101, 3, 102, 4, 103, 6, 104, 7, 8, 9, 10, 105, 11, 12, 13, 14, 106, 17, 16, 107, 18, 20, 22, 21, 23, 108, 109, 24};
      break;
    }
    default: {
      mean_face_3D.load(path + "mean_face_3D_68.txt");
      db_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 1, 119, 2, 121, 3, 128, 129, 130, 17, 16, 133, 134, 135, 18, 4, 124, 5, 126, 6, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
      break;
    }
  }
  for (const FacePart &ann_part : ann.parts)
    for (const FaceLandmark &ann_landmark : ann_part.landmarks)
    {
      auto pos = std::find(db_landmarks.begin(),db_landmarks.end(),ann_landmark.feature_idx);
      if (pos == db_landmarks.end())
        continue;
      image_pts.push_back(ann_landmark.pos);
      cv::Point3f aux = mean_face_3D.getCoordinatesById(static_cast<int>(std::distance(db_landmarks.begin(),pos))+1);
      world_pts.push_back(cv::Point3f(aux.z,-aux.x,-aux.y));
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
cv::Point3f
getHeadpose
  (
  const FaceAnnotation &ann
  )
{
  // Default annotation or modern POSIT feature-based algorithm
  if (ann.headpose == upm::FaceAnnotation().headpose)
  {
    std::vector<cv::Point2f> image_pts;
    std::vector<cv::Point3f> world_pts;
    setPointsToComputeProjectionMatrix("faces_framework/headpose/posit/data/", ann, image_pts, world_pts);

    // Intrinsic parameters (image -> camera)
    cv::Mat img = cv::imread(ann.filename);
    double focal_length = static_cast<double>(img.cols) * 1.5;
    cv::Point2f img_center = cv::Point2f(img.cols, img.rows) * 0.5f;
    cv::Mat cam_matrix = (cv::Mat_<float>(3,3) << focal_length,0,img_center.x, 0,focal_length,img_center.y, 0,0,1);
    // Extrinsic parameters (camera -> 3D world)
    ModernPosit modernPosit;
    cv::Mat rot_matrix = modernPosit.run(world_pts, image_pts, cam_matrix, 100);
//    std::cout << rot_matrix << std::endl;
//    cv::Mat rvec, tvec = (cv::Mat_<float>(3,1) << rot_matrix.at<float>(0,3),rot_matrix.at<float>(1,3),rot_matrix.at<float>(2,3));
//    cv::Rodrigues(rot_matrix.colRange(0,3), rvec);
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

//    std::vector<cv::Point2f> image_pts_proj1, image_pts_proj2, image_pts_proj3;
//    cv::projectPoints(world_pts, rvec, tvec, cam_matrix, cv::Mat(), image_pts_proj1);
//    cv::projectPoints(world_pts, rvec2, tvec2, cam_matrix, cv::Mat(), image_pts_proj2);
//    cv::projectPoints(world_pts, rvec3, tvec3, cam_matrix, cv::Mat(), image_pts_proj3);
//    cv::Mat copy = img.clone();
//    for (int w=0; w < image_pts.size(); w++)
//    {
//      cv::circle(copy, cv::Point2f(image_pts[w].x,image_pts[w].y), 3, cv::Scalar(255,0,0), -1);
//      cv::circle(copy, cv::Point2f(image_pts_proj1[w].x,image_pts_proj1[w].y), 3, cv::Scalar(0,255,0), -1);
//      cv::circle(copy, cv::Point2f(image_pts_proj2[w].x,image_pts_proj2[w].y), 3, cv::Scalar(255,255,0), -1);
//      cv::circle(copy, cv::Point2f(image_pts_proj3[w].x,image_pts_proj3[w].y), 3, cv::Scalar(0,0,255), -1);
//    }
//    cv::imshow("posit", copy);
//    cv::waitKey(0);
//    std::cout << cv::Mat(image_pts) << std::endl;
//    std::cout << cv::Mat(image_pts_proj1) << std::endl;
//    std::cout << cv::Mat(image_pts_proj2) << std::endl;
//    std::cout << cv::Mat(image_pts_proj3) << std::endl;

    // Decomposition of a rotation matrix into three Euler angles
    cv::Vec3d euler = modernPosit.getEulerAngles(rot_matrix);
    return cv::Point3f(static_cast<float>(euler(0)), static_cast<float>(euler(1)), static_cast<float>(euler(2)));
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
      normalization = ann.bbox.pos.height;
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

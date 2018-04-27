/** ****************************************************************************
 *  @file    ModernPosit.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

#include <ModernPosit.h>
#include <MeanFace3DModel.hpp>
#include <trace.hpp>

void
ModernPosit::setCorrespondences
  (
  const std::string &path,
  const upm::FaceAnnotation &ann,
  const unsigned int &num_landmarks,
  std::vector<cv::Point3f> &world_all,
  std::vector<cv::Point3f> &world_pts,
  std::vector<cv::Point2f> &image_pts,
  const std::vector<int> &mask
  )
{
  // Load 3D mean face coordinates
  MeanFace3DModel mean_face_3D;
  std::vector<int> db_landmarks;
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
  for (const upm::FacePart &ann_part : ann.parts)
    for (const upm::FaceLandmark &ann_landmark : ann_part.landmarks)
    {
      auto pos = std::find(db_landmarks.begin(),db_landmarks.end(),ann_landmark.feature_idx);
      if (pos == db_landmarks.end())
        continue;
      cv::Point3f pt = mean_face_3D.getCoordinatesById(static_cast<int>(std::distance(db_landmarks.begin(),pos))+1);
      pt = cv::Point3f(pt.z,-pt.x,-pt.y);
      world_all.emplace_back(pt);
      if ((not ann_landmark.visible) or (std::find(mask.begin(),mask.end(),ann_landmark.feature_idx)) != mask.end())
        continue;
      world_pts.emplace_back(pt);
      image_pts.emplace_back(ann_landmark.pos);
    }
};

void
ModernPosit::run
  (
  const std::vector<cv::Point3f> &world_pts,
  const std::vector<cv::Point2f> &image_pts,
  const cv::Mat &cam_matrix,
  const int &max_iters,
  cv::Mat &rot_matrix,
  cv::Mat &trl_matrix
  )
{
  /// Homogeneous world points
  const unsigned int num_landmarks = static_cast<unsigned int>(image_pts.size());
  cv::Mat A(num_landmarks,4,CV_64F);
  for (int i=0; i < num_landmarks; i++)
  {
    A.at<double>(i,0) = static_cast<double>(world_pts[i].x);
    A.at<double>(i,1) = static_cast<double>(world_pts[i].y);
    A.at<double>(i,2) = static_cast<double>(world_pts[i].z);
    A.at<double>(i,3) = 1.0;
  }
  cv::Mat B = A.inv(cv::DECOMP_SVD);

  /// Normalize image points
  float focal_length = cam_matrix.at<float>(0,0);
  cv::Point2f img_center = cv::Point2f(cam_matrix.at<float>(0,2),cam_matrix.at<float>(1,2));
  std::vector<cv::Point2f> centered_pts;
  for (const cv::Point2f &pt : image_pts)
    centered_pts.push_back(cv::Point2f(pt - img_center) * (1.0f/focal_length));
  cv::Mat Ui(num_landmarks,1,CV_64F), Vi(num_landmarks,1,CV_64F);
  for (int i=0; i < num_landmarks; i++)
  {
    Ui.at<double>(i,0) = centered_pts[i].x;
    Vi.at<double>(i,0) = centered_pts[i].y;
  }

  /// POSIT loop
  double Tx = 0.0, Ty = 0.0, Tz = 0.0;
  std::vector<double> r1(4), r2(4), r3(4);
  std::vector<double> oldUi(num_landmarks), oldVi(num_landmarks), deltaUi(num_landmarks), deltaVi(num_landmarks);
  for (unsigned int iter=0; iter < max_iters; iter++)
  {
    cv::Mat I = B * Ui;
    cv::Mat J = B * Vi;

    /// Estimate translation vector and rotation matrix
    double normI = 1.0/std::sqrt(cv::sum(I.rowRange(0,3).mul(I.rowRange(0,3)))[0]);
    double normJ = 1.0/std::sqrt(cv::sum(J.rowRange(0,3).mul(J.rowRange(0,3)))[0]);
    Tz = std::sqrt(normI*normJ); // geometric average instead of arithmetic average of classicPosit
    for (int j=0; j < 4; j++)
    {
      r1[j] = I.at<double>(j,0)*Tz;
      r2[j] = J.at<double>(j,0)*Tz;
    }
    for (int j=0; j < 3; j++)
    {
      if ((r1[j] > 1.0) or (r1[j] < -1.0))
        r1[j] = std::max(-1.0,std::min(1.0,r1[j]));
      if ((r2[j] > 1.0) or (r2[j] < -1.0))
        r2[j] = std::max(-1.0,std::min(1.0,r2[j]));
    }
    r3[0] = r1[1]*r2[2] - r1[2]*r2[1];
    r3[1] = r1[2]*r2[0] - r1[0]*r2[2];
    r3[2] = r1[0]*r2[1] - r1[1]*r2[0];
    r3[3] = Tz;
    Tx = r1[3];
    Ty = r2[3];

    /// Compute epsilon, update Ui and Vi and check convergence
    std::vector<double> eps(num_landmarks,0.0);
    for (int i=0; i < num_landmarks; i++)
      for (int j=0; j < 4; j++)
        eps[i] += A.at<double>(i,j) * r3[j] / Tz;
    for (int i=0; i < num_landmarks; i++)
    {
      oldUi[i] = Ui.at<double>(i,0);
      oldVi[i] = Vi.at<double>(i,0);
      Ui.at<double>(i,0) = eps[i] * centered_pts[i].x;
      Vi.at<double>(i,0) = eps[i] * centered_pts[i].y;
      deltaUi[i] = Ui.at<double>(i,0) - oldUi[i];
      deltaVi[i] = Vi.at<double>(i,0) - oldVi[i];
    }
    double delta = 0.0;
    for (int i=0; i < num_landmarks; i++)
      delta += deltaUi[i] * deltaUi[i] + deltaVi[i] * deltaVi[i];
    delta = delta*focal_length*focal_length;
    if ((iter > 0) and (delta < 0.01)) // converged
      break;
  }
  /// Return rotation and translation matrices
  rot_matrix = (cv::Mat_<float>(3,3) << static_cast<float>(r1[0]),static_cast<float>(r1[1]),static_cast<float>(r1[2]),
                                        static_cast<float>(r2[0]),static_cast<float>(r2[1]),static_cast<float>(r2[2]),
                                        static_cast<float>(r3[0]),static_cast<float>(r3[1]),static_cast<float>(r3[2]));
  trl_matrix = (cv::Mat_<float>(3,1) << static_cast<float>(Tx), static_cast<float>(Ty), static_cast<float>(Tz));
}

cv::Vec3d
ModernPosit::getEulerAngles
  (
  const cv::Mat &rot_matrix,
  const cv::Mat &trl_matrix
  )
{
  cv::Mat rotate_coord_system, matrix;
  rotate_coord_system = (cv::Mat_<float>(4,4) << 0,0,-1,0, -1,0,0,0, 0,1,0,0, 0,0,0,1);
  matrix = (cv::Mat_<float>(4,4) << rot_matrix.at<float>(0,0),rot_matrix.at<float>(0,1),rot_matrix.at<float>(0,2),trl_matrix.at<float>(0,0),
                                    rot_matrix.at<float>(1,0),rot_matrix.at<float>(1,1),rot_matrix.at<float>(1,2),trl_matrix.at<float>(1,0),
                                    rot_matrix.at<float>(2,0),rot_matrix.at<float>(2,1),rot_matrix.at<float>(2,2),trl_matrix.at<float>(2,0),
                                    0,0,0,1);
  matrix = rotate_coord_system * matrix;

  const double a11 = matrix.at<float>(0,0), a12 = matrix.at<float>(0,1), a13 = matrix.at<float>(0,2);
  const double a21 = matrix.at<float>(1,0), a22 = matrix.at<float>(1,1), a23 = matrix.at<float>(1,2);
  const double a31 = matrix.at<float>(2,0), a32 = matrix.at<float>(2,1), a33 = matrix.at<float>(2,2);

  double roll, pitch, yaw;
  if (fabs(1.0 - a31) <= DBL_EPSILON) // special case a31 == +1
  {
    UPM_PRINT("Gimbal lock case a31 == " << a31);
    pitch = -M_PI_2;
    yaw   = M_PI_4; // arbitrary value
    roll  = atan2(a12,a13) - yaw;
  }
  else if (fabs(-1.0 - a31) <= DBL_EPSILON) // special case a31 == -1
  {
    UPM_PRINT("Gimbal lock case a31 == " << a31);
    pitch = M_PI_2;
    yaw   = M_PI_4; // arbitrary value
    roll  = atan2(a12,a13) + yaw;
  }
  else // standard case a31 != +/-1
  {
    pitch = asin(-a31);
    /// Two cases depending on where pitch angle lies
    if ((pitch < M_PI_2) and (pitch > -M_PI_2))
    {
      roll = atan2(a32,a33);
      yaw  = atan2(a21,a11);
    }
    else if ((pitch < 3.0*M_PI_2) and (pitch > M_PI_2))
    {
      roll = atan2(-a32,-a33);
      yaw  = atan2(-a21,-a11);
    }
    else
    {
      UPM_ERROR("This should never happen in pitch-roll-yaw computation");
      roll = 2.0 * M_PI;
      yaw  = 2.0 * M_PI;
    }
  }
  /// Convert to degrees
  return cv::Vec3d(-yaw, pitch, -roll) * (180.0/M_PI);
}

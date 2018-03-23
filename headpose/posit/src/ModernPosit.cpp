/*
 * ModernPosit.cpp
 *
 *  Created on: 22.10.2010
 *      Author: pwohlhart
 *
 *
 *  Derived from Matlab implementation http://www.cfar.umd.edu/~daniel/modernPosit.m
 *
 */

#include "ModernPosit.h"
#include <iostream>

cv::Mat
ModernPosit::run
  (
  const std::vector<cv::Point3f> &world_pts,
  const std::vector<cv::Point2f> &image_pts,
  const cv::Mat &cam_matrix,
  int max_iters
  )
{
  // Normalize image points
  float focal_length = cam_matrix.at<float>(0,0);
  cv::Point2f img_center = cv::Point2f(cam_matrix.at<float>(0,2),cam_matrix.at<float>(1,2));
  std::vector<cv::Point2f> centered_pts;
  for (std::vector<cv::Point2f>::const_iterator it = image_pts.begin(); it != image_pts.end(); ++it)
    centered_pts.push_back(*it - img_center);
  for (std::vector<cv::Point2f>::iterator  it = centered_pts.begin(); it != centered_pts.end(); ++it)
    (*it) *= 1.0f/focal_length;

  unsigned int npoints = static_cast<unsigned int>(image_pts.size());
  std::vector<double> ui(npoints), vi(npoints);
  for (int i=0; i < npoints; ++i)
  {
    ui[i] = centered_pts[i].x;
    vi[i] = centered_pts[i].y;
  }

  cv::Mat homogeneousWorldPts(npoints,4,CV_32F);
  for (int i=0; i < npoints; ++i)
  {
    homogeneousWorldPts.at<float>(i,0) = world_pts[i].x;
    homogeneousWorldPts.at<float>(i,1) = world_pts[i].y;
    homogeneousWorldPts.at<float>(i,2) = world_pts[i].z;
    homogeneousWorldPts.at<float>(i,3) = 1; // homogeneous
  }
  cv::Mat objectMat;
  cv::invert(homogeneousWorldPts, objectMat, cv::DECOMP_SVD);

  double Tx = 0.0, Ty = 0.0, Tz = 0.0, r1T[4], r2T[4], r1N[4], r2N[4], r3[4];
  std::vector<double> oldUi(npoints), oldVi(npoints), deltaUi(npoints), deltaVi(npoints);
  bool converged = false;
  int iterationCount = 0;
  while ((not converged) and ((max_iters < 0 ) or (iterationCount < max_iters)))
  {
    for (int j=0; j < 4; ++j)
    {
      r1T[j] = 0;
      r2T[j] = 0;
      for (int i=0; i < npoints; ++i)
      {
        r1T[j] += ui[i] * objectMat.at<float>(j,i);
        r2T[j] += vi[i] * objectMat.at<float>(j,i);
      }
    }

    double Tz1, Tz2;
    Tz1 = 1/sqrt(r1T[0]*r1T[0] + r1T[1]*r1T[1]+ r1T[2]*r1T[2]);
    Tz2 = 1/sqrt(r2T[0]*r2T[0] + r2T[1]*r2T[1]+ r2T[2]*r2T[2]);

    Tz = sqrt(Tz1*Tz2); // geometric average instead of arithmetic average of classicPosit

    for (int j=0; j < 4; ++j)
    {
      r1N[j] = r1T[j]*Tz;
      r2N[j] = r2T[j]*Tz;
    }

    for (int j=0; j < 3; ++j)
    {
      if ((r1N[j] > 1.0) or (r1N[j] < -1.0))
        r1N[j] = std::max(-1.0,std::min(1.0,r1N[j]));
      if ((r2N[j] > 1.0) or (r2N[j] < -1.0))
        r2N[j] = std::max(-1.0,std::min(1.0,r2N[j]));
    }

    r3[0] = r1N[1]*r2N[2] - r1N[2]*r2N[1];
    r3[1] = r1N[2]*r2N[0] - r1N[0]*r2N[2];
    r3[2] = r1N[0]*r2N[1] - r1N[1]*r2N[0];
    r3[3] = Tz;

    Tx = r1N[3];
    Ty = r2N[3];

    std::vector<double> wi(npoints);
    for (int i=0; i < npoints; ++i)
    {
      wi[i] = 0;
      for (int j=0; j < 4; ++j)
        wi[i] += homogeneousWorldPts.at<float>(i,j) * r3[j] / Tz;
    }

    for (int i=0; i < npoints; ++i)
    {
      oldUi[i] = ui[i];
      oldVi[i] = vi[i];
      ui[i] = wi[i] * centered_pts[i].x;
      vi[i] = wi[i] * centered_pts[i].y;
      deltaUi[i] = ui[i] - oldUi[i];
      deltaVi[i] = vi[i] - oldVi[i];
    }

    double delta = 0.0;
    for (int i=0; i < npoints; ++i)
      delta += deltaUi[i] * deltaUi[i] + deltaVi[i] * deltaVi[i];
    delta = delta*focal_length*focal_length;

    converged = (iterationCount > 0) && (delta < 0.01);
    ++iterationCount;
  }
  // Compute rotation matrix
  cv::Mat rot_matrix = (cv::Mat_<float>(3,4) << 0,0,0,Tx, 0,0,0,Ty, 0,0,0,Tz);
  for (int i=0; i < 3; i++)
  {
    rot_matrix.at<float>(0,i) = static_cast<float>(r1N[i]);
    rot_matrix.at<float>(1,i) = static_cast<float>(r2N[i]);
    rot_matrix.at<float>(2,i) = static_cast<float>(r3[i]);
  }
  return rot_matrix;
}

cv::Vec3d
ModernPosit::getEulerAngles
  (
  cv::Mat &rot_matrix
  )
{
  cv::Mat rotate_coord_system = (cv::Mat_<float>(4,4) << 0,0,-1,0, -1,0,0,0, 0,1,0,0, 0,0,0,1);
  rot_matrix = (cv::Mat_<float>(4,4) << rot_matrix.at<float>(0,0),rot_matrix.at<float>(0,1),rot_matrix.at<float>(0,2),rot_matrix.at<float>(0,3), rot_matrix.at<float>(1,0),rot_matrix.at<float>(1,1),rot_matrix.at<float>(1,2),rot_matrix.at<float>(1,3), rot_matrix.at<float>(2,0),rot_matrix.at<float>(2,1),rot_matrix.at<float>(2,2),rot_matrix.at<float>(2,3), 0,0,0,1);
  rot_matrix = rotate_coord_system * rot_matrix;

  const double a11 = rot_matrix.at<float>(0,0), a12 = rot_matrix.at<float>(0,1), a13 = rot_matrix.at<float>(0,2);
  const double a21 = rot_matrix.at<float>(1,0), a22 = rot_matrix.at<float>(1,1), a23 = rot_matrix.at<float>(1,2);
  const double a31 = rot_matrix.at<float>(2,0), a32 = rot_matrix.at<float>(2,1), a33 = rot_matrix.at<float>(2,2);

  double roll, pitch, yaw;
  if (fabs(1.0 - a31) <= DBL_EPSILON) // special case a31 == +1
  {
    std::cout << "Gimbal lock case a31 == " << a31 << std::endl;
    pitch = -M_PI_2;
    yaw   = M_PI_4; // arbitrary value
    roll  = atan2(a12,a13) - yaw;
  }
  else if (fabs(-1.0 - a31) <= DBL_EPSILON) // special case a31 == -1
  {
    std::cout << "Gimbal lock case a31 == " << a31 << std::endl;
    pitch = M_PI_2;
    yaw   = M_PI_4; // arbitrary value
    roll  = atan2(a12,a13) + yaw;
  }
  else // standard case a31 != +/-1
  {
    pitch = asin(-a31);
    // Two cases depending on where pitch angle lies
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
      std::cerr << "This should never happen in pitch-roll-yaw computation" << std::endl;
      roll = 2.0 * M_PI;
      yaw  = 2.0 * M_PI;
    }
  }
  // Convert to degrees
  return cv::Vec3d(-yaw, pitch, -roll) * (180.0/M_PI);
}

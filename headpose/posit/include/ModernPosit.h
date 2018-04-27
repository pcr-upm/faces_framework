/** ****************************************************************************
 *  @file    ModernPosit.h
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

#ifndef MODERN_POSIT_H
#define MODERN_POSIT_H

#include <FaceAnnotation.hpp>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * POSIT is a fast iterative algorithm for finding the pose (rotation and translation)
 * of an object or scene with respect to a camera when points of the object are given
 * in some object coordinate system and these points are visible in the camera image
 * and recognizable, so that corresponding image points and object points can be listed
 * in the same order.
 */
class ModernPosit
{
public:
  ModernPosit() {};

  virtual
  ~ModernPosit() {};

  static void
  setCorrespondences
    (
    const std::string &path,
    const upm::FaceAnnotation &ann,
    const unsigned int &num_landmarks,
    std::vector<cv::Point3f> &world_all,
    std::vector<cv::Point3f> &world_pts,
    std::vector<cv::Point2f> &image_pts,
    const std::vector<int> &mask = {}
    );

  static void
  run
    (
    const std::vector<cv::Point3f> &world_pts,
    const std::vector<cv::Point2f> &image_pts,
    const cv::Mat &cam_matrix,
    const int &max_iters,
    cv::Mat &rot_matrix,
    cv::Mat &trl_matrix
    );

  static cv::Vec3d
  getEulerAngles
    (
    const cv::Mat &rot_matrix,
    const cv::Mat &trl_matrix
    );
};

#endif /* MODERN_POSIT_H */

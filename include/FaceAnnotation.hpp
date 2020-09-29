/** ****************************************************************************
 *  @file    FaceAnnotation.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ANNOTATION_HPP
#define FACE_ANNOTATION_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace upm {

struct FaceBox
{
  bool operator==(const FaceBox &src) const
  {
    return ((detector_idx==src.detector_idx) and (pos==src.pos) and (score==src.score));
  };
  unsigned int detector_idx;
  cv::Rect_<float> pos;
  float score;
};

struct FaceLandmark
{
  bool operator==(const FaceLandmark &src) const
  {
    return ((feature_idx==src.feature_idx) and (pos==src.pos) and (occluded==src.occluded));
  };
  unsigned int feature_idx;
  cv::Point2f pos;
  float occluded;
};

enum FacePartLabel { leyebrow, reyebrow, leye, reye, nose, tmouth, bmouth, lear, rear, chin };
struct FacePart
{
  bool operator==(const FacePart &src) const
  {
    return ((label==src.label) and (landmarks==src.landmarks));
  };
  FacePartLabel label;
  std::vector<FaceLandmark> landmarks;
};

struct FaceAttribute
{
  bool operator==(const FaceAttribute &src) const
  {
    return ((male==src.male) and (age==src.age) and (glasses==src.glasses) and (hat==src.hat) and (moustache==src.moustache) and (beard==src.beard));
  };
  float male;
  float age;
  float glasses;
  float hat;
  float moustache;
  float beard;
  float fake;
};

/** ****************************************************************************
 * @class FaceAnnotation
 * @brief Class used to save face objects ground truth.
 ******************************************************************************/
class FaceAnnotation
{
public:
  FaceAnnotation() :
    filename(""),
    bbox({0,cv::Rect_<float>(-1.0f,-1.0f,-1.0f,-1.0f),0.0f}),
    headpose(cv::Point3f(-FLT_MAX,-FLT_MAX,-FLT_MAX)),
    parts({{leyebrow,{}}, {reyebrow,{}}, {leye,{}}, {reye,{}}, {nose,{}}, {tmouth,{}}, {bmouth,{}}, {lear,{}}, {rear,{}}, {chin,{}}}),
    attribute({0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}) {};

  std::string filename;
  FaceBox bbox;
  cv::Point3f headpose;
  std::vector<FacePart> parts;
  FaceAttribute attribute;
};

} // namespace upm

#endif /* FACE_ANNOTATION_HPP */

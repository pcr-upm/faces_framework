/** ****************************************************************************
 *  @file    FaceAlignment.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_HPP
#define FACE_ALIGNMENT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <FaceComponent.hpp>
#include <FaceAnnotation.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

enum class ErrorMeasure { pupils, corners, height };

/** ****************************************************************************
 * @class FaceAlignment
 * @brief Class interface for facial feature point detection.
 ******************************************************************************/
class FaceAlignment : public FaceComponent
{
public:
  FaceAlignment() : FaceComponent(2) {};

  virtual
  ~FaceAlignment() {};

  virtual void
  parseOptions
    (
    int argc,
    char **argv
    );

  virtual void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
    ) = 0;

  virtual void
  load() = 0;

  virtual void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    ) = 0;

  void
  show
    (
    const boost::shared_ptr<Viewer> &viewer,
    const std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

  void
  evaluate
    (
    boost::shared_ptr<std::ostream> output,
    const std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

  void
  save
    (
    const std::string dirpath,
    const std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

  ErrorMeasure _measure;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_HPP */

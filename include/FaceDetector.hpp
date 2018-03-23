/** ****************************************************************************
 *  @file    FaceDetector.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <FaceComponent.hpp>
#include <FaceAnnotation.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceDetector
 * @brief Class interface for unconstrained face detection.
 ******************************************************************************/
class FaceDetector : public FaceComponent
{
public:
  FaceDetector
    () : FaceComponent(0) {};

  virtual
  ~FaceDetector() {};

  virtual void
  parseOptions
    (
    int argc,
    char **argv
    ) = 0;

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
    std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    ) = 0;

  void
  show
    (
    const boost::shared_ptr<upm::Viewer> &viewer,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    );

  void
  evaluate
    (
    boost::shared_ptr<std::ostream> output,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    );

  void
  save
    (
    const std::string dirpath,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    );
};

} // namespace upm

#endif /* FACE_DETECTOR_HPP */

/** ****************************************************************************
 *  @file    FaceComponent.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_COMPONENT_HPP
#define FACE_COMPONENT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <FaceAnnotation.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceComponent
 * @brief Class face component interface.
 ******************************************************************************/
class FaceComponent
{
public:
  FaceComponent
    (
    unsigned int part
    ) : m_part(part) {};

  virtual
  ~FaceComponent() {};

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

  virtual void
  show
    (
    const boost::shared_ptr<upm::Viewer> &viewer,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    ) = 0;

  virtual void
  evaluate
    (
    boost::shared_ptr<std::ostream> output,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    ) = 0;

  virtual void
  save
    (
    const std::string dirpath,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    ) = 0;

  unsigned int
  getComponentClass() { return m_part; };

private:
  unsigned int m_part; // 1-FaceDetector, 2-FaceHeadPose, 3-FaceAlignment
};

} // namespace upm

#endif /* FACE_COMPONENT_HPP */

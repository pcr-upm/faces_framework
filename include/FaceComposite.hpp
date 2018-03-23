/** ****************************************************************************
 *  @file    FaceComposite.hpp
 *  @brief   Composition of one-or-more FaceComponent objects
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_COMPOSITE_HPP
#define FACE_COMPOSITE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <FaceComponent.hpp>
#include <FaceAnnotation.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceComposite
 * @brief Composition of one-or-more similar objects.
 ******************************************************************************/
class FaceComposite : public FaceComponent
{
public:
  FaceComposite() : FaceComponent(0) {};

  ~FaceComposite() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->parseOptions(argc, argv);
  };

  void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->train(anns_train, anns_valid);
  };

  void
  load()
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->load();
  };

  void
  process
    (
    cv::Mat frame,
    std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->process(frame, faces, ann);
  };

  void
  show
    (
    const boost::shared_ptr<upm::Viewer> &viewer,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->show(viewer, faces, ann);
  };

  void
  evaluate
    (
    boost::shared_ptr<std::ostream> output,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->evaluate(output, faces, ann);
  };

  void
  save
    (
    const std::string dirpath,
    const std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      m_components[i]->save(dirpath, faces, ann);
  };

  void
  addComponent
    (
    boost::shared_ptr<upm::FaceComponent> component
    )
  {
    m_components.push_back(component);
  };

  bool
  containsPart
    (
    unsigned int part
    )
  {
    for (unsigned int i=0; i < m_components.size(); i++)
      if (m_components[i]->getComponentClass() == part)
        return true;
    return false;
  };

private:
  std::vector< boost::shared_ptr<upm::FaceComponent> > m_components;
};

} // namespace upm

#endif /* FACE_COMPOSITE_HPP */

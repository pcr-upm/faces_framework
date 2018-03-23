/** ****************************************************************************
 *  @file    FaceRecognition.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <FaceRecognition.hpp>
#include <numeric>

namespace upm {

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
FaceRecognition::show
  (
  const boost::shared_ptr<upm::Viewer> &viewer,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{

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
FaceRecognition::evaluate
  (
  boost::shared_ptr<std::ostream> output,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{

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
FaceRecognition::save
  (
  const std::string dirpath,
  const std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{

};

} // namespace upm

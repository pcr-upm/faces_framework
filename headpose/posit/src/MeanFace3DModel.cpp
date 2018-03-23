/** ****************************************************************************
 *  @file    MeanFace3DModel.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

#include <MeanFace3DModel.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

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
MeanFace3DModel::load
  (
  std::string filename
  )
{
  std::fstream ifs(filename, std::ios::in);
  std::string line;
  while (std::getline(ifs, line))
  {
    std::vector<std::string> data;
    boost::split(data, line, boost::is_any_of("|"));
    cv::Point3f pt;
    int id = boost::lexical_cast<int>(data[0]);
    pt.x = boost::lexical_cast<float>(data[1]);
    pt.y = boost::lexical_cast<float>(data[2]);
    pt.z = boost::lexical_cast<float>(data[3]);
    _coordinatesById[id] = pt;
  }
  ifs.close();
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
MeanFace3DModel::getCoordinatesById
  (
  int id
  ) const
{
  std::map<int,cv::Point3f>::const_iterator it = _coordinatesById.find(id);
  if (it == _coordinatesById.end())
    return cv::Point3f(0.0f, 0.0f, 0.0f);
  return it->second;
};
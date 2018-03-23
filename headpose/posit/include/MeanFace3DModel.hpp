/** ****************************************************************************
 *  @file    MeanFace3DModel.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

#ifndef MEAN_FACE_3D_MODEL_HPP
#define MEAN_FACE_3D_MODEL_HPP

#include <map>
#include <string>
#include <opencv2/opencv.hpp>

enum FaceCenterTypes { CENTER_BETWEEN_EYES, CENTER_FOR_ELLIPSE };

class MeanFace3DModel
{
public:
  MeanFace3DModel() {};

  virtual
  ~MeanFace3DModel() {};

  void
  load
    (
    std::string filename
    );

  cv::Point3f
  getCoordinatesById
    (
    int id
    ) const;

private:
  std::map<int,cv::Point3f> _coordinatesById;
};

#endif /* MEAN_FACE_3D_MODEL_HPP */

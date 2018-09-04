/** ****************************************************************************
 *  @file    FaceAlignment.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <FaceAlignment.hpp>
#include <numeric>
#include <boost/program_options.hpp>

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
FaceAlignment::parseOptions
  (
  int argc,
  char **argv
  )
{
  // Declare the supported program options
  namespace po = boost::program_options;
  po::options_description desc("FaceAlignment options");
  desc.add_options()
    ("measure", po::value<std::string>()->default_value("height"), "Select measure [pupils, corners, height]")
    ("database", po::value<std::string>()->default_value("aflw"), "Choose database [300w_public, 300w_private, cofw, aflw, menpo, wflw]");
  UPM_PRINT(desc);

  // Process the command line parameters
  po::variables_map vm;
  po::command_line_parser parser(argc, argv);
  parser.options(desc);
  const po::parsed_options parsed_opt(parser.allow_unregistered().run());
  po::store(parsed_opt, vm);
  po::notify(vm);

  std::string measure;
  if (vm.count("measure"))
    measure = vm["measure"].as<std::string>();
  _measure = (measure == "pupils") ? ErrorMeasure::pupils : (measure == "corners") ? ErrorMeasure::corners : ErrorMeasure::height;

  if (vm.count("database"))
    _database = vm["database"].as<std::string>();
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
FaceAlignment::show
  (
  const boost::shared_ptr<Viewer> &viewer,
  const std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  const int radius = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  const int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.005f)), 2);

  /// Ground truth
  cv::Scalar cyan_color(255,122,0), blue_color(255,0,0);
  for (const FacePart &ann_part : ann.parts)
    for (auto it=ann_part.landmarks.begin(), next=std::next(it); it < ann_part.landmarks.end(); it++, next++)
    {
      if (next != ann_part.landmarks.end())
        viewer->line((*it).pos.x, (*it).pos.y, (*next).pos.x, (*next).pos.y, thickness, cyan_color);
      viewer->circle((*it).pos.x, (*it).pos.y, radius, -1, (*it).visible ? cyan_color : blue_color);
    }

  /// Detected landmarks
  cv::Scalar green_color(0,255,0), red_color(0,0,255);
  for (const FaceAnnotation &face : faces)
    for (const FacePart &face_part : face.parts)
      for (auto it=face_part.landmarks.begin(), next=std::next(it); it < face_part.landmarks.end(); it++, next++)
      {
        if (next != face_part.landmarks.end())
          viewer->line((*it).pos.x, (*it).pos.y, (*next).pos.x, (*next).pos.y, thickness, green_color);
        viewer->circle((*it).pos.x, (*it).pos.y, radius, -1, (*it).visible ? green_color : red_color);
      }
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
FaceAlignment::evaluate
  (
  boost::shared_ptr<std::ostream> output,
  const std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Estimate normalized error and occlusion for each visible landmark
  unsigned int tp = 0, fp = 0, fn = 0, tn = 0;
  for (const FaceAnnotation &face : faces)
  {
    *output << getComponentClass() << " " << ann.filename;
    for (const FacePart &face_part : face.parts)
      for (const FaceLandmark &face_landmark : face_part.landmarks)
      {
        unsigned int idx = face_landmark.feature_idx;
        bool is_visible = false;
        for (const FacePart &ann_part : ann.parts)
        {
          auto found = std::find_if(ann_part.landmarks.begin(), ann_part.landmarks.end(), [&idx](const FaceLandmark &obj){return obj.feature_idx == idx;});
          if (found != ann_part.landmarks.end())
          {
            is_visible = (*found).visible;
            break;
          }
        }
        not face_landmark.visible ? not is_visible ? tp++ : fp++ : not is_visible ? fn++ : tn++;
      }
    *output << " " << tp << " " << fp << " " << fn << " " << tn;
    std::vector<unsigned int> indices;
    std::vector<float> errors;
    getNormalizedErrors(face, ann, _measure, indices, errors);
    for (unsigned int j=0; j < errors.size(); j++)
      *output << " " << indices[j] << " " << errors[j];
    *output << std::endl;
  }
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
FaceAlignment::save
  (
  const std::string dirpath,
  const std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Save images with mean error greater than threshold
  const float threshold = 8.0f;
  const int radius = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.01f)), 3);
  const int thickness = MAX(static_cast<int>(roundf(ann.bbox.pos.height*0.005f)), 2);
  cv::Scalar cyan_color(255,122,0), blue_color(255,0,0), green_color(0,255,0), red_color(0,0,255);
  cv::Mat image = cv::imread(ann.filename, CV_LOAD_IMAGE_COLOR);
  if (_database != "menpo")
  {
    for (const FacePart &ann_part : ann.parts)
      for (auto it=ann_part.landmarks.begin(), next=std::next(it); it < ann_part.landmarks.end(); it++, next++)
      {
        if (next != ann_part.landmarks.end())
          cv::line(image, (*it).pos, (*next).pos, cyan_color, thickness);
        cv::circle(image, (*it).pos, radius, (*it).visible ? cyan_color : blue_color, -1);
      }
    for (const FaceAnnotation &face : faces)
    {
      for (const FacePart &face_part : face.parts)
        for (auto it=face_part.landmarks.begin(), next=std::next(it); it < face_part.landmarks.end(); it++, next++)
        {
          if (next != face_part.landmarks.end())
            cv::line(image, (*it).pos, (*next).pos, green_color, thickness);
          cv::circle(image, (*it).pos, radius, (*it).visible ? green_color : red_color, -1);
        }
      std::vector<unsigned int> indices;
      std::vector<float> errors;
      getNormalizedErrors(face, ann, _measure, indices, errors);
      float err = static_cast<float>(std::accumulate(errors.begin(),errors.end(),0.0)) / static_cast<float>(errors.size());
      std::string text = std::to_string(err);
      cv::putText(image, text, cv::Point(10, image.rows-10), cv::FONT_HERSHEY_SIMPLEX, 1, red_color);
      if (err > threshold)
      {
        std::size_t found = face.filename.find_last_of("/");
        std::string filepath = dirpath + face.filename.substr(found+1);
        cv::imwrite(filepath, image);
      }
    }
  }
  else
  {
    // Save points file
    for (const FaceAnnotation &face : faces)
    {
      std::size_t found = face.filename.substr(0,face.filename.find_last_of("/")).find_last_of("/");
      std::string filepath = face.filename.substr(found+1);
      std::string pose = filepath.substr(0,filepath.find_last_of("/"));
      std::vector<unsigned int> menpo_landmarks;
      if (pose == "semifrontal")
        menpo_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
      else if (getHeadpose(face).x > 0)
        menpo_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 1, 119, 2, 121, 128, 129, 130, 17, 133, 16, 139, 138, 7, 142, 141, 22, 151, 150, 20, 160, 159, 23, 163, 162, 161, 168, 167};
      else
        menpo_landmarks = {117, 116, 115, 114, 113, 112, 111, 110, 24, 108, 107, 106, 6, 126, 5, 124, 128, 129, 130, 17, 135, 18, 144, 145, 12, 147, 148, 22, 153, 154, 21, 156, 157, 23, 163, 164, 165, 166, 167};
      std::ofstream ofs("output/err/points/" + filepath.substr(0,filepath.size()-3) + "pts");
//      ofs << "output/err/points/" + filepath.substr(0,filepath.size()-3) + "pts" << std::endl;
      ofs << "version: 1" << std::endl;
      ofs << "n_points: " << menpo_landmarks.size() << std::endl;
      ofs << "{" << std::endl;
//      cv::rectangle(image, face.bbox.pos.tl(), face.bbox.pos.br(), cv::Scalar(0,255,0), thickness);
      for (const unsigned int idx: menpo_landmarks)
        for (const FacePart &face_part : face.parts)
          for (auto it=face_part.landmarks.begin(), next=std::next(it); it < face_part.landmarks.end(); it++, next++)
            if (idx == (*it).feature_idx)
            {
              if (next != face_part.landmarks.end())
                cv::line(image, (*it).pos, (*next).pos, green_color, thickness);
              cv::circle(image, (*it).pos, radius, (*it).visible ? green_color : red_color, -1);
              ofs << (*it).pos.x << " " << (*it).pos.y << std::endl;
            }
      ofs << "}" << std::endl;
      ofs.close();
      cv::imwrite(dirpath + filepath, image);
    }
  }
};

} // namespace upm

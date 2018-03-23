/** ****************************************************************************
 *  @file    Viewer.cpp
 *  @brief   Class viewer interface implementation
 *  @author  Roberto Valle Fernandez
 *  @date    2015/02
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace upm {

// -----------------------------------------------------------------------------
//
// Purpose and Method: Constructor
// Inputs: 
// Outputs: 
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
Viewer::Viewer
  (): 
  m_drawing(false), m_initialised(false), m_width(-1), m_height(-1) {};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs: 
// Outputs: 
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
Viewer::~Viewer
  ()
{
  if (m_initialised)
    cv::destroyWindow(m_window_title.c_str());

  m_initialised = false;
  m_drawing     = false;
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
Viewer::init
  (
  int width,
  int height,
  std::string window_title
  )
{
  if (m_initialised)
    cv::destroyWindow(m_window_title.c_str());

  m_canvas       = cv::Mat(cv::Size(width,height), CV_8UC3);
  m_initialised  = true;
  m_window_title = window_title;
  m_width        = width;
  m_height       = height;

  cv::namedWindow(m_window_title.c_str(), cv::WINDOW_AUTOSIZE);
  cv::moveWindow(m_window_title.c_str(), 0, 0);
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
Viewer::resizeCanvas
  (
  int width,
  int height
  )
{
  if ((m_initialised) && ((m_width != width) || (m_height != height)))
  {
    m_canvas = cv::Mat(cv::Size(width,height), CV_8UC3);
    m_width  = width;
    m_height = height;
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
Viewer::beginDrawing
  ()
{
  if (m_initialised && !m_drawing)
  {
    // Fill drawing canvas on black
    cv::Point rectangle[1][4];
    rectangle[0][0] = cv::Point(0, 0);
    rectangle[0][1] = cv::Point(m_width, 0);
    rectangle[0][2] = cv::Point(m_width, m_height);
    rectangle[0][3] = cv::Point(0, m_height);

    const cv::Point *curve_arr[1] = {rectangle[0]};
    int vertices[] = {4};
    int polygons   = 1;
    cv::fillPoly(m_canvas, curve_arr, vertices, polygons, cv::Scalar(0, 0, 0));

    m_drawing = true;
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
Viewer::endDrawing
  (
  int delay
  )
{
  if (m_initialised && m_drawing)
  {
    cv::imshow(m_window_title.c_str(), m_canvas);
    cv::waitKey(delay);

    m_drawing = false;
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
Viewer::rectangle
  (
  int x, 
  int y, 
  int width, 
  int height, 
  int line_width,
  cv::Scalar color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  cv::rectangle(m_canvas, cv::Point(x,y), cv::Point(x+width, y+height), color, line_width);
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
Viewer::filled_rectangle
  (
  int x, 
  int y, 
  int width, 
  int height, 
  cv::Scalar fill_color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  cv::Point rectangle[1][4];
  rectangle[0][0] = cv::Point(x, y);
  rectangle[0][1] = cv::Point(x+width, y);
  rectangle[0][2] = cv::Point(x+width, y+height);
  rectangle[0][3] = cv::Point(x, y+height);
  const cv::Point *curve_arr[1] = {rectangle[0]};
  int vertices[] = {4};
  int polygons = 1;

  cv::fillPoly(m_canvas, curve_arr, vertices, polygons, fill_color);
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
Viewer::line
  (
  int x1, 
  int y1, 
  int x2, 
  int y2, 
  int line_width,
  cv::Scalar color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  cv::line(m_canvas, cv::Point(x1,y1), cv::Point(x2,y2), color, line_width);
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
Viewer::circle
  (
  int x_center,
  int y_center,
  int radius,
  int line_width,
  cv::Scalar color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  cv::circle(m_canvas, cv::Point(x_center,y_center), radius, color, line_width);
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
Viewer::ellipse
  (
  int major_axis_length,
  int minor_axis_length,
  float angle,
  int x_center,
  int y_center,
  int line_width,
  cv::Scalar color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  double fi, majorCos, minorSin;
  float sin_angle, cos_angle;
  const unsigned ELLIPSE_NUMBER_OF_POINTS = 100;
  cv::Point ellipse[1][ELLIPSE_NUMBER_OF_POINTS];

  for (int i=0; i < ELLIPSE_NUMBER_OF_POINTS; i++)
  {
    fi            = 2*M_PI*(i/static_cast<double>(ELLIPSE_NUMBER_OF_POINTS));
    majorCos      = major_axis_length*cos(fi);
    minorSin      = minor_axis_length*sin(fi);
    cos_angle     = cosf(angle);
    sin_angle     = sinf(angle);
    ellipse[0][i] = cv::Point(static_cast<int>(x_center+(cos_angle*majorCos)-(sin_angle*minorSin)),
                              static_cast<int>(y_center+(sin_angle*majorCos)+(cos_angle*minorSin)));
  }
  const cv::Point *curve_arr[1] = {ellipse[0]};
  int vertices[] = {ELLIPSE_NUMBER_OF_POINTS};
  int polygons = 1;

  cv::polylines(m_canvas, curve_arr, vertices, 1, true, color, line_width);
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
Viewer::filled_ellipse
  (
  int major_axis_length,
  int minor_axis_length,
  float angle,
  int x_center,
  int y_center,
  cv::Scalar color
  )
{
  if (!m_initialised || !m_drawing)
    return;

  double fi, majorCos, minorSin;
  float sin_angle, cos_angle;
  const unsigned ELLIPSE_NUMBER_OF_POINTS = 100;
  cv::Point ellipse[1][ELLIPSE_NUMBER_OF_POINTS];

  for (int i=0; i < ELLIPSE_NUMBER_OF_POINTS; i++)
  {
    fi            = 2*M_PI*(i/static_cast<double>(ELLIPSE_NUMBER_OF_POINTS));
    majorCos      = major_axis_length*cos(fi);
    minorSin      = minor_axis_length*sin(fi);
    cos_angle     = cosf(angle);
    sin_angle     = sinf(angle);
    ellipse[0][i] = cv::Point(static_cast<int>(x_center+(cos_angle*majorCos)-(sin_angle*minorSin)),
                              static_cast<int>(y_center+(sin_angle*majorCos)+(cos_angle*minorSin)));
  }
  const cv::Point *curve_arr[1] = {ellipse[0]};
  int vertices[] = {ELLIPSE_NUMBER_OF_POINTS};
  int polygons = 1;

  cv::fillPoly(m_canvas, curve_arr, vertices, polygons, color);
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
Viewer::text
  (
  std::string text, 
  int x, 
  int y, 
  cv::Scalar color,
  float scale,
  int line_width
  )
{
  if (!m_initialised || !m_drawing)
    return;

  cv::putText(m_canvas, text.c_str(), cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX, scale, color, line_width);
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
Viewer::image
  (
  cv::Mat frame,
  int x, 
  int y, 
  int width, 
  int height
  )
{
  if (!m_initialised || !m_drawing)
    return;

  // If the image does not exceed the limits we can display the whole image
  if (x+width > m_canvas.cols)
    width  = m_canvas.cols - x;

  if (y+height > m_canvas.rows)
    height = m_canvas.rows - y;

  // If the area for the image is very small we do nothing
  if ((width <= 0) || (height <= 0))
    return;

  // Check image dimensions coincide with 'width' and 'height'
  cv::Mat frame_aux;
  if ((frame.cols != width) || (frame.rows != height))
  {
    frame_aux = cv::Mat(cv::Size(width, height), frame.depth());
    cv::resize(frame, frame_aux, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
  }
  else
  {
    frame_aux = frame(cv::Rect(0,0,width, height));
  }

  // Join the input image with the canvas
  cv::Mat canvas_img;
  if (m_canvas.channels() != frame_aux.channels())
  {
    canvas_img = m_canvas(cv::Rect(x,y,width,height));
    frame_aux.copyTo(m_canvas);
  }
  else
  {
    canvas_img = m_canvas(cv::Rect(x,y,width,height));
    frame_aux.copyTo(canvas_img);
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
Viewer::saveCanvas
  (
  std::string path
  )
{
  cv::imwrite(path.c_str(), m_canvas);
};

} // namespace upm

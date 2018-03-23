/** ****************************************************************************
 *  @file    Viewer.hpp
 *  @brief   Class viewer interface definition
 *  @author  Roberto Valle Fernandez
 *  @date    2015/02
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef VIEWER_HPP
#define VIEWER_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <string>
#include <opencv2/highgui/highgui.hpp>

namespace upm {

/** ****************************************************************************
 * @class Viewer
 * @brief Class interface used by the computer vision algorithms to plot results.
 * The viewer implementation using OpenCV API (HighGUI). This class can be 
 * extended making a Viewer for GTK+, Win32, QT, etc. Just making a subclass
 * will do the work
 ******************************************************************************/
class Viewer
{
public:
  Viewer();

  virtual ~Viewer();

  /**
   *  @brief Initialize the viewer canvas using width and height in pixels
   *  @param width        The given width in pixels of the window canvas
   *  @param height       The given height in pixels of the window canvas
   *  @param window_title The string with the window title to display
   */
  virtual void
  init
    (
    int width,
    int height,
    std::string window_title
    );

  /**
   *  @brief Resize the viewer canvas using width and height in pixels
   */
  void
  resizeCanvas
    (
    int width,
    int height
    );

  /** 
   *  @brief All the drawing commands should be between "beginDrawing" and
   *  "endDrawing". Only after "endDrawing" the drawing commands result is
   *  displayed over the window canvas
   */
  virtual void
  beginDrawing();

  /** 
   *  @brief All the drawing commands should be between "beginDrawing" and
   *  "endDrawing". Only after "endDrawing" the drawing commands result is
   *  displayed over the window canvas
   *  @param delay Time in milliseconds. 0 is the value that means “forever”
   */
  virtual void
  endDrawing
    (
    int delay = 0
    );

  /** 
   *  @brief Draw a rectangle parallel to the viewer axes. The x axis runs
   *  from left to right of the window and the y axis is going from top to
   *  down of the window
   *  @param x          Horizontal coordinate of the top left corner
   *  @param y          Vertical coordinate of the top left corner
   *  @param width      Size of the window in the horizontal axis
   *  @param height     Size of the window in the vertical axis
   *  @param line_width Width in pixels of the rectangle border
   *  @param color      Pen color to draw with
   */
  virtual void 
  rectangle
    (
    int x, 
    int y, 
    int width, 
    int height, 
    int line_width,
    cv::Scalar color
    );

  /** 
   *  @brief Draw a color filled rectangle parallel to the viewer axes. The
   *  x axis runs from left to right of the window and the y axis is going
   *  from top to down of the window
   *  @param x      Horizontal coordinate of the top left corner
   *  @param y      Vertical coordinate of the top left corner
   *  @param width  Size of the window in the horizontal axis
   *  @param height Size of the window in the vertical axis
   *  @param color  Fill color
   */
  virtual void 
  filled_rectangle
    (
    int x, 
    int y, 
    int width, 
    int height, 
    cv::Scalar color
    );

  /** 
   *  @brief Draw a line on the viewer
   *  @param x1         Horizontal coordinate of the starting pixel
   *  @param y1         Vertical coordinate of the starting pixel
   *  @param x2         Horizontal coordinate of the ending pixel
   *  @param y2         Vertical coordinate of the ending pixel
   *  @param line_width Width in pixels of the line
   *  @param color      Line color
   */ 
  virtual void 
  line
    (
    int x1, 
    int y1, 
    int x2, 
    int y2, 
    int line_width,
    cv::Scalar color
    );

  /**
   * @brief Draw a general circle on the viewer
   * @param x_center   Horizontal pixel coordinate of the ellipse center
   * @param y_center   Vertical pixel coordinate of the ellipse center
   * @param radius     Radius of the circle
   * @param line_width Type of the circle boundary
   * @param color      Circle color
   */
  virtual void
  circle
    (
    int x_center,
    int y_center,
    int radius,
    int line_width,
    cv::Scalar color
    );

  /**
   *  @brief Draw a general ellipse on the viewer
   *  @param major_axis_length Major axis length in pixels
   *  @param minor_axis_length Minor axis length in pixels
   *  @param angle      Angle of the major axis with respect to the horizontal axis
   *  @param x_center   Horizontal pixel coordinate of the ellipse center
   *  @param y_center   Vertical pixel coordinate of the ellipse center
   *  @param line_width Width in pixels of the line
   *  @param color      Border color
   */
  virtual void
  ellipse
    (
    int major_axis_length,
    int minor_axis_length,
    float angle, /* with respect to the horizontal axis */
    int x_center, 
    int y_center, 
    int line_width,
    cv::Scalar color
    );

  /** 
   *  @brief Draw a general filled ellipse on the viewer
   *  @param major_axis_length Major axis length in pixels
   *  @param minor_axis_length Minor axis length in pixels
   *  @param angle    Angle of the major axis with respect to the horizontal axis
   *  @param x_center Horizontal pixel coordinate of the ellipse center
   *  @param y_center Vertical pixel coordinate of the ellipse center
   *  @param color    Fill color
   */
  virtual void
  filled_ellipse
    (
    int major_axis_length,
    int minor_axis_length,
    float angle, /* with respect to the horizontal axis */
    int x_center, 
    int y_center, 
    cv::Scalar color
    );

  /**
   * @brief Show text string at the viewer's (x,y) position
   * @param text  The text to put over the viewer current canvas
   * @param x     The text appear with the top left corner at x horizontal coordinate
   * @param y     The text will appear with the top left corner at y vertical coordinate
   * @param color Text color
   * @param scale Font scale factor that is multiplied by the font-specific base size
   * @param line_width Thickness of the lines used to draw a text
   */
  virtual void 
  text
    (
    std::string text, 
    int x, 
    int y, 
    cv::Scalar color,
    float scale = 1.0,
    int line_width = 1
    );

  /** 
   * @brief Draw an image over the background on the viewer
   * @param frame  Image to put over the viewer current canvas
   * @param x      The image appear with the top left corner at x horizontal coordinate
   * @param y      The image will appear with the top left corner at y vertical coordinate
   * @param width  The image width in pixels over the viewer (it could need to be scaled)
   * @param height The image height in pixels over the viewer (it could need to be scaled)
   */
  virtual void 
  image
    (
    cv::Mat frame,
    int x, 
    int y, 
    int width, 
    int height
    );

  /**
   * @brief Save canvas in a file
   * @param path The selected output image name
   */
  virtual void
  saveCanvas
    (
    std::string path
    );

protected:
  bool m_initialised;
  bool m_drawing;
  cv::Mat m_canvas;
  std::string m_window_title;
  int m_width;
  int m_height;
};

} // namespace upm

#endif /* VIEWER_HPP */

#------------------------------------------------------------------------------
# Test face alignment algorithms
#------------------------------------------------------------------------------
#-- Assert this CMakeLists.txt has not been called before
if(NOT TARGET faces_framework_test)
  project(faces_framework)
  cmake_minimum_required(VERSION 3.2)

  #-- Configuration specific to your own machine
  site_name(HOSTNAME)
  message(STATUS CURRENT_HOSTNAME=${HOSTNAME})

  #-- Type of build
  set(CMAKE_BUILD_TYPE Release)
  add_compile_options(-std=c++11)

  #-- Setup required libraries
  find_package(JPEG REQUIRED)
  message(STATUS JPEG_LIBRARIES=${JPEG_LIBRARIES})

  find_package(PNG REQUIRED)
  message(STATUS PNG_LIBRARIES=${PNG_LIBRARIES})

  find_package(OpenCV 3 REQUIRED)
  message(STATUS OPENCV_VERSION=${OpenCV_VERSION})
  message(STATUS OPENCV_LIBRARIES=${OpenCV_LIBS})
  message(STATUS OPENCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS})

  find_package(Boost REQUIRED COMPONENTS program_options serialization system filesystem thread REQUIRED)
  message(STATUS BOOST_LIBRARIES=${Boost_LIBRARIES})
  message(STATUS BOOST_INCLUDE_DIRS=${Boost_INCLUDE_DIR})

  #-- Find include files
  set(faces_framework_include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/headpose/posit/include/
  )

  set(faces_framework_src
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FaceDetector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FaceHeadPose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FaceAlignment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FaceRecognition.cpp
    ${CMAKE_CURRENT_LIST_DIR}/headpose/posit/src/MeanFace3DModel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/headpose/posit/src/ModernPosit.cpp
  )

  set(faces_framework_test
    ${CMAKE_CURRENT_LIST_DIR}/test/faces_framework_test.cpp
  )

  set(faces_framework_libs
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${PNG_LIBRARIES}
  )

  #-- Setup CMake to run tests
  enable_testing()

  include_directories(${faces_framework_include})

  foreach(test ${faces_framework_test})
    get_filename_component(test_name ${test} NAME_WE)
    add_executable(${test_name}
      ${faces_framework_src}
      ${test}
    )
    target_link_libraries(${test_name} ${faces_framework_libs})
    add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name})
  endforeach()
endif()
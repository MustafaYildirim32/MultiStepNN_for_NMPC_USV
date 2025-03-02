# Install script for directory: /home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs/msg" TYPE FILE FILES
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs/srv" TYPE FILE FILES
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs/cmake" TYPE FILE FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/build/uuv_gazebo_ros_plugins_msgs/catkin_generated/installspace/uuv_gazebo_ros_plugins_msgs-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/devel/include/uuv_gazebo_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/mustafa-yildirim/.matlab/R2024a/ros1/glnxa64/venv/bin/python3" -m compileall "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/devel/lib/python3/dist-packages/uuv_gazebo_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/devel/lib/python3/dist-packages/uuv_gazebo_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/build/uuv_gazebo_ros_plugins_msgs/catkin_generated/installspace/uuv_gazebo_ros_plugins_msgs.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs/cmake" TYPE FILE FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/build/uuv_gazebo_ros_plugins_msgs/catkin_generated/installspace/uuv_gazebo_ros_plugins_msgs-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs/cmake" TYPE FILE FILES
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/build/uuv_gazebo_ros_plugins_msgs/catkin_generated/installspace/uuv_gazebo_ros_plugins_msgsConfig.cmake"
    "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/build/uuv_gazebo_ros_plugins_msgs/catkin_generated/installspace/uuv_gazebo_ros_plugins_msgsConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_gazebo_ros_plugins_msgs" TYPE FILE FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/devel/lib/libuuv_gazebo_ros_plugins_msgs_matlab.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libuuv_gazebo_ros_plugins_msgs_matlab.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libuuv_gazebo_ros_plugins_msgs_matlab.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libuuv_gazebo_ros_plugins_msgs_matlab.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/m/" TYPE DIRECTORY FILES "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/m/" FILES_MATCHING REGEX "/[^/]*\\.m$")
endif()


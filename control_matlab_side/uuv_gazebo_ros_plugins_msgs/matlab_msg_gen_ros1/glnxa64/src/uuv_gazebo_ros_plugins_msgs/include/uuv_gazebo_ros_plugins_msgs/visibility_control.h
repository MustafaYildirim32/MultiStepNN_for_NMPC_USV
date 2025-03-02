#ifndef UUV_GAZEBO_ROS_PLUGINS_MSGS__VISIBILITY_CONTROL_H_
#define UUV_GAZEBO_ROS_PLUGINS_MSGS__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT __attribute__ ((dllexport))
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_IMPORT __attribute__ ((dllimport))
  #else
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT __declspec(dllexport)
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_IMPORT __declspec(dllimport)
  #endif
  #ifdef UUV_GAZEBO_ROS_PLUGINS_MSGS_BUILDING_LIBRARY
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT
  #else
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC UUV_GAZEBO_ROS_PLUGINS_MSGS_IMPORT
  #endif
  #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC_TYPE UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC
  #define UUV_GAZEBO_ROS_PLUGINS_MSGS_LOCAL
#else
  #define UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT __attribute__ ((visibility("default")))
  #define UUV_GAZEBO_ROS_PLUGINS_MSGS_IMPORT
  #if __GNUC__ >= 4
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC __attribute__ ((visibility("default")))
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC
    #define UUV_GAZEBO_ROS_PLUGINS_MSGS_LOCAL
  #endif
  #define UUV_GAZEBO_ROS_PLUGINS_MSGS_PUBLIC_TYPE
#endif
#endif  // UUV_GAZEBO_ROS_PLUGINS_MSGS__VISIBILITY_CONTROL_H_
// Generated 29-Jul-2024 19:46:26
// Copyright 2019-2020 The MathWorks, Inc.

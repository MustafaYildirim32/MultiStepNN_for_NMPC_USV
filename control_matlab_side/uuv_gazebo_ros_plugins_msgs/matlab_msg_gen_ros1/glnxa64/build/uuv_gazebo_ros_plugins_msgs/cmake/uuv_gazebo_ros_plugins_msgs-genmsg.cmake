# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "uuv_gazebo_ros_plugins_msgs: 3 messages, 10 services")

set(MSG_I_FLAGS "-Iuuv_gazebo_ros_plugins_msgs:/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg;-Istd_msgs:/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/std_msgs/cmake/../msg;-Igeometry_msgs:/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg;-Istd_msgs:/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(uuv_gazebo_ros_plugins_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg" "geometry_msgs/Vector3:geometry_msgs/Inertia"
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv" "uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel:geometry_msgs/Vector3:geometry_msgs/Inertia"
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv" "uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn"
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv" ""
)

get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv" NAME_WE)
add_custom_target(_uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uuv_gazebo_ros_plugins_msgs" "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv" ""
)

#
#  langs = gencpp;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg"
  "${MSG_I_FLAGS}"
  "/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_msg_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_msg_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg"
  "${MSG_I_FLAGS}"
  "/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Vector3.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Inertia.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)

### Generating Services
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv"
  "${MSG_I_FLAGS}"
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Vector3.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Inertia.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv"
  "${MSG_I_FLAGS}"
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_cpp(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)

### Generating Module File
_generate_module_cpp(uuv_gazebo_ros_plugins_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages uuv_gazebo_ros_plugins_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uuv_gazebo_ros_plugins_msgs_gencpp)
add_dependencies(uuv_gazebo_ros_plugins_msgs_gencpp uuv_gazebo_ros_plugins_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uuv_gazebo_ros_plugins_msgs_generate_messages_cpp)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg"
  "${MSG_I_FLAGS}"
  "/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_msg_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_msg_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg"
  "${MSG_I_FLAGS}"
  "/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Vector3.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Inertia.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)

### Generating Services
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv"
  "${MSG_I_FLAGS}"
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Vector3.msg;/usr/local/MATLAB/R2024a/sys/ros1/glnxa64/ros1/share/geometry_msgs/cmake/../msg/Inertia.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv"
  "${MSG_I_FLAGS}"
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)
_generate_srv_py(uuv_gazebo_ros_plugins_msgs
  "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
)

### Generating Module File
_generate_module_py(uuv_gazebo_ros_plugins_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(uuv_gazebo_ros_plugins_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages uuv_gazebo_ros_plugins_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/FloatStamped.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/ThrusterConversionFcn.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/msg/UnderwaterObjectModel.msg" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetFloat.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetListParam.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetModelProperties.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterConversionFcn.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterEfficiency.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/GetThrusterState.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetFloat.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterEfficiency.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetThrusterState.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/mustafa-yildirim/Documents/Repositories/heron_gazebo/uuv_gazebo_ros_plugins_msgs/matlab_msg_gen_ros1/glnxa64/src/uuv_gazebo_ros_plugins_msgs/srv/SetUseGlobalCurrentVel.srv" NAME_WE)
add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py _uuv_gazebo_ros_plugins_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uuv_gazebo_ros_plugins_msgs_genpy)
add_dependencies(uuv_gazebo_ros_plugins_msgs_genpy uuv_gazebo_ros_plugins_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uuv_gazebo_ros_plugins_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs)
  install(CODE "execute_process(COMMAND \"/home/mustafa-yildirim/.matlab/R2024a/ros1/glnxa64/venv/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uuv_gazebo_ros_plugins_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(uuv_gazebo_ros_plugins_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()

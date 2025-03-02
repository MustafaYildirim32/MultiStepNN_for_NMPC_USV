function [data, info] = setThrusterStateRequest
%SetThrusterState gives an empty data for uuv_gazebo_ros_plugins_msgs/SetThrusterStateRequest
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'uuv_gazebo_ros_plugins_msgs/SetThrusterStateRequest';
[data.On, info.On] = ros.internal.ros.messages.ros.default_type('logical',1);
info.MessageType = 'uuv_gazebo_ros_plugins_msgs/SetThrusterStateRequest';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,1);
info.MatPath{1} = 'on';

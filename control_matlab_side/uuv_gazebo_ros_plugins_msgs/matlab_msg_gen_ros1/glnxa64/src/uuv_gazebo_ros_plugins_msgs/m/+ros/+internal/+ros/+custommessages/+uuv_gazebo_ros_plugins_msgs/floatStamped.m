function [data, info] = floatStamped
%FloatStamped gives an empty data for uuv_gazebo_ros_plugins_msgs/FloatStamped
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'uuv_gazebo_ros_plugins_msgs/FloatStamped';
[data.Header, info.Header] = ros.internal.ros.messages.std_msgs.header;
info.Header.MLdataType = 'struct';
[data.Data, info.Data] = ros.internal.ros.messages.ros.default_type('double',1);
info.MessageType = 'uuv_gazebo_ros_plugins_msgs/FloatStamped';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,7);
info.MatPath{1} = 'header';
info.MatPath{2} = 'header.seq';
info.MatPath{3} = 'header.stamp';
info.MatPath{4} = 'header.stamp.sec';
info.MatPath{5} = 'header.stamp.nsec';
info.MatPath{6} = 'header.frame_id';
info.MatPath{7} = 'data';

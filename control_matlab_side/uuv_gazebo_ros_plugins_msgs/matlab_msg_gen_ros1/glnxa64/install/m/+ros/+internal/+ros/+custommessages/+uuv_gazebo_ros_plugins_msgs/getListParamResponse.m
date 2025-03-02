function [data, info] = getListParamResponse
%GetListParam gives an empty data for uuv_gazebo_ros_plugins_msgs/GetListParamResponse
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'uuv_gazebo_ros_plugins_msgs/GetListParamResponse';
[data.Description, info.Description] = ros.internal.ros.messages.ros.char('string',0);
[data.Tags, info.Tags] = ros.internal.ros.messages.ros.char('string',NaN);
[data.Data, info.Data] = ros.internal.ros.messages.ros.default_type('double',NaN);
info.MessageType = 'uuv_gazebo_ros_plugins_msgs/GetListParamResponse';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,3);
info.MatPath{1} = 'description';
info.MatPath{2} = 'tags';
info.MatPath{3} = 'data';

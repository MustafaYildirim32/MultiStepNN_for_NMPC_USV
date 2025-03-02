function [data, info] = getThrusterConversionFcnResponse
%GetThrusterConversionFcn gives an empty data for uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse';
[data.Fcn, info.Fcn] = ros.internal.ros.custommessages.uuv_gazebo_ros_plugins_msgs.thrusterConversionFcn;
info.Fcn.MLdataType = 'struct';
info.MessageType = 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,6);
info.MatPath{1} = 'fcn';
info.MatPath{2} = 'fcn.function_name';
info.MatPath{3} = 'fcn.tags';
info.MatPath{4} = 'fcn.data';
info.MatPath{5} = 'fcn.lookup_table_input';
info.MatPath{6} = 'fcn.lookup_table_output';

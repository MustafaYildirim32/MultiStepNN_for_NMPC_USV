function [data, info] = thrusterConversionFcn
%ThrusterConversionFcn gives an empty data for uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn';
[data.FunctionName, info.FunctionName] = ros.internal.ros.messages.ros.char('string',0);
[data.Tags, info.Tags] = ros.internal.ros.messages.ros.char('string',NaN);
[data.Data, info.Data] = ros.internal.ros.messages.ros.default_type('double',NaN);
[data.LookupTableInput, info.LookupTableInput] = ros.internal.ros.messages.ros.default_type('double',NaN);
[data.LookupTableOutput, info.LookupTableOutput] = ros.internal.ros.messages.ros.default_type('double',NaN);
info.MessageType = 'uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,5);
info.MatPath{1} = 'function_name';
info.MatPath{2} = 'tags';
info.MatPath{3} = 'data';
info.MatPath{4} = 'lookup_table_input';
info.MatPath{5} = 'lookup_table_output';

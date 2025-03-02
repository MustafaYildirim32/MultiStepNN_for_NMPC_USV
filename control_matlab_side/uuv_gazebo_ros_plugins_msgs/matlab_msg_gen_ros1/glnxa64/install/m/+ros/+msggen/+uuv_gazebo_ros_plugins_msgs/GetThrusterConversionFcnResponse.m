
classdef GetThrusterConversionFcnResponse < ros.Message
    %GetThrusterConversionFcnResponse MATLAB implementation of uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse
    %   This class was automatically generated by
    %   ros.internal.pubsubEmitter.
    %   Copyright 2014-2020 The MathWorks, Inc.
    properties (Constant)
        MessageType = 'uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse' % The ROS message type
    end
    properties (Constant, Hidden)
        MD5Checksum = 'b489744fdf1ea3660acd86f33ee041a7' % The MD5 Checksum of the message definition
        PropertyList = { 'Fcn' } % List of non-constant message properties
        ROSPropertyList = { 'fcn' } % List of non-constant ROS message properties
        PropertyMessageTypes = { 'ros.msggen.uuv_gazebo_ros_plugins_msgs.ThrusterConversionFcn' ...
            } % Types of contained nested messages
    end
    properties (Constant)
    end
    properties
        Fcn
    end
    methods
        function set.Fcn(obj, val)
            validAttributes = {'nonempty', 'scalar'};
            validClasses = {'ros.msggen.uuv_gazebo_ros_plugins_msgs.ThrusterConversionFcn'};
            validateattributes(val, validClasses, validAttributes, 'GetThrusterConversionFcnResponse', 'Fcn')
            obj.Fcn = val;
        end
    end
    methods (Static, Access = {?matlab.unittest.TestCase, ?ros.Message})
        function obj = loadobj(strObj)
        %loadobj Implements loading of message from MAT file
        % Return an empty object array if the structure element is not defined
            if isempty(strObj)
                obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetThrusterConversionFcnResponse.empty(0,1);
                return
            end
            % Create an empty message object
            obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetThrusterConversionFcnResponse(strObj);
        end
    end
end

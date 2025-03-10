
classdef GetModelPropertiesRequest < ros.Message
    %GetModelPropertiesRequest MATLAB implementation of uuv_gazebo_ros_plugins_msgs/GetModelPropertiesRequest
    %   This class was automatically generated by
    %   ros.internal.pubsubEmitter.
    %   Copyright 2014-2020 The MathWorks, Inc.
    properties (Constant)
        MessageType = 'uuv_gazebo_ros_plugins_msgs/GetModelPropertiesRequest' % The ROS message type
    end
    properties (Constant, Hidden)
        MD5Checksum = 'd41d8cd98f00b204e9800998ecf8427e' % The MD5 Checksum of the message definition
        PropertyList = {} % List of non-constant message properties
        ROSPropertyList = {} % List of non-constant ROS message properties
        PropertyMessageTypes = {} % Types of contained nested messages
    end
    properties (Constant)
    end
    properties
    end
    methods
    end
    methods (Static, Access = {?matlab.unittest.TestCase, ?ros.Message})
        function obj = loadobj(strObj)
        %loadobj Implements loading of message from MAT file
        % Return an empty object array if the structure element is not defined
            if isempty(strObj)
                obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetModelPropertiesRequest.empty(0,1);
                return
            end
            % Create an empty message object
            obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetModelPropertiesRequest(strObj);
        end
    end
end

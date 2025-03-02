
classdef GetListParamResponse < ros.Message
    %GetListParamResponse MATLAB implementation of uuv_gazebo_ros_plugins_msgs/GetListParamResponse
    %   This class was automatically generated by
    %   ros.internal.pubsubEmitter.
    %   Copyright 2014-2020 The MathWorks, Inc.
    properties (Constant)
        MessageType = 'uuv_gazebo_ros_plugins_msgs/GetListParamResponse' % The ROS message type
    end
    properties (Constant, Hidden)
        MD5Checksum = 'd3415e6ab074fdc7b0bd82ed3d6fccc7' % The MD5 Checksum of the message definition
        PropertyList = { 'Description' 'Tags' 'Data' } % List of non-constant message properties
        ROSPropertyList = { 'description' 'tags' 'data' } % List of non-constant ROS message properties
        PropertyMessageTypes = { '' ...
            '' ...
            '' ...
            } % Types of contained nested messages
    end
    properties (Constant)
    end
    properties
        Description
        Tags
        Data
    end
    methods
        function set.Description(obj, val)
            val = convertStringsToChars(val);
            validClasses = {'char', 'string'};
            validAttributes = {};
            validateattributes(val, validClasses, validAttributes, 'GetListParamResponse', 'Description');
            obj.Description = char(val);
        end
        function set.Tags(obj, val)
            val = convertStringsToChars(val);
            validClasses = {'cell', 'string'};
            if isempty(val)
                % Allow empty [] input
                val = cell.empty(0, 1);
            end
            val = val(:);
            validAttributes = {'vector'};
            validateattributes(val, validClasses, validAttributes, 'GetListParamResponse', 'Tags');
            obj.Tags = cell(val);
        end
        function set.Data(obj, val)
            validClasses = {'numeric'};
            if isempty(val)
                % Allow empty [] input
                val = double.empty(0, 1);
            end
            val = val(:);
            validAttributes = {'vector'};
            validateattributes(val, validClasses, validAttributes, 'GetListParamResponse', 'Data');
            obj.Data = double(val);
        end
    end
    methods (Static, Access = {?matlab.unittest.TestCase, ?ros.Message})
        function obj = loadobj(strObj)
        %loadobj Implements loading of message from MAT file
        % Return an empty object array if the structure element is not defined
            if isempty(strObj)
                obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetListParamResponse.empty(0,1);
                return
            end
            % Create an empty message object
            obj = ros.msggen.uuv_gazebo_ros_plugins_msgs.GetListParamResponse(strObj);
        end
    end
end

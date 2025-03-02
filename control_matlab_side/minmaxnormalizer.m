classdef minmaxnormalizer
    %NORMALIZER A class to perform min-max normalization on velocity and thrust data.
    %
    %   This class normalizes velocity and thrust data based on provided minimum
    %   and maximum values. The normalization formula used is:
    %       normalized_value = 2 * (value - min_val) / range_val + bias
    %
    %   Example:
    %       min_vals_vel = [-1.8111, -1.1361, -0.9798];
    %       max_vals_vel = [2.7571, 0.9619, 0.9120];
    %       min_vals_thrust = [-19.8800, -19.8800];
    %       max_vals_thrust = [33.6000, 33.6000];
    %       normalizer = Normalizer(min_vals_vel, max_vals_vel, min_vals_thrust, max_vals_thrust);
    %       normalized_vel = normalizer.normalize_vel_11([0, 0, 0]);
    %       normalized_thrust = normalizer.normalize_thrust_11([10, 10]);





    properties
        min_vals_vel      % 1x3 vector of minimum velocity values
        max_vals_vel      % 1x3 vector of maximum velocity values
        range_vals_vel    % 1x3 vector of velocity ranges (max - min)
        vel_bias          % Scalar bias for velocity normalization (default: -1)
        
        min_vals_thrust   % 1x2 vector of minimum thrust values
        max_vals_thrust   % 1x2 vector of maximum thrust values
        range_vals_thrust % 1x2 vector of thrust ranges (max - min)
        thrust_bias       % Scalar bias for thrust normalization (default: -1)
        range_vals_thrust_single
    end
    
    methods
        function obj = minmaxnormalizer()
            %NORMALIZER Constructor for the Normalizer class.
            %
            %   obj = Normalizer(min_vals_vel, max_vals_vel, min_vals_thrust, max_vals_thrust)
            %
            %   Inputs:
            %       min_vals_vel    - 1x3 vector of minimum velocity values
            %       max_vals_vel    - 1x3 vector of maximum velocity values
            %       min_vals_thrust - 1x2 vector of minimum thrust values
            %       max_vals_thrust - 1x2 vector of maximum thrust values
            
            % min_vals_vel: tensor([[-1.8093, -1.0340, -1.4979]])
            % max_vals_vel: tensor([[2.6907, 1.0978, 1.5171]])
            % range_vals_vel: tensor([[4.5000, 2.1318, 3.0150]])
            % min_vals_thrust: tensor([[-19.8800, -19.8800]])
            % max_vals_thrust: tensor([[33.6000, 33.6000]])
            % range_vals_thrust: tensor([[53.4800, 53.4800]])
            % Initialize velocity properties
            obj.min_vals_vel = [-1.8093, -1.0340, -1.4979]';
            obj.max_vals_vel = [2.6907, 1.0978, 1.5171]';
            obj.range_vals_vel = obj.max_vals_vel - obj.min_vals_vel;
            obj.vel_bias = -1;
            
            % Initialize thrust properties
            obj.min_vals_thrust =[-19.8800, -19.8800]';
            obj.max_vals_thrust =  [33.6000, 33.6000]';
            obj.range_vals_thrust = obj.max_vals_thrust - obj.min_vals_thrust;
            obj.range_vals_thrust_single = obj.max_vals_thrust(1)-obj.min_vals_thrust(1);
            obj.thrust_bias = -1;
        end
        
        function normalized_vel = normalize_vel(obj, vel)
            %NORMALIZE_VEL_11 Normalize velocity data.
            %
            %   normalized_vel = normalize_vel_11(obj, vel)
            %
            %   Inputs:
            %       vel - Nx3 matrix of velocity data to be normalized
            %
            %   Outputs:
            %       normalized_vel - Nx3 matrix of normalized velocity data
            
            % Validate input dimensions
            if size(vel, 1) ~= 3
                error('Input velocity must have 3 columns (components).');
            end
            
            % Perform normalization
            normalized_vel = 2*(vel - obj.min_vals_vel) ./ obj.range_vals_vel + obj.vel_bias;
        end
        
        function normalized_thrust = normalize_thrust(obj, thrust)
            %NORMALIZE_THRUST_11 Normalize thrust data.
            %
            %   normalized_thrust = normalize_thrust_11(obj, thrust)
            %
            %   Inputs:
            %       thrust - Nx2 matrix of thrust data to be normalized
            %
            %   Outputs:
            %       normalized_thrust - Nx2 matrix of normalized thrust data
            
            % Validate input dimensions
            
            % Perform normalization
            normalized_thrust = 2*(thrust + 19.88) ./ 53.48 + obj.thrust_bias;
        end
    end
end
%% Clear Workspace, Close Figures, and Set Up Paths
clear; clc; close all;
load("dlnet_one_shot_linear.mat");   % Load the traced PyTorch network
addpath("utils/");                   % Add utilities folder to path
addpath(genpath("uuv_gazebo_ros_plugins_msgs/"))

%% Define Simulation and Network Parameters
state_dim   = 3;    % e.g., surge, sway, yaw rate
control_dim = 2;    % e.g., two thrusters
horizon     = 10;   % Prediction horizon (steps)
input_size  = state_dim + horizon * control_dim;  % Network input size (e.g., 23)
normalizer  = minmaxnormalizer();  % Instantiate the normalizer

%% Set Reference Trajectory and Initial Conditions
% Define initial position and reference trajectory waypoints (columns)
pos_0 = [0; 0; 0];  % [x; y; yaw]
y_m = 20;           % y-coordinate of the circle centers
r_m = 35;            % Radius of the main (reference) circle (for plotting)
r_i = 30;            % Radius of the inner circle
r_o = 40;            % Radius of the outer circle
n = 10;             % Number of points on each inner and outer circle
phase_delay = deg2rad(0.5*360/n);  % 10 degree phase delay in radians

start_idx = 16;  

%% Generate Zigzag Points
angles = linspace(0, 2*pi - 2*pi/n, n);
outer_x = r_o * cos(angles);
outer_y = y_m + r_o * sin(angles);
inner_x = r_i * cos(angles + phase_delay);
inner_y = y_m + r_i * sin(angles + phase_delay);
zigzag_x = zeros(2*n, 1);
zigzag_y = zeros(2*n, 1);
zigzag_x(1:2:end) = outer_x;
zigzag_y(1:2:end) = outer_y;
zigzag_x(2:2:end) = inner_x;
zigzag_y(2:2:end) = inner_y;

%% Rotate the Zigzag Points based on the starting index

% Check that start_idx is within valid bounds
totalPoints = length(zigzag_x);
if start_idx < 1 || start_idx > totalPoints
    warning('start_idx is out of bounds. Resetting to 1.');
    start_idx = 1;
end

% Rotate the points so that the first point is at start_idx
zigzag_x = [zigzag_x(start_idx:end); zigzag_x(1:start_idx-1)];
zigzag_y = [zigzag_y(start_idx:end); zigzag_y(1:start_idx-1)];

%% Set Reference Trajectory and Initial Conditions

% Define initial position [x; y; yaw]
pos_0 = [0; 0; 0];

% Create pos_ref_points from the rotated zigzag path.
% Each column is a waypoint: [x; y; yaw] with yaw set to zero.
pos_ref_points = [zigzag_x.'; zigzag_y.'; zeros(1, totalPoints)];

% u_traj is the initial guess for control inputs
u_traj    = 19 * rand(2 * horizon, 1);  
lb = -19.88 * ones(2 * horizon, 1);
ub =  33.6  * ones(2 * horizon, 1);


%% Initialize Logging Variables
Pos_real_traj      = [];  % Logged real positions
Pos_predicted_traj = [];  % Logged predicted positions
U_applied_traj     = [];  % Logged applied control inputs
TimeLog            = [];  % Logged simulation time
Vel_ref_traj       = [];  % Logged reference velocities
fminconDurations   = [];  % fmincon computation times
ThrustForceLog     = [];  % Logged applied thrust forces
Vel_predicted_traj = [];  % Logged one-step predicted velocities
Vel_real_raj       = [];  % Logged real velocities

%% Simulation Time Settings
dt = 0.1;  
tf = 200;
time_grid = 0:dt:tf;

%% fmincon Options (using active-set algorithm)
options = optimoptions('fmincon', ...
    'Algorithm', 'active-set', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 10, ...
    'UseParallel', false, ...
    'FunctionTolerance', 5e-2);

%% ROS Environment Setup
setenv("ROS_MASTER_URI", "http://172.20.0.2:11311");
setenv("ROS_IP", "172.20.0.1");
rosshutdown;      % Shut down any previous ROS sessions
rosinit;          % Initialize ROS
rosparam set /use_sim_time true;

% Set up publishers and subscribers
thrust0Pub = rospublisher("/heron/thrusters/0/input", "uuv_gazebo_ros_plugins_msgs/FloatStamped");
thrust1Pub = rospublisher("/heron/thrusters/1/input", "uuv_gazebo_ros_plugins_msgs/FloatStamped");
clockSub   = rossubscriber("/clock", "rosgraph_msgs/Clock");
odomSubs   = rossubscriber("/odometry/base");

% Create ROS messages for thrusters and get initial odometry
thrust0Msg = rosmessage(thrust0Pub);
thrust1Msg = rosmessage(thrust1Pub);
odomData   = receive(odomSubs, 5);

%% Define Propeller Conversion Functions
% Maps between PWM input values and thrust forces
input_to_propeller = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
output_thrust      = [-19.88, -16.52, -12.6, -5.6, -1.4, 0.0, 2.24, 9.52, 21.28, 28.0, 33.6];
Th_get_force = @(x) interp1(input_to_propeller, output_thrust, x, 'linear');
Th_get_pwm   = @(x) interp1(output_thrust, input_to_propeller, x, 'linear');

%% ROS Pause/Unpause Service Clients Setup
pauseClient   = rossvcclient('/gazebo/pause_physics');
pauseMsg      = rosmessage(pauseClient);
unpauseClient = rossvcclient('/gazebo/unpause_physics');
unpauseMsg    = rosmessage(unpauseClient);

% Initialize simulation clock
InitialTimeMsg = receive(clockSub);
InitialSeconds = InitialTimeMsg.Clock_.Sec + InitialTimeMsg.Clock_.Nsec * 1e-9;
i = 1;  % Index for reference trajectory
lastSimTime = 0; % Initialize last simulation time

%% Main Simulation Loop
for k = 1:length(time_grid)
    % Select current reference point
    pos_ref = pos_ref_points(:, i);
    
    % For the first iteration, record the initial simulation time
    if k == 1
        lastSimTimeMsg = receive(clockSub, 10);
        lastSimTime = lastSimTimeMsg.Clock_.Sec + lastSimTimeMsg.Clock_.Nsec * 1e-9;
    end
    
    % Retrieve current odometry data and pause physics for planning
    odomData = receive(odomSubs, 5);
    call(pauseClient, pauseMsg);
    
    % Determine if plot should be refreshed (every 10 iterations)
    refresh_plot = (mod(k, 10) == 0);
    
    % Extract current position and orientation from odometry
    PositionX = odomData.Pose.Pose.Position.X;
    PositionY = odomData.Pose.Pose.Position.Y;
    quat = [odomData.Pose.Pose.Orientation.X, odomData.Pose.Pose.Orientation.Y, ...
            odomData.Pose.Pose.Orientation.Z, odomData.Pose.Pose.Orientation.W];
    eulZYX = quat2eul(quat);
    AngleZ = wrapToPi(eulZYX(3));
    pos_current = [PositionX; PositionY; AngleZ];
    
    % Extract current velocity
    vel_current = [odomData.Twist.Twist.Linear.X;
                   odomData.Twist.Twist.Linear.Y;
                   odomData.Twist.Twist.Angular.Z];
    
    % Compute heading error (difference between current and desired heading)
    phi_current = atan2(pos_ref(2) - pos_current(2), pos_ref(1) - pos_current(1));
    [diff_angle_current, ~, ~] = Angle_Diff(pos_current(3), phi_current);
    
    % Optimization: compute cost and gradient for current control inputs
    tic;
    fun = @(U) costAndGradient_vel(U, dlnet, pos_ref, pos_current, ...
                                     vel_current, diff_angle_current, ...
                                     normalizer, dt);
    [U_opt, ~] = fmincon(fun, u_traj, [], [], [], [], lb, ub, [], options);
    elapsedTime = toc;
    fminconDurations = [fminconDurations, elapsedTime];
    
    % Use first two control inputs for the current time step
    u_first = U_opt(1:2);
    ThrustForceLog = [ThrustForceLog, u_first];
    
    % Convert computed forces to PWM signals and update ROS messages
    thrust0Msg.Data = Th_get_pwm(u_first(1));
    thrust1Msg.Data = Th_get_pwm(u_first(2));
    
    % Predict trajectory using the network
    net_input = [dlarray(normalizer.normalize_vel(vel_current), 'CB'); ...
                 dlarray(normalizer.normalize_thrust(U_opt), 'CB')];
    vel_pred_traj = forward(dlnet, net_input);
    
    % Initialize or update visualization object
    if k == 1
        heron = heron_object(pos_current, [0;0;0], pos_ref_points);
        Vel_predicted_traj = [Vel_predicted_traj, vel_current, vel_pred_traj(1:3)];
    elseif k ~= length(time_grid)
        heron = heron.update(pos_current, [0;0;0], refresh_plot);
        Vel_predicted_traj = [Vel_predicted_traj, vel_pred_traj(1:3)];
    end
    
    % Log applied control inputs and velocities
    U_applied_traj = [U_applied_traj, u_first];
    Vel_real_raj   = [Vel_real_raj, vel_current];
    Vel_ref_traj   = [Vel_ref_traj, [1;0;0]];  % Example constant reference
    
    % Resume physics and send control commands to thrusters
    call(unpauseClient, unpauseMsg);
    send(thrust0Pub, thrust0Msg);
    send(thrust1Pub, thrust1Msg);
    
    % Record simulation time
    simTimeMsg = receive(clockSub, 0.1);
    currentSimTime = simTimeMsg.Clock_.Sec + simTimeMsg.Clock_.Nsec * 1e-9;
    TimeLog = [TimeLog, currentSimTime - InitialSeconds];
    Pos_real_traj = [Pos_real_traj, pos_current];
    
    % Warm-start: use current solution as next initial guess
    u_traj = U_opt;
    
    % Wait until next control cycle (ensuring dt period)
    while true
        simTimeMsg = receive(clockSub, 0.1);
        newSimTime = simTimeMsg.Clock_.Sec + simTimeMsg.Clock_.Nsec * 1e-9;
        if (newSimTime - lastSimTime) >= dt - 0.002
            break;
        end
    end
    lastSimTime = newSimTime;
    
    % Move to next reference point if close enough
    if norm(pos_ref(1:2) - pos_current(1:2)) < 3
        i = i + 1;
        if i > size(pos_ref_points, 2)
            i = i-1;
            if norm(pos_ref(1:2) - pos_current(1:2)) < 1
                break;
            end
        end
    end
end

%% Shutdown: Stop Thrusters and Display Optimization Performance
thrust0Msg.Data = 0;
thrust1Msg.Data = 0;
send(thrust0Pub, thrust0Msg);
send(thrust1Pub, thrust1Msg);
meanFminconDuration = mean(fminconDurations);
disp(['Mean fmincon duration: ', num2str(meanFminconDuration), ' seconds']);

%% Plot: Real vs. Predicted vs. Reference Velocities
figure('Name', 'Predicted vs. Real vs. Reference Velocities');
t_vec = TimeLog;

% Surge Velocity Plot
subplot(3,1,1); hold on; grid on;
title('Surge Velocity');
plot(t_vec, Vel_real_raj(1,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(1,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(1,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Surge [m/s]'); legend('show'); ylim([-1.5 2.5]);

% Sway Velocity Plot
subplot(3,1,2); hold on; grid on;
title('Sway Velocity');
plot(t_vec, Vel_real_raj(2,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(2,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(2,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Sway [m/s]'); legend('show'); ylim([-1 1]);

% Yaw Rate Plot
subplot(3,1,3); hold on; grid on;
title('Yaw Rate');
plot(t_vec, Vel_real_raj(3,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(3,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(3,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Yaw Rate [rad/s]'); legend('show'); ylim([-1.5 1.5]);

%% Plot: Applied Thruster Inputs
figure('Name', 'Thruster Commands');
plot(t_vec, U_applied_traj(1,:), 'LineWidth', 1.2); hold on;
plot(t_vec, U_applied_traj(2,:), 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Force');
legend('Thruster 0', 'Thruster 1');

%% --- Local Functions ---

% Wrapper for computing cost and gradient for velocity tracking using autodiff
function [costVal, gradVal] = costAndGradient_vel(U, dlnet, pos_ref, pos_0, ...
                                                  vel_0, diff_angle_current, ...
                                                  normalizer, dt)
    U_dl = dlarray(U, 'CB');
    [cost_dl, grad_dl] = dlfeval(@(u) internalCostGrad_vel(u, dlnet, pos_ref, pos_0, ...
                                                               vel_0, diff_angle_current, ...
                                                               normalizer, dt), U_dl);
    costVal = double(extractdata(cost_dl));
    gradVal = double(extractdata(grad_dl));
end

% Compute cost and gradient using automatic differentiation
function [cost_dl, grad_dl] = internalCostGrad_vel(U_dl, dlnet, pos_ref, pos_0, ...
                                                    vel_0, diff_angle_current, ...
                                                    normalizer, dt)
    cost_dl = cost_function_for_nn2(dlnet, pos_ref, pos_0, vel_0, ...
                                      U_dl, diff_angle_current, normalizer, dt);
    grad_dl = dlgradient(cost_dl, U_dl);
end

% Define the cost function for tracking error and control effort over the horizon
function costVal = cost_function_for_nn2(dlnet, pos_ref, pos_current, vel_current, ...
                                         U_dl, diff_angle, normalizer, dt)
    costVal = dlarray(0.0, 'CB');
    norm_vel    = normalizer.normalize_vel(dlarray(vel_current, 'CB'));
    norm_thrust = normalizer.normalize_thrust(U_dl);
    net_input   = [norm_vel; norm_thrust];
    v_pred_traj = forward(dlnet, net_input);
    
    horizon = 10;  % Prediction horizon (steps)
    for i = 1:horizon
        % Extract predicted velocity for current step
        vel_next = v_pred_traj(3*i-2:3*i);
        
        % Calculate position update using trapezoidal integration
        psi       = pos_current(3);
        cosPsi    = cos(psi);
        sinPsi    = sin(psi);
        vel_sum   = (vel_next + vel_current) / 2;
        pos_next  = pos_current + dt * [ cosPsi * vel_sum(1) - sinPsi * vel_sum(2);
                                         sinPsi * vel_sum(1) + cosPsi * vel_sum(2);
                                         vel_sum(3) ];
        % Update heading error
        diff_angle = diff_angle - dt * vel_sum(3);
        
        % Compute error terms: position error, control effort, and heading error penalty
        pxError = pos_next(1) - pos_ref(1);
        pyError = pos_next(2) - pos_ref(2);
        costVal = costVal + (pxError^2 + pyError^2) ...            % Position error
                            + 5e-4 * sum(U_dl(2*i-1:2*i).^2) ...  % Control effort penalty
                            + 400 * (2 - cos(diff_angle)) * (diff_angle^2); % Heading error penalty
                        
        % Update state for next iteration
        vel_current = vel_next;
        pos_current = pos_next;
    end
end

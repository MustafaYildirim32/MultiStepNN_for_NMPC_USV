%% Clear Workspace, Close Figures, and Set Up Paths
clear; clc; close all;
load("dlnet_one_step_linear.mat");        % Load the traced PyTorch network
addpath("utils/");            % Add utilities folder to path
addpath(genpath("uuv_gazebo_ros_plugins_msgs/"))
%% Define Simulation and Network Parameters
state_dim   = 3;              % e.g., surge, sway, yaw rate
control_dim = 2;              % e.g., two thrusters
horizon     = 10;             % Prediction horizon (steps)
input_size  = state_dim + horizon * control_dim;  % Expected network input size (e.g., 23)
normalizer  = minmaxnormalizer();  % Instantiate the normalizer

%% Set Reference Trajectory and Initial Conditions
% Reference velocities (each column is a segment reference)
velRefs = [ 2,  0.0,   -1.5,   0    ;
            0.0,   0.0,   0,   0    ;
            0,   0.0,   0,   0];
durations = [3, 2, 3, 2];  % Duration (seconds) for each segment
u_traj    = 19 * rand(2 * horizon, 1);  % Initial guess for control inputs
lb = -19.88 * ones(2 * horizon, 1);
ub =  33.6  * ones(2 * horizon, 1);

%% Initialize Logging Variables
Vel_real_traj      = [];   % Real velocity trajectory
Vel_predicted_traj = [];   % Predicted velocity trajectory
U_applied_traj   = [];   % Applied control inputs
TimeLog          = [];   % Simulation time log
Vel_ref_traj       = [];   % Reference velocities
fminconDurations = [];   % fmincon computation times

%% Simulation Time Settings
dt = 0.1;  
tf = sum(durations);  
time_grid = 0:dt:tf;

%% fmincon Options (using active-set algorithm)
options = optimoptions('fmincon', ...
    'Algorithm', 'active-set', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 10, ...
    'UseParallel', true, ...
    'FunctionTolerance', 5e-3, ...
    'Display', 'iter');

%% ROS Environment Setup
setenv("ROS_MASTER_URI", "http://172.20.0.2:11311");
setenv("ROS_IP", "172.20.0.1");
rosshutdown;           % Shut down any previous ROS sessions
rosinit;               % Initialize ROS
rosparam set /use_sim_time true;

% Set up publishers and subscribers
thrust0Pub = rospublisher("/heron/thrusters/0/input", "uuv_gazebo_ros_plugins_msgs/FloatStamped");
thrust1Pub = rospublisher("/heron/thrusters/1/input", "uuv_gazebo_ros_plugins_msgs/FloatStamped");
clockSub   = rossubscriber("/clock", "rosgraph_msgs/Clock");
odomSubs   = rossubscriber("/odometry/base");

% Create ROS messages for thrusters
thrust0Msg = rosmessage(thrust0Pub);
thrust1Msg = rosmessage(thrust1Pub);

%% Define Propeller Conversion Functions
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
call(unpauseClient, unpauseMsg);
%% Main Simulation Loop
for k = 1:length(time_grid)
    % For the first iteration, get the initial simulation time
    if k == 1
        lastSimTimeMsg = receive(clockSub, 10);
        lastSimTime = lastSimTimeMsg.Clock_.Sec + lastSimTimeMsg.Clock_.Nsec * 1e-9;
    end

    % Receive odometry and current simulation clock messages
    odomData = receive(odomSubs, 5);
    % Pause physics during optimization
    call(pauseClient, pauseMsg);
    
    % Update reference velocity for next iteration
    current_t = time_grid(k);
    cumulative_durations = cumsum(durations);
    seg_idx = find(current_t < cumulative_durations, 1, 'first');
    if isempty(seg_idx), seg_idx = length(durations); end
    vel_ref = velRefs(:, seg_idx);
    
    % Extract current velocities: surge, sway, and yaw rate
    vel_current = [odomData.Twist.Twist.Linear.X;
                   odomData.Twist.Twist.Linear.Y;
                   odomData.Twist.Twist.Angular.Z];
    
    % Solve optimization problem (fmincon) to update control inputs
    tic;
    fun = @(U) costAndGradient_vel(U, dlnet, vel_ref, vel_current, normalizer, dt);
    [U_opt, ~] = fmincon(fun, u_traj, [], [], [], [], lb, ub, [], options);
    elapsedTime = toc;
    fminconDurations = [fminconDurations, elapsedTime];
    
    % Use the first two control inputs for current step
    u_first = U_opt(1:2);
    
    % Convert computed forces to PWM signals and update ROS messages
    thrust0Msg.Data = Th_get_pwm(u_first(1));
    thrust1Msg.Data = Th_get_pwm(u_first(2));
    
    % Predict trajectory using the network
    vel_pred_traj = forward(dlnet, [dlarray(normalizer.normalize_vel(vel_current), 'CB'); ...
                                    dlarray(normalizer.normalize_thrust(U_opt(1:2)), 'CB')]);
    if k == 1
        Vel_predicted_traj = [Vel_predicted_traj, vel_current, vel_pred_traj(1:3)];
    elseif k~=length(time_grid)
        Vel_predicted_traj = [Vel_predicted_traj, vel_pred_traj(1:3)];
    end
    

    
    % Log applied control and real velocities
    U_applied_traj   = [U_applied_traj, u_first];
    Vel_real_traj      = [Vel_real_traj, vel_current];
    Vel_ref_traj       = [Vel_ref_traj, vel_ref];
    
    % Resume physics and send control commands to the thrusters
    call(unpauseClient, unpauseMsg);
    send(thrust0Pub, thrust0Msg);
    send(thrust1Pub, thrust1Msg);
    
    % Record simulation time
    simTimeMsg = receive(clockSub, 0.1);
    currentSimTime = simTimeMsg.Clock_.Sec + simTimeMsg.Clock_.Nsec * 1e-9;
    TimeLog = [TimeLog, currentSimTime - InitialSeconds];
    
    % Warm-start: use current solution as the next initial guess
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
numSteps = size(Vel_real_traj, 2);
t_vec = TimeLog;

% Surge Velocity
subplot(3,1,1); hold on; grid on;
title('Surge Velocity');
plot(t_vec, Vel_real_traj(1,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(1,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(1,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Surge [m/s]'); legend('show'); ylim([-1.5 2.5]);

% Sway Velocity
subplot(3,1,2); hold on; grid on;
title('Sway Velocity');
plot(t_vec, Vel_real_traj(2,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(2,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(2,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Sway [m/s]'); legend('show'); ylim([-1 1]);

% Yaw Rate
subplot(3,1,3); hold on; grid on;
title('Yaw Rate');
plot(t_vec, Vel_real_traj(3,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
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
% Wrapper for computing cost and gradient for velocity tracking
function [costVal, gradVal] = costAndGradient_vel(U, net, vel_ref, vel_current, normalizer, dt)
    U_dl = dlarray(U, 'CB');
    [cost_dl, grad_dl] = dlfeval(@(u) internalCostGrad_vel(u, net, vel_ref, vel_current, normalizer, dt), U_dl);
    costVal = double(extractdata(cost_dl));
    gradVal = double(extractdata(grad_dl));
end

% Compute cost and gradient using automatic differentiation
function [cost_dl, grad_dl] = internalCostGrad_vel(U_dl, dlnet, vel_ref, vel_current, normalizer, dt)
    cost_dl = cost_function_for_nn_vel(dlnet, vel_ref, vel_current, U_dl, normalizer, dt);
    grad_dl = dlgradient(cost_dl, U_dl);
end

% Define the cost function for tracking error and control effort
function costVal = cost_function_for_nn_vel(dlnet, vel_ref, vel_current, U_dl, normalizer, dt)
    % U_dl: (2*N x 1)
    N = size(U_dl,1)/2;
    costVal = dlarray(0.0,'CB');
    vel_current = dlarray(vel_current,'CB');
    dt = dlarray(dt,'CB');
    for i = 1:N
        u_i = U_dl((2*i - 1):(2*i));
        input_nn  = [normalizer.normalize_vel(vel_current);...
                     normalizer.normalize_thrust(u_i)];
        vel_next  = forward(dlnet, input_nn);
        vel_diff=vel_ref-vel_next;
        vel_diff(2) = 0;
        vel_diff(3) = 0;

        costVal  = costVal + 100 * sum((vel_ref-vel_next).^2,'all') + 5e-3*sum(u_i.^2,'all') ;
        vel_current    = vel_next;
    end
end

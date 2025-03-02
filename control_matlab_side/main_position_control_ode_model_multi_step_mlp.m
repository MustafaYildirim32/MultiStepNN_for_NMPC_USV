%% Clear Workspace, Close Figures, and Set Up Paths
clear; clc; close all;
load("dlnet_one_shot_linear.mat");   % Load the traced PyTorch network
addpath("utils/");                   % Add utilities folder to path

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
U_applied_traj     = [];  % Logged applied control inputs
TimeLog            = [];  % Logged simulation time
Vel_ref_traj       = [];  % Logged reference velocities (for plotting)
fminconDurations   = [];  % fmincon computation times
ThrustForceLog     = [];  % Logged computed thrust forces
Vel_predicted_traj = [];  % Logged one‐step predicted velocities
Vel_real_raj       = [];  % Logged real velocities

%% Simulation Time Settings
dt = 0.1;  
tf = 200; %It also finish when the last checkpoint is reached
time_grid = 0:dt:tf;

%% Initialize Simulation States
pos_current = pos_0;         % Inertial position [x; y; psi]
vel_current = [0; 0; 0];       % Body–frame velocity [surge; sway; yaw rate]
i = 1;  % Index for the current reference point

%% fmincon Options (using active-set algorithm)
options = optimoptions('fmincon', ...
    'Algorithm', 'active-set', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 5, ...
    'UseParallel', false, ...
    'FunctionTolerance', 5e-2);

%% Main Simulation Loop
for k = 1:length(time_grid)
    
    % Select current reference position
    pos_ref = pos_ref_points(:, i);
    
    % Compute desired heading toward the reference point
    phi_desired = atan2(pos_ref(2) - pos_current(2), pos_ref(1) - pos_current(1));
    [diff_angle_current, ~, ~] = Angle_Diff(pos_current(3), phi_desired);
    
    % --- Optimization: Determine Control Input ---
    tic;
    fun = @(U) costAndGradient_vel(U, dlnet, pos_ref, pos_current, ...
                                     vel_current, diff_angle_current, ...
                                     normalizer, dt);
    [U_opt, ~] = fmincon(fun, u_traj, [], [], [], [], lb, ub, [], options);
    elapsedTime = toc;
    fminconDurations = [fminconDurations, elapsedTime];
    
    % Use the first two control inputs for the current time step
    u_first = U_opt(1:2);
    ThrustForceLog = [ThrustForceLog, u_first];
    
    % (Optional) Predict the next velocity using the network (for logging)
    net_input = [dlarray(normalizer.normalize_vel(vel_current), 'CB'); ...
                 dlarray(normalizer.normalize_thrust(U_opt), 'CB')];
    vel_pred = forward(dlnet, net_input);
    if k == 1
        Vel_predicted_traj = [Vel_predicted_traj, vel_current,vel_pred(1:3)];
    else
        Vel_predicted_traj = [Vel_predicted_traj, vel_pred(1:3)];
    end
    
    % Log applied control and real velocity for plotting
    U_applied_traj = [U_applied_traj, u_first];
    Vel_real_raj   = [Vel_real_raj, vel_current];
    % For illustration, log a constant reference velocity (adjust as needed)
    Vel_ref_traj   = [Vel_ref_traj, [1; 0; 0]];  
    
    % --- Update Simulation State ---
    % First, store current velocity before updating
    old_vel = vel_current;
    % Update velocity using your discrete model (simulate dynamics)
    new_vel = disc_forward_single_step(old_vel, u_first);
    % Compute the mean velocity over the interval
    v_mean = (old_vel + new_vel) / 2;
    
    % Update the inertial position using the mean velocity.
    % Compute the transformation matrix T based on the current heading angle.
    psi = pos_current(3);
    T = [cos(psi), -sin(psi), 0;
         sin(psi),  cos(psi), 0;
         0,         0,        1];
    pos_current = pos_current + dt * (T * v_mean);
    
    % Set the updated velocity for the next iteration
    vel_current = new_vel;
    
    % --- Visualization Update with heron_object ---
    % (Assuming heron_object is defined in your utilities)
    refresh_plot = (mod(k, 10) == 0);
    if k == 1
        heron = heron_object(pos_current, [0;0;0], pos_ref_points);
    else
        heron = heron.update(pos_current, [0;0;0], refresh_plot);
    end
    
    % Log the real position and simulation time
    Pos_real_traj = [Pos_real_traj, pos_current];
    TimeLog = [TimeLog, time_grid(k)];
    
    % Warm-start: use current solution as next initial guess
    u_traj = U_opt;
    
    % Move to next reference point if close enough in the inertial plane
    if norm(pos_ref(1:2) - pos_current(1:2)) < 3
        i = i + 1;
        if i > size(pos_ref_points, 2)
            break;
        end
    end
end
Vel_predicted_traj=Vel_predicted_traj(:,1:end-1);
mean(fminconDurations)
%% Plot: Predicted vs. Real Velocities
figure('Name', 'Predicted vs. Real Velocities');
t_vec = TimeLog;  % Simulation time vector

% Surge Velocity
subplot(3,1,1); hold on; grid on;
title('Surge Velocity');
plot(t_vec, Vel_real_raj(1,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(1,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time (s)'); ylabel('Surge [m/s]');
legend('show'); ylim([-1.5 2.5]);

% Sway Velocity
subplot(3,1,2); hold on; grid on;
title('Sway Velocity');
plot(t_vec, Vel_real_raj(2,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(2,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time (s)'); ylabel('Sway [m/s]');
legend('show'); ylim([-1 1]);

% Yaw Rate
subplot(3,1,3); hold on; grid on;
title('Yaw Rate');
plot(t_vec, Vel_real_raj(3,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(3,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time (s)'); ylabel('Yaw Rate [rad/s]');
legend('show'); ylim([-1.5 1.5]);
%% --- Local Functions ---

% Wrapper for computing cost and gradient (using autodiff)
function [costVal, gradVal] = costAndGradient_vel(U, dlnet, pos_ref, pos_current, vel_current, diff_angle_current, normalizer, dt)
    U_dl = dlarray(U, 'CB');
    [cost_dl, grad_dl] = dlfeval(@(u) internalCostGrad_vel(u, dlnet, pos_ref, pos_current, vel_current, diff_angle_current, normalizer, dt), U_dl);
    costVal = double(extractdata(cost_dl));
    gradVal = double(extractdata(grad_dl));
end

function [cost_dl, grad_dl] = internalCostGrad_vel(U_dl, dlnet, pos_ref, pos_current, vel_current, diff_angle_current, normalizer, dt)
    cost_dl = cost_function_for_nn2(dlnet, pos_ref, pos_current, vel_current, U_dl, diff_angle_current, normalizer, dt);
    grad_dl = dlgradient(cost_dl, U_dl);
end

% Cost function for position tracking over the prediction horizon
function costVal = cost_function_for_nn2(dlnet, pos_ref, pos_current, vel_current, U_dl, diff_angle, normalizer, dt)
    costVal = dlarray(0.0, 'CB');
    norm_vel    = normalizer.normalize_vel(dlarray(vel_current, 'CB'));
    norm_thrust = normalizer.normalize_thrust(U_dl);
    net_input   = [norm_vel; norm_thrust];
    
    % Obtain the predicted velocity trajectory from the network
    v_pred_traj = forward(dlnet, net_input);
    
    horizon = 10;  % Prediction horizon (steps)
    for i = 1:horizon
        % Extract predicted velocity for the i-th step
        vel_next = v_pred_traj(3*i-2:3*i);
        
        % Compute the next position update using trapezoidal integration
        psi       = pos_current(3);
        cosPsi    = cos(psi);
        sinPsi    = sin(psi);
        vel_sum   = (vel_next + vel_current) / 2;
        pos_next  = pos_current + dt * [ cosPsi * vel_sum(1) - sinPsi * vel_sum(2);
                                         sinPsi * vel_sum(1) + cosPsi * vel_sum(2);
                                         vel_sum(3) ];
        % Update heading error (integrate yaw rate)
        diff_angle = diff_angle - dt * vel_sum(3);
        
        % Compute error terms: position error, control effort, and heading error penalty
        pxError = pos_next(1) - pos_ref(1);
        pyError = pos_next(2) - pos_ref(2);
        costVal = costVal + (pxError^2 + pyError^2) ...           
                            + 5e-4 * sum(U_dl(2*i-1:2*i).^2) ...  
                            + 400 * (2 - cos(diff_angle)) * (diff_angle^2);
                        
        % Update states for next prediction step
        vel_current = vel_next;
        pos_current = pos_next;
    end
end

% Discrete forward step: update velocity based on control inputs
function x_next = disc_forward_single_step(x, thrust)
    % Initialize the next velocity vector
    x_next = zeros(size(x));
    
    % Compute x_next(1)
    x_next(1) = x(1) + (-0.55245 * x(1) ...
            + (-0.00653252) * (x(3)^2) ...
            + 0.869158 * x(2) * x(3) ...
            + (-0.0833071) * abs(x(1)) * x(1) ...
            + 0.0320901 * sum(thrust)) * 0.1;
    
    % Compute x_next(2)
    x_next(2) = x(2) + 0.1 * (-0.57293 * x(2) ...
            + (-0.0726335) * x(3) ...
            + (-0.119536) * abs(x(2)) * x(2) ...
            + 0.0896787 * abs(x(3)) * x(3) ...
            + (-0.000300261) * (thrust(1) - thrust(2)) ...
            + (-1.06866) * x(1) * x(3) ...
            + (-0.0177022) * x(1) * x(2));
    
    % Compute x_next(3)
    x_next(3) = x(3) + 0.1 * (0.028676 * x(2) ...
            + (-0.49859) * x(3) ...
            + 0.0112123 * abs(x(2)) * x(2) ...
            + (-0.435785) * abs(x(3)) * x(3) ...
            + 0.0325477 * (thrust(1) - thrust(2)) ...
            + 0.0600072 * x(1) * x(3) ...
            + 0.0077453 * x(1) * x(2));
end

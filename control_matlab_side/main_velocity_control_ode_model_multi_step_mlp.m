%% Clear Workspace, Close Figures, and Set Up Paths
clear; clc; close all;
load("dlnet_one_shot_linear.mat");   % Load the traced PyTorch network
addpath("utils/");                   % Add utilities folder to path

%% Define Simulation and Network Parameters
state_dim   = 3;                % e.g., surge, sway, yaw rate
control_dim = 2;                % e.g., two thrusters
horizon     = 10;               % Prediction horizon (steps)
input_size  = state_dim + horizon * control_dim;  % Network input size (e.g., 23)
normalizer  = minmaxnormalizer();          % Instantiate the normalizer

%% Set Reference Trajectory and Initial Conditions
% Define reference velocities for each segment (columns)
velRefs = [ 0,  0.0,   -1.2,   0    ;
            0.0,   0.0,   0,   0    ;
            1.2,   0.0,   0,   0];
durations = [3, 2, 3, 2];  % Duration (seconds) for each segment

% Initial guess and bounds for control inputs
u_traj = 19 * rand(2 * horizon, 1);
lb = -19.88 * ones(2 * horizon, 1);
ub =  33.6  * ones(2 * horizon, 1);

%% Initialize Logging Variables and Simulation State
Vel_real_traj      = [];   % Logged real state trajectory (from simulation)
Vel_predicted_traj = [];   % Logged predicted state trajectory (from network)
U_applied_traj   = [];   % Logged applied control inputs
TimeLog          = [];   % Logged simulation time
Vel_ref_traj       = [];   % Logged reference trajectory
fminconDurations = [];   % Logged fmincon computation times
ThrustForceLog   = [];   % Logged computed thrust forces

% Initial simulation state (starting from rest)
state = [0; 0; 0];

%% Simulation Time Settings
dt = 0.1;  
tf = sum(durations);  
time_grid = 0:dt:tf;
cumulative_durations = cumsum(durations);

%% fmincon Options (using active-set algorithm)
options = optimoptions('fmincon', ...
    'Algorithm', 'active-set', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 10, ...
    'UseParallel', true, ...
    'FunctionTolerance', 5e-3, ...
    'Display', 'iter');

%% Main Simulation Loop
for k = 1:length(time_grid)
    current_t = time_grid(k);
    
    % Determine current segment based on cumulative durations
    seg_idx = find(current_t < cumulative_durations, 1, 'first');
    if isempty(seg_idx)
        seg_idx = length(durations);
    end
    vel_ref = velRefs(:, seg_idx);  % Current reference velocity
    
    % Use current simulation state as measured velocity
    vel_current = state;
    
    % Solve optimization problem using fmincon with the NN predictor
    tic;
    fun = @(U) costAndGradient_vel(U, dlnet, vel_ref, vel_current, normalizer, dt);
    [U_opt, ~] = fmincon(fun, u_traj, [], [], [], [], lb, ub, [], options);
    elapsedTime = toc;
    fminconDurations = [fminconDurations, elapsedTime];
    
    % Apply only the first two control inputs for this time step
    u_first = U_opt(1:2);
    ThrustForceLog = [ThrustForceLog, u_first];  % Log computed thrust forces
    
    % Predict next state using the neural network
    net_input = [dlarray(normalizer.normalize_vel(vel_current), 'CB'); ...
                 dlarray(normalizer.normalize_thrust(U_opt), 'CB')];
    vel_pred_traj = forward(dlnet, net_input);
    
    % Initialize or update predicted trajectory log
    if k == 1
        Vel_predicted_traj = [Vel_predicted_traj, vel_current, vel_pred_traj(1:3)];
    elseif k~=length(time_grid)
        Vel_predicted_traj = [Vel_predicted_traj, vel_pred_traj(1:3)];
    end
    
    % Log applied control, real state, and reference
    U_applied_traj = [U_applied_traj, u_first];
    Vel_real_traj    = [Vel_real_traj, vel_current];
    Vel_ref_traj     = [Vel_ref_traj, vel_ref];
    
    % Update the simulation state using a discrete forward step
    state = disc_forward_single_step(state, u_first);
    
    % Log current simulation time
    TimeLog = [TimeLog, current_t];
    
    % Warm-start: use current solution as next initial guess
    u_traj = U_opt;
end

%% Plot: Real vs. Predicted vs. Reference Velocities
figure('Name', 'Predicted vs. Real vs. Reference Velocities');
t_vec = TimeLog;

% Surge Velocity Plot
subplot(3,1,1); hold on; grid on;
title('Surge Velocity');
plot(t_vec, Vel_real_traj(1,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(1,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(1,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Surge [m/s]'); legend('show'); ylim([-1.5 2.5]);

% Sway Velocity Plot
subplot(3,1,2); hold on; grid on;
title('Sway Velocity');
plot(t_vec, Vel_real_traj(2,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(2,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(2,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Sway [m/s]'); legend('show'); ylim([-1 1]);

% Yaw Rate Plot
subplot(3,1,3); hold on; grid on;
title('Yaw Rate');
plot(t_vec, Vel_real_traj(3,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real');
plot(t_vec, Vel_predicted_traj(3,:), '--r', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
plot(t_vec, Vel_ref_traj(3,:), ':g', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Time (s)'); ylabel('Yaw Rate [rad/s]'); legend('show'); ylim([-1.5 1.5]);

%% Plot: Applied Control Inputs
figure('Name', 'Control Inputs');
plot(t_vec, U_applied_traj(1,:), 'LineWidth', 1.2); hold on;
plot(t_vec, U_applied_traj(2,:), 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Force');
legend('Control 1', 'Control 2');

%% Display Optimization Performance
meanFminconDuration = mean(fminconDurations);
disp(['Mean fmincon duration: ', num2str(meanFminconDuration), ' seconds']);

%% --- Local Functions ---

% Wrapper for computing cost and gradient for velocity tracking using autodiff
function [costVal, gradVal] = costAndGradient_vel(U, dlnet, vel_ref, vel_current, normalizer, dt)
    U_dl = dlarray(U, 'CB');   
    [cost_dl, grad_dl] = dlfeval(@(u) internalCostGrad_vel(u, dlnet, vel_ref, vel_current, normalizer, dt), U_dl);
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
    % Normalize current velocity and thrust inputs
    norm_vel    = normalizer.normalize_vel(dlarray(vel_current, 'CB'));
    norm_thrust = normalizer.normalize_thrust(U_dl);
    % Concatenate normalized inputs to form the network input
    net_input = [norm_vel; norm_thrust];
    
    % Obtain predicted trajectory from the network
    v_pred_traj = forward(dlnet, net_input);
    
    % Determine the prediction horizon based on network output size
    horizon = numel(v_pred_traj) / numel(vel_ref); 
    % Repeat the reference velocity to match prediction horizon
    error = v_pred_traj - repmat(vel_ref, horizon, 1);
    error(2) = 0;
    % Compute cost as weighted sum of squared prediction errors and control effort
    costVal = 10000 * sum(error.^2, 'all') + 1 * sum(U_dl.^2);
end

% Discrete forward step using provided dynamics (Euler integration)
function x_next = disc_forward_single_step(x, thrust)
    x_next = zeros(size(x));
    
    % Update surge (x-direction)
    x_next(1) = x(1) + (-0.55245 * x(1) ...
                  - 0.00653252 * (x(3)^2) ...
                  + 0.869158 * x(2) * x(3) ...
                  - 0.0833071 * abs(x(1)) * x(1) ...
                  + 0.0320901 * sum(thrust)) * 0.1;
    
    % Update sway (y-direction)
    x_next(2) = x(2) + 0.1 * (-0.57293 * x(2) ...
                  - 0.0726335 * x(3) ...
                  - 0.119536 * abs(x(2)) * x(2) ...
                  + 0.0896787 * abs(x(3)) * x(3) ...
                  - 0.000300261 * (thrust(1) - thrust(2)) ...
                  - 1.06866 * x(1) * x(3) ...
                  - 0.0177022 * x(1) * x(2));
    
    % Update yaw rate (rotation)
    x_next(3) = x(3) + 0.1 * (0.028676 * x(2) ...
                  - 0.49859 * x(3) ...
                  + 0.0112123 * abs(x(2)) * x(2) ...
                  - 0.435785 * abs(x(3)) * x(3) ...
                  + 0.0325477 * (thrust(1) - thrust(2)) ...
                  + 0.0600072 * x(1) * x(3) ...
                  + 0.0077453 * x(1) * x(2));
end

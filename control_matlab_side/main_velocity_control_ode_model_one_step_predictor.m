%% Clear Workspace, Close Figures, and Set Up Paths
clear; clc; close all;
load("dlnet_one_step_linear.mat");  % Load the traced PyTorch network
addpath("utils/");                      % Add utilities folder to path

%% Define Simulation and Network Parameters
state_dim   = 3;                % e.g., surge, sway, yaw rate
control_dim = 2;                % e.g., two thrusters
horizon     = 10;               % Prediction horizon (steps)
input_size  = state_dim + horizon * control_dim;  % Expected network input size (e.g., 23)
normalizer  = minmaxnormalizer();        % Instantiate the normalizer

%% Set Reference Trajectory and Initial Conditions
% Define reference velocities for each segment (each column corresponds to one segment)
velRefs = [ 2.0,  -1.5,   0,   0,    0;
            0.0,   0.0,   0,   0,    0;
            0.0,   0,     0,   1.3, -1.3];
durations = [2, 2, 2, 2, 2];  % Duration (seconds) for each segment

% Initial guess and bounds for control inputs
u_traj = 19 * rand(2 * horizon, 1);
lb = -19.88 * ones(2 * horizon, 1);
ub =  33.6  * ones(2 * horizon, 1);

%% Initialize Logging Variables and Simulation State
Vel_real_traj      = [];   % Logged real state trajectory (from simulation)
Vel_predicted_traj = [];   % Logged predicted state trajectory (from NN predictor)
U_applied_traj   = [];   % Logged applied control inputs
TimeLog          = [];   % Logged simulation times
Vel_ref_traj       = [];   % Logged reference trajectories
fminconDurations = [];   % Logged fmincon computation times

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
    'FunctionTolerance', 5e-2, ...
    'Display', 'iter');

%% Main Simulation Loop
for k = 1:length(time_grid)
    current_t = time_grid(k);
    
    % Determine the current segment based on cumulative durations
    seg_idx = find(current_t < cumulative_durations, 1, 'first');
    if isempty(seg_idx)
        seg_idx = length(durations);
    end
    vel_ref = velRefs(:, seg_idx);  % Current reference velocity
    
    % Solve the optimization problem using the NN predictor starting from the current state
    tic;
    fun = @(U) costAndGradient_vel(U, dlnet, vel_ref, state, normalizer, dt);
    [U_opt, ~] = fmincon(fun, u_traj, [], [], [], [], lb, ub, [], options);
    elapsedTime = toc;
    fminconDurations = [fminconDurations, elapsedTime];
    
    % Use the first two control inputs for the current time step and log them
    u_first = U_opt(1:2);
    U_applied_traj = [U_applied_traj, u_first];
    
    % Predict the next state using the neural network predictor
    input_nn = [normalizer.normalize_vel(state); ...
                normalizer.normalize_thrust(u_first)];
    vel_pred_dl = forward(dlnet, dlarray(input_nn, 'CB'));
    vel_pred = extractdata(vel_pred_dl);
    
    % Log the predicted trajectory: for the first iteration, store the initial state
    if k == 1
        Vel_predicted_traj = [Vel_predicted_traj, state,vel_pred(1:3)];
    elseif k~=length(time_grid)
        Vel_predicted_traj = [Vel_predicted_traj, vel_pred(1:3)];
    end
    
    % Log the reference trajectory
    Vel_ref_traj = [Vel_ref_traj, vel_ref];
    Vel_real_traj = [Vel_real_traj, state]; 
    % Update the real simulation state using the discrete forward step function
    state = disc_forward_single_step(state, u_first);
    % Log the updated state
    
    % Log current simulation time and warm-start the next optimization
    TimeLog = [TimeLog, current_t];
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
    % Number of control steps (prediction horizon)
    N = size(U_dl,1) / 2;
    costVal = dlarray(0.0, 'CB');
    % Convert current state and dt to dlarray for compatibility
    vel_current = dlarray(vel_current, 'CB');
    dt = dlarray(dt, 'CB');
    
    for i = 1:N
        % Extract the current control input for step i
        u_i = U_dl((2*i - 1):(2*i));
        % Form the network input: normalized current state, normalized thrust, and dt
        input_nn = [normalizer.normalize_vel(vel_current); ...
                    normalizer.normalize_thrust(u_i)];
        % Predict next state using the network
        vel_next = forward(dlnet, input_nn);
        % Accumulate cost: tracking error and control effort penalty
        costVal = costVal + 100 * sum((vel_ref - vel_next).^2, 'all') ...
                            + 5e-5 * sum(u_i.^2, 'all');
        % Update state for the next iteration
        vel_current = vel_next;
    end
end

% Discrete forward step function (Euler integration using provided dynamics)
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

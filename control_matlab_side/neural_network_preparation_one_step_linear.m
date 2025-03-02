% Define network architecture parameters (ensure these match your Python
% model and this script is written for neural networks contain only linear fcc layers)
clear all;close all;clc
addpath("model_weights/");
input_size = 5;   % e.g., state_dim + horizon*control_dim
hidden_size = 8; % e.g., 16
output_size = 3; % e.g., horizon * state_dim

%Load the weights come from training with pytorch
data = load('model_weights_one_step_linear.mat');

% Create the layers manually (Default is for 2 hidden layer the hidden size can be arranged upper side)
layers = [
    featureInputLayer(input_size, 'Name', 'input')
    fullyConnectedLayer(hidden_size, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(hidden_size, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(output_size, 'Name', 'fc3')
    ];
lgraph = layerGraph(layers);
% Convert the layers into a layer graph and then a dlnetwork
% Extract layers from the layer graph
layers = lgraph.Layers;

% --- Update fc1 ---
fc1_idx = find(arrayfun(@(x) strcmp(x.Name, 'fc1'), layers));
fc1 = layers(fc1_idx);
%data.network_0_weight is 16 x 23 single matrix vector
%data.network_0_bias is 1 x 16 single row vector
fc1.Weights = data.network_0_weight;
fc1.Bias = data.network_0_bias';  % Transpose to make it 16

% --- Update fc2 ---
fc2_idx = find(arrayfun(@(x) strcmp(x.Name, 'fc2'), layers));
fc2 = layers(fc2_idx);
fc2.Weights = data.network_2_weight;
fc2.Bias    = data.network_2_bias';  % Transposed bias

% --- Update fc3 ---
fc3_idx = find(arrayfun(@(x) strcmp(x.Name, 'fc3'), layers));
fc3 = layers(fc3_idx);
fc3.Weights = data.network_4_weight;
fc3.Bias    = data.network_4_bias';  % Transposed bias

% Replace the updated layers back into the layer graph
lgraph = replaceLayer(lgraph, 'fc1', fc1);
lgraph = replaceLayer(lgraph, 'fc2', fc2);
lgraph = replaceLayer(lgraph, 'fc3', fc3);

dlnet = dlnetwork(lgraph);
state_dim   = 3;      % e.g., surge, sway, yaw
control_dim = 2;      % e.g., two thrusters
horizon     = 10;     % prediction horizon (number of steps predicted at once)
input_size  = state_dim + horizon * control_dim;  % should be 23
%%
save("dlnet_custom.mat","dlnet")
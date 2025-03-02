import os
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid Qt errors
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

# Import your utility functions
from utils.data_loader import load_data
from utils.normalizator import Normalizer

# Set device (GPU if available, else CPU)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################
# User settings
###############################################################################
do_train = True  # Set to True to train (even if a model exists); False to load a model
model_path = 'models/model_multi_step_tracd.pt'

###############################################################################
# Define the Direct Multi-Step Predictor (a shallow network for real-time MPC)
###############################################################################
class MultiStepPredictor(nn.Module):
    def __init__(self, horizon, hidden_size=16, num_hidden_layers=2):
        """
        Direct multi-step predictor network.
        
        Parameters:
          - horizon (int): Number of time steps to predict.
          - hidden_size (int): Number of neurons in each hidden layer.
          - num_hidden_layers (int): Number of hidden layers.
        """
        super(MultiStepPredictor, self).__init__()
        self.horizon = horizon
        self.state_dim = 3   # e.g., surge, sway, yaw
        self.control_dim = 2 # e.g., two thrusters

        # Input: initial state (state_dim) and flattened control sequence (horizon*control_dim)
        self.input_size = self.state_dim + horizon * self.control_dim
        # Output: a sequence of states: horizon * state_dim (flattened to 30 values)
        self.output_size = horizon * self.state_dim

        # Build a shallow feedforward network
        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights (using Xavier initialization)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
          x: Tensor of shape [batch_size, input_size]
        Returns:
          out: Tensor of shape [batch_size, 30] (flattened output)
        """
        out = self.network(x)
        return out

###############################################################################
# Prepare the Data (create sliding windows for multi-step training)
###############################################################################
# Load training and validation data
vel_obs, thrust_obs = load_data(data_path=r'datas/data_train_gazebointerpolated.mat')
vel_valid_obs, thrust_valid_obs = load_data(data_path=r'datas/data2gazebointerpolated.mat')

vel_obs = vel_obs[9:, :]  # remove the first 9 samples
thrust_obs = thrust_obs[9:, :]  # remove the first 9 samples

vel_valid_obs = vel_valid_obs[9:, :]  # remove the first 9 samples
thrust_valid_obs = thrust_valid_obs[9:, :]  # remove the first 9 samples
print(f"vel_obs shape: {vel_obs.shape}")
print(f"thrust_obs shape: {thrust_obs.shape}")
print(f"vel_valid_obs shape: {vel_valid_obs.shape}")
print(f"thrust_valid_obs shape: {thrust_valid_obs.shape}")

# Get training normalization statistics from training data
min_train_vel, _ = vel_obs.min(dim=0, keepdim=True)
max_train_vel, _ = vel_obs.max(dim=0, keepdim=True)
min_train_thrust, _ = thrust_obs.min(dim=0, keepdim=True)
max_train_thrust, _ = thrust_obs.max(dim=0, keepdim=True)

# Instantiate the normalizer (assumed to have methods normalize_vel_11 and normalize_thrust_11)
norma = Normalizer(min_train_vel, max_train_vel, min_train_thrust, max_train_thrust)
# Normalize the data (for network inputs)
normalized_vel_obs = norma.normalize_vel_11(vel_obs)
normalized_thrust_obs = norma.normalize_thrust_11(thrust_obs)
normalized_vel_valid_obs = norma.normalize_vel_11(vel_valid_obs)
normalized_thrust_valid_obs = norma.normalize_thrust_11(thrust_valid_obs)
norma.print_all_attributes()

###############################################################################
# Create sliding-window training samples for the multi-step predictor
###############################################################################
def create_multi_step_samples(vel, thrust, norm_vel, norm_thrust, horizon):
    """
    Constructs training samples using a sliding window.
    For each sample:
      - Input: [normalized initial velocity at time i, flattened normalized thrust commands from i to i+horizon-1]
      - Target: actual (unnormalized) velocities from time i+1 to i+horizon
    """
    num_samples = vel.shape[0] - horizon
    X_list = []
    Y_list = []
    for i in range(num_samples):
        init_vel = norm_vel[i, :]  # [state_dim]
        thrust_seq = norm_thrust[i:i+horizon, :]  # [horizon, control_dim]
        thrust_seq_flat = thrust_seq.reshape(-1)   # [horizon*control_dim]
        X_sample = torch.cat([init_vel, thrust_seq_flat], dim=0)
        X_list.append(X_sample)
        target_seq = vel[i+1:i+horizon+1, :]  # [horizon, state_dim]
        Y_list.append(target_seq)
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    return X, Y

# Set prediction horizon (number of time steps predicted in one block)
horizon = 10

# Create training samples
X_train, Y_train = create_multi_step_samples(vel_obs, thrust_obs,
                                               normalized_vel_obs, normalized_thrust_obs,
                                               horizon)
# Create validation samples
X_valid, Y_valid = create_multi_step_samples(vel_valid_obs, thrust_valid_obs,
                                               normalized_vel_valid_obs, normalized_thrust_valid_obs,
                                               horizon)

# Move data to device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_valid = X_valid.to(device)
Y_valid = Y_valid.to(device)

###############################################################################
# Instantiate the model and training parameters
###############################################################################
hidden_size = 16
num_hidden_layers = 2
model = MultiStepPredictor(horizon, hidden_size, num_hidden_layers).to(device)

criterion = nn.MSELoss()  # Mean Squared Error loss
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

num_iterations = 50000
training_losses = []
validation_losses = []
iterations_list = []

###############################################################################
# Train or load the model based on the do_train flag
###############################################################################
if do_train:
    print("Training mode enabled. Training the model from scratch (even if a model already exists)...")
    start_time = time.time()
    for iteration in range(1, num_iterations + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)  # [num_train_samples, 30] flattened output
        loss = criterion(outputs, Y_train.view(outputs.shape))
        loss.backward()
        optimizer.step()

        if iteration % 200 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_loss = criterion(train_outputs, Y_train.view(train_outputs.shape))
                val_outputs = model(X_valid)
                validation_loss = criterion(val_outputs, Y_valid.view(val_outputs.shape))
                training_losses.append(train_loss.item())
                validation_losses.append(validation_loss.item())
                iterations_list.append(iteration)
                print(f"Iteration {iteration}: Training Loss: {train_loss.item()}, Validation Loss: {validation_loss.item()}")
        if iteration % 6000 == 0:
            scheduler.step()
            print(f"Learning Rate {optimizer.param_groups[0]['lr']}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Plot the logarithm of the training and validation losses
    if len(iterations_list) > 0:
        plt.figure(figsize=(8,6))
        plt.semilogy(iterations_list, training_losses, label='Training Loss')
        plt.semilogy(iterations_list, validation_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Save model weights to a MAT file for MATLAB compatibility
    state_dict = model.state_dict()
    weights = {}
    for key, param in state_dict.items():
        key_modified = key.replace('.', '_')
        weights[key_modified] = param.detach().cpu().numpy()
    sio.savemat('model_weights.mat', weights)
    print("Weights and biases saved to model_weights.mat")
    
    traced_model = torch.jit.trace(model, torch.randn(1, model.input_size, device=device))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.jit.save(traced_model, model_path)
    print(f"Traced model saved to {model_path}")
    norma.print_all_attributes()
else:
    if os.path.exists(model_path):
        print("do_train is False. Loading the pre-trained model from file...")
        model = torch.jit.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("Model not found and do_train is False. Please train the model first.")

###############################################################################
# Closed-Loop Simulation (Point-by-Point Prediction) with Raw Predictions Only
###############################################################################
def closed_loop_simulation_point_prediction(model, valid_vel_data, thrust_data,
                                              device, norma, horizon,
                                              start_idx=0, end_idx=None,
                                              pred_step=None):
    """
    For each time step k (between start_idx and end_idx), simulate a horizon-ahead prediction 
    using the current measurement x[k] and extract only the predicted point corresponding 
    to pred_step. Only raw predictions are computed.
    """
    if pred_step is None:
        pred_step = horizon
    if pred_step < 1 or pred_step > horizon:
        raise ValueError(f"pred_step must be between 1 and {horizon}, got {pred_step}")

    model.eval()
    valid_vel_data = valid_vel_data.to(device)
    thrust_data = thrust_data.to(device)
    total_steps = thrust_data.shape[0]
    state_dim = valid_vel_data.shape[1]
    
    if start_idx < 0:
        start_idx = 0
    if end_idx is None or end_idx < 0:
        end_idx = total_steps - horizon
    else:
        end_idx = min(end_idx, total_steps - horizon)
    
    predicted_raw_points = []
    
    for k in range(start_idx, end_idx):
        # Current measured state at time k (unnormalized)
        current_state = valid_vel_data[k, :]  # shape: [state_dim]
        # Get thrust commands from k to k+horizon-1; shape: [horizon, control_dim]
        thrust_seq = thrust_data[k:k+horizon, :]
        # Normalize thrust commands
        norm_thrust_seq = norma.normalize_thrust_11(thrust_seq)
        # Normalize current state (the network expects normalized state)
        current_state_norm = norma.normalize_vel_11(current_state).squeeze(0)
        # Prepare network input: [normalized state, flattened normalized thrust sequence]
        input_tensor = torch.cat([current_state_norm, norm_thrust_seq.reshape(-1)], dim=0).unsqueeze(0)
        
        # Get raw prediction block (shape: [1, 30] flattened output)
        with torch.no_grad():
            raw_pred_flat = model(input_tensor)
        # Reshape using the local 'horizon' and 'state_dim' variables
        raw_pred = raw_pred_flat.view(horizon, state_dim)  # shape: [horizon, state_dim]
        
        # Raw prediction: use the pred_step-th predicted point (1-indexed)
        raw_final = raw_pred[pred_step - 1, :]
        
        predicted_raw_points.append(raw_final.cpu().numpy())
    
    predicted_raw_points = np.array(predicted_raw_points)   # shape: [num_predictions, state_dim]
    sim_range = np.arange(start_idx, end_idx)
    
    return predicted_raw_points, sim_range

# Define simulation indices and prediction step
start_idx = -1   # starting index (will default to 0)
end_idx = -1     # ending index (will default to total_steps - horizon)
pred_step = 10   # choose which prediction (1 to horizon) to extract

predicted_raw_pts, sim_range = closed_loop_simulation_point_prediction(
    model, vel_valid_obs, thrust_valid_obs, device, norma, horizon,
    start_idx=start_idx, end_idx=end_idx, pred_step=pred_step
)

# Compute effective expected points: since prediction is at time k+pred_step
effective_start = sim_range[0]
effective_end = sim_range[-1] + 1
expected_pts = vel_valid_obs[effective_start + pred_step : effective_end + pred_step, :].cpu().numpy()

###############################################################################
# Plot Validation Comparisons: All Three Velocity Channels in a Single Figure
###############################################################################

time_axis = sim_range * 0.1

velocity_labels = ['Surge Velocity', 'Sway Velocity', 'Yaw Velocity']
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
for i, label in enumerate(velocity_labels):
    axes[i].plot(time_axis, expected_pts[:, i], label="Actual", linewidth=2)
    axes[i].plot(time_axis, predicted_raw_pts[:, i], '--', label="Raw Prediction")
    axes[i].set_ylabel(label, fontsize=12)
    axes[i].set_title(f"{label} - {pred_step}-step Ahead Prediction", fontsize=14)
    axes[i].legend(fontsize=10)
    axes[i].grid(True)
axes[-1].set_xlabel("Time (seconds)", fontsize=12)

plt.tight_layout()
plt.show()

# Compute and Print MSE for Each Velocity Channel (Raw Predictions)
print("\nMean Squared Error (MSE) on Validation Data (Prediction at time k+pred_step):")
for i, label in enumerate(velocity_labels):
    mse_raw = np.mean((predicted_raw_pts[:, i] - expected_pts[:, i])**2)
    print(f"{label}: Raw MSE = {mse_raw:.6f}")

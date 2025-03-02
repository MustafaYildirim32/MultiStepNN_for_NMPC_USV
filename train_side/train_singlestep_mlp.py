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

# Set device (CPU if available; here we choose CPU for training this shallow network)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################
# User settings
###############################################################################
do_train = True  # Set to True to train; False to load a pre-trained model
model_path = 'models/model_one_step_linear.pt'

###############################################################################
# Define the One-Step Predictor Network
###############################################################################
class OneStepPredictor(nn.Module):
    def __init__(self, hidden_size=16, num_hidden_layers=2):
        """
        One-step predictor network.
        Input: concatenated normalized current state (3) and normalized thrust command (2)
        Output: next state (3) (predicted in unnormalized form)
        """
        super(OneStepPredictor, self).__init__()
        self.state_dim = 3   # e.g., surge, sway, yaw
        self.control_dim = 2 # e.g., two thrusters
        self.input_size = self.state_dim + self.control_dim  # 3 + 2 = 5
        self.output_size = self.state_dim                     # 3
        
        # Build a shallow feedforward network
        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        Forward pass.
          x: Tensor of shape [batch_size, 5]
        Returns:
          out: Tensor of shape [batch_size, 3] (predicted next state)
        """
        out = self.network(x)
        return out

###############################################################################
# Prepare the Data (create sliding windows for one-step training)
###############################################################################
# Load training and validation data
vel_obs, thrust_obs = load_data(data_path=r'datas/data_train_gazebointerpolated.mat')
vel_valid_obs, thrust_valid_obs = load_data(data_path=r'datas/datagazebodcthrust_smoothed_interp.mat')

# Remove the first 9 samples (if necessary)
vel_obs = vel_obs[9:, :]
thrust_obs = thrust_obs[9:, :]

vel_valid_obs = vel_valid_obs[9:, :]
thrust_valid_obs = thrust_valid_obs[9:, :]
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
# Normalize the data for network inputs
normalized_vel_obs = norma.normalize_vel_11(vel_obs)
normalized_thrust_obs = norma.normalize_thrust_11(thrust_obs)
normalized_vel_valid_obs = norma.normalize_vel_11(vel_valid_obs)
normalized_thrust_valid_obs = norma.normalize_thrust_11(thrust_valid_obs)
norma.print_all_attributes()

###############################################################################
# Create sliding-window training samples for the one-step predictor
###############################################################################
def create_one_step_samples(vel, thrust, norm_vel, norm_thrust):
    """
    Constructs training samples for a one-step predictor.
    For each sample:
      - Input: [normalized current velocity at time i, normalized thrust command at time i]
      - Target: actual (unnormalized) velocity at time i+1
    """
    num_samples = vel.shape[0] - 1  # one-step prediction
    X_list = []
    Y_list = []
    for i in range(num_samples):
        curr_vel = norm_vel[i, :]           # [state_dim]
        thrust_cmd = norm_thrust[i, :]        # [control_dim]
        X_sample = torch.cat([curr_vel, thrust_cmd], dim=0)  # [5]
        X_list.append(X_sample)
        target = vel[i+1, :]                # [state_dim]
        Y_list.append(target)
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    return X, Y

# Create training samples (for one-step prediction)
X_train, Y_train = create_one_step_samples(vel_obs, thrust_obs,
                                           normalized_vel_obs, normalized_thrust_obs)
# Create validation samples
X_valid, Y_valid = create_one_step_samples(vel_valid_obs, thrust_valid_obs,
                                           normalized_vel_valid_obs, normalized_thrust_valid_obs)

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
model = OneStepPredictor(hidden_size, num_hidden_layers).to(device)

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
    print("Training mode enabled. Training the one-step predictor from scratch...")
    start_time = time.time()
    for iteration in range(1, num_iterations + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)  # [num_train_samples, 3]
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        if iteration % 200 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(X_train), Y_train)
                validation_loss = criterion(model(X_valid), Y_valid)
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
    sio.savemat('models/model_weights_one_step.mat', weights)
    print("Weights and biases saved to models/model_weights_one_step.mat")
    
    traced_model = torch.jit.trace(model, torch.randn(1, model.input_size, device=device))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.jit.save(traced_model, model_path)
    print(f"Traced model saved to {model_path}")
    norma.print_all_attributes()
else:
    if os.path.exists(model_path):
        print("do_train is False. Loading the pre-trained one-step predictor...")
        model = torch.jit.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("Model not found and do_train is False. Please train the model first.")

###############################################################################
# Closed-Loop Recursive Simulation (Iteratively Applying the One-Step Predictor)
###############################################################################
def closed_loop_recursive_prediction(model, valid_vel_data, thrust_data,
                                     device, norma, sim_steps,
                                     start_idx=0, end_idx=None):
    """
    For each time step k (between start_idx and end_idx), simulate a closed-loop prediction
    by recursively feeding the one-step predictor's output back as input.
    sim_steps: number of iterative predictions (e.g., 10 for a 10-step-ahead prediction)
    """
    model.eval()
    valid_vel_data = valid_vel_data.to(device)
    thrust_data = thrust_data.to(device)
    total_steps = thrust_data.shape[0]
    state_dim = valid_vel_data.shape[1]
    
    if start_idx < 0:
        start_idx = 0
    if end_idx is None or end_idx < 0:
        end_idx = total_steps - sim_steps
    
    predicted_points = []
    
    for k in range(start_idx, end_idx):
        # Use the actual measured state at time k as the starting state (unnormalized)
        current_state = valid_vel_data[k, :].unsqueeze(0)  # shape: [1, state_dim]
        
        # Recursive prediction: for sim_steps, use the one-step predictor iteratively
        for i in range(sim_steps):
            # Get the corresponding thrust command for the current prediction step.
            # (Assuming you want to use the ground truth thrust at time k+i)
            thrust_command = thrust_data[k + i, :].unsqueeze(0)  # shape: [1, control_dim]
            # Normalize current state and thrust command
            current_state_norm = norma.normalize_vel_11(current_state).squeeze(0)  # shape: [state_dim]
            thrust_norm = norma.normalize_thrust_11(thrust_command).squeeze(0)       # shape: [control_dim]
            # Prepare network input: [normalized current state, normalized thrust command]
            input_tensor = torch.cat([current_state_norm, thrust_norm], dim=0).unsqueeze(0)  # shape: [1, 5]
            
            # Get the one-step prediction (unnormalized next state)
            with torch.no_grad():
                pred_next_state = model(input_tensor)  # shape: [1, state_dim]
            # Update current_state for the next iteration
            current_state = pred_next_state
        
        # After sim_steps predictions, record the final predicted state
        predicted_points.append(current_state.squeeze(0).cpu().numpy())
    
    predicted_points = np.array(predicted_points)   # shape: [num_predictions, state_dim]
    sim_range = np.arange(start_idx, end_idx)
    return predicted_points, sim_range

# Set the number of iterative predictions (simulate 10 steps ahead)
sim_steps = 10

# Use the closed-loop recursive simulation function
predicted_recursive_pts, sim_range = closed_loop_recursive_prediction(
    model, vel_valid_obs, thrust_valid_obs, device, norma, sim_steps,
    start_idx=-1, end_idx=-1  # -1 defaults to start=0 and end=total_steps - sim_steps
)

# Compute effective expected points: since prediction is for time k+sim_steps
effective_start = sim_range[0]
effective_end = sim_range[-1] + 1
expected_pts = vel_valid_obs[effective_start + sim_steps : effective_end + sim_steps, :].cpu().numpy()

###############################################################################
# Plot Validation Comparisons: All Three Velocity Channels in a Single Figure
###############################################################################
time_axis = sim_range * 0.1

velocity_labels = ['Surge Velocity', 'Sway Velocity', 'Yaw Velocity']
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
for i, label in enumerate(velocity_labels):
    axes[i].plot(time_axis, expected_pts[:, i], label="Actual", linewidth=2)
    axes[i].plot(time_axis, predicted_recursive_pts[:, i], '--', label="Recursive Prediction")
    axes[i].set_ylabel(label, fontsize=12)
    axes[i].set_title(f"{label} - {sim_steps}-step Ahead Prediction", fontsize=14)
    axes[i].legend(fontsize=10)
    axes[i].grid(True)
axes[-1].set_xlabel("Time (seconds)", fontsize=12)

plt.tight_layout()
plt.show()

# Compute and Print MSE for Each Velocity Channel (Recursive Predictions)
print("\nMean Squared Error (MSE) on Validation Data (Prediction at time k+sim_steps):")
for i, label in enumerate(velocity_labels):
    mse_recursive = np.mean((predicted_recursive_pts[:, i] - expected_pts[:, i])**2)
    print(f"{label}: Recursive MSE = {mse_recursive:.6f}")

num_trials = 1000
inference_times = []

# Warm-up loop to mitigate initial overhead
with torch.no_grad():
    for _ in range(10):
        dummy_input = torch.randn(1, 5, device=device)
        _ = model(dummy_input)

# Measure inference time over multiple trials (for one-step prediction)
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()  # Start timer
        _ = model(dummy_input)    # Perform inference
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        inference_times.append(elapsed_time)

mean_inference_time = np.mean(inference_times)
print(f"Mean Inference Time over {num_trials} runs: {mean_inference_time:.6f} seconds")
np.save('predicted_points_one_step_linear.npy', predicted_recursive_pts)
np.save('expected_points_one_step_linear.npy', expected_pts)
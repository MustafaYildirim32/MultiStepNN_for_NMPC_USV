import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the .mat files
one_shot_data = sio.loadmat('vel_velocity_control_one_shot.mat')
one_step_data = sio.loadmat('vel_velocity_control_one_step.mat')

# Extract the arrays with fixed variable names
vel_one_shot = one_shot_data['Vel_one_shot']     # shape: 3 x N
Vel_ref_traj = one_shot_data['Vel_ref_traj']       # shape: 3 x N
vel_one_step = one_step_data['Vel_one_step']       # shape: 3 x N

# Extract the surge speeds (first row)
surge_one_shot = vel_one_shot[0, :]
surge_reftraj  = Vel_ref_traj[0, :]
surge_one_step = vel_one_step[0, :]

# Create a time vector: starting at 0 with a 0.1 second interval
N = surge_one_shot.size
time = np.arange(N) * 0.1

# Compute the error (actual velocity minus reference) for each controller
error_one_shot = surge_one_shot - surge_reftraj
error_one_step = surge_one_step - surge_reftraj

# Create subplots: first for surge speeds, second for errors
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Top subplot: Surge Speed Control Comparison
axs[0].plot(time, surge_reftraj, 'k--', linewidth=1.5, label='Reference Trajectory')
axs[0].plot(time, surge_one_shot, color='royalblue', linewidth=1, label='Multi-Step MLP')
axs[0].plot(time, surge_one_step, color='darkorange', linewidth=1, label='Single-Step MLP')
axs[0].set_title('Surge Speed Control: Comparison of Models')
axs[0].set_ylabel('Surge Speed (m/s)')
axs[0].legend()
axs[0].grid(True)

# Bottom subplot: Error (Controller Output - Reference)
axs[1].plot(time, error_one_shot, color='royalblue', linewidth=1, label='Error Multi-Step MLP')
axs[1].plot(time, error_one_step, color='darkorange', linewidth=1, label='Error Single-Step MLP')
axs[1].set_title('Surge Speed Control: Error (Actual - Reference)')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Error (m/s)')
axs[1].grid(True)

plt.tight_layout()
plt.savefig('comparison_surge_comparison.pdf', dpi=400)
plt.show()

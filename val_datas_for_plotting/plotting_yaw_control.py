import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the yaw control .mat files
one_shot_data = sio.loadmat('vel_yaw_control_one_shot.mat')
one_step_data = sio.loadmat('vel_yaw_control_one_step.mat')

# Extract the arrays with fixed variable names
vel_one_shot = one_shot_data['Vel_one_shot']     # shape: 3 x N
Vel_ref_traj = one_shot_data['Vel_ref']       # shape: 3 x N
vel_one_step = one_step_data['Vel_one_step']       # shape: 3 x N

# Extract the yaw data (third row)
yaw_one_shot = vel_one_shot[2, :]
yaw_reftraj  = Vel_ref_traj[2, :]
yaw_one_step = vel_one_step[2, :]

# Create a time vector: starting at 0 with a 0.1 second interval
N = yaw_one_shot.size
time = np.arange(N) * 0.1

# Compute the error (controller output minus reference) for each controller
error_one_shot = yaw_one_shot - yaw_reftraj
error_one_step = yaw_one_step - yaw_reftraj

# Create subplots: one for yaw control comparison and one for error plot
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Top subplot: Yaw Control Comparison
axs[0].plot(time, yaw_reftraj, 'k--', linewidth=1, label='Reference Trajectory')
axs[0].plot(time, yaw_one_shot, color='royalblue', linewidth=1, label='Multi-Step MLP')
axs[0].plot(time, yaw_one_step, color='darkorange', linewidth=1, label='Single-Step MLP')
axs[0].set_title('Yaw Control: Comparison of Controllers')
axs[0].set_ylabel('Yaw (rad/s)')
axs[0].legend()
axs[0].grid(True)

# Bottom subplot: Error (Controller Output - Reference)
axs[1].plot(time, error_one_shot, color='royalblue', linewidth=1, label='Error Multi-Step MLP')
axs[1].plot(time, error_one_step, color='darkorange', linewidth=1, label='Error Single-Step MLP')
axs[1].set_title('Yaw Control: Error (Actual - Reference)')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Error (rad/s)')
axs[1].grid(True)

plt.tight_layout()
plt.savefig('comparison_yaw_comparison.pdf', dpi=400)
plt.show()

plt.show()

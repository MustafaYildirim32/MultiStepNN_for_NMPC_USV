#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# User Settings: Adjust these indices to select the data segment to plot
# =============================================================================
START_INDEX = 0        # starting index (inclusive)
END_INDEX = None       # ending index (exclusive)

# =============================================================================
# Load the saved prediction data
# =============================================================================
predicted_one_step_linear = np.load('predicted_points_one_step_linear.npy')
predicted_one_shot_linear = np.load('predicted_points_one_shot_linear.npy')
predicted_one_shot_LSTM   = np.load('predicted_points_LSTM_one_shot.npy')
expected_pts              = np.load('expected_points.npy')

# Slice the arrays to use only the selected segment
predicted_one_step_linear = predicted_one_step_linear[START_INDEX:END_INDEX]
predicted_one_shot_linear = predicted_one_shot_linear[START_INDEX:END_INDEX]
predicted_one_shot_LSTM   = predicted_one_shot_LSTM[START_INDEX:END_INDEX]
expected_pts              = expected_pts[START_INDEX:END_INDEX]

# =============================================================================
# Create Time Axis
# =============================================================================
# Assuming each prediction corresponds to 0.1 seconds (adjust if necessary)
time_axis = np.arange(expected_pts.shape[0]) * 0.1

# =============================================================================
# Define Plotting Parameters, Colors, and Units
# =============================================================================
velocity_labels = ['Surge Velocity', 'Sway Velocity', 'Yaw Velocity']
velocity_units  = ['(m/s)', '(m/s)', '(rad/s)']  # Corresponding units for each velocity

# Chosen colors and line styles:
#   - Actual: Dark Slate Gray (solid)
#   - One-Step Linear: Royal Blue (dashed)
#   - One-Shot Linear: Dark Orange (dash-dot)
#   - One-Shot LSTM: Forest Green (dotted)

# =============================================================================
# Create and Customize Figure
# =============================================================================
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
line_width_prediction = 1.4
line_width_actual = 1.6

for i, label in enumerate(velocity_labels):
    ax = axes[i]
    ax.plot(time_axis, expected_pts[:, i], 
            color='darkslategray', linewidth=line_width_actual, label='Actual')
    ax.plot(time_axis, predicted_one_step_linear[:, i],
            linestyle=':', color='blue', linewidth=line_width_prediction, label='Single-Step MLP')
    ax.plot(time_axis, predicted_one_shot_linear[:, i],
            linestyle='-.', color='darkorange', linewidth=line_width_prediction, label='Multi-Step MLP')
    ax.plot(time_axis, predicted_one_shot_LSTM[:, i],
            linestyle='--', color='forestgreen', linewidth=line_width_prediction, label='Multi-Step LSTM')

    # Update y-label to include the unit
    ax.set_ylabel(f"{label} {velocity_units[i]}", fontsize=14)
    ax.set_title(f'{label} Comparison', fontsize=16)
    
    if i == 0:
        ax.legend(fontsize=12, loc='best')
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)", fontsize=14)
plt.tight_layout()

# Save a high-resolution version for publication

# =============================================================================
# Optional: Compute and Print Mean Squared Error (MSE) for Each Velocity Channel
# =============================================================================
print("\nMean Squared Error (MSE) on Validation Data:")
for i, label in enumerate(velocity_labels):
    mse_one_step  = np.mean((predicted_one_step_linear[:, i] - expected_pts[:, i])**2)
    mse_one_shot  = np.mean((predicted_one_shot_linear[:, i] - expected_pts[:, i])**2)
    mse_lstm      = np.mean((predicted_one_shot_LSTM[:, i] - expected_pts[:, i])**2)
    print(f"{label}: One-Step Linear = {mse_one_step:.6f}, "
          f"One-Shot Linear = {mse_one_shot:.6f}, "
          f"One-Shot LSTM = {mse_lstm:.6f}")


plt.savefig('comparison_velocity_predictions.pdf', dpi=300)
plt.show()

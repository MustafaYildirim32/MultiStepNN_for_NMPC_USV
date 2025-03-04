# Neural Network NMPC Control Repository

This repository implements neural network based system identification for Nonlinear Model Predictive Control (NMPC) in robotics. It is divided into two main components:

- **Train Side**: Contains Python scripts to train various neural network architectures.
- **Control MATLAB Side**: Contains MATLAB scripts to perform velocity and position control using NMPC, leveraging the trained neural network models.

---


---

## Train Side

The `train_side` directory contains the training scripts and data for various neural network architectures. In the paper, three architectures are presented:
- **Single-Step MLP** (`train_singlestep_mlp.py`)
- **Multi-Step MLP** (`train_multistep_mlp.py`)
- **Multi-Step LSTM** (`train_multistep_LSTM.py`)

Additionally, an extra architecture, **Multi-Step RNN**, is provided in `train_multistep_RNN.py`.

### Training Configuration

Each training script contains a `do_Train` setting:
- **True**: Trains a new model (even if a pre-trained model exists).
- **False**: Loads an existing model (if available) so you can directly examine the validation results.

### Data Files

- **Training Data**: `datas/data_train_gazebointerpolated.mat`  
  This file contains simulation data collected by applying different thruster signals (chirps, sinusoids with varying magnitudes, phase delays, and constant values) to the robot.
  
- **Validation Data**: `datas/datagazebodcthrust_smoothed_interp.mat`  
  This file is used for validating the performance of all neural network architectures.

---

## Control MATLAB Side

The `control_matlab_side` directory contains MATLAB scripts for implementing NMPC for both velocity and position control. These scripts use the trained neural network models as system identifiers.

### Available MATLAB Scripts

- **Position Control (NMPC with Multi-Step MLP)**:
  - `main_position_control_ode_model_multi_step_mlp.m`
  - `main_position_control_ros_model_multi_step_mlp.m`

- **Velocity Control**:
  - Multi-Step MLP based:
    - `main_velocity_control_ode_model_multi_step_mlp.m`
    - `main_velocity_control_ros_model_multi_step_mlp.m`
  - Single-Step MLP based:
    - `main_velocity_control_ode_model_one_step_predictor.m`
    - `main_velocity_control_ros_model_one_step_predictor.m`

### Docker & Simulation Notes

- The files with the `_ros_` suffix are designed to run in a Docker container that bridges MATLAB with ROS and includes simulation packages for ClearpathRobotics Heron.
- The files with the `_ode_` suffix are intended for users who prefer not to use the Docker container. In these cases, a fixed mathematical model (identified using MATLAB’s System Identification Toolbox) is used, closely approximating the real Heron model in simulation.

### Velocity Control Details

For velocity control, two different models are provided (Multi-Step MLP and Single-Step MLP). Users can configure reference velocities and their durations by adjusting the `velRefs` and `durations` variables in the respective scripts.

---

## Neural Network Preparation

Additional files (not explicitly shown in the folder tree) are provided for transferring weights and biases from the PyTorch models to MATLAB neural network models constructed using MATLAB’s own toolbox. This approach was chosen because MATLAB’s `importNetworkFromPyTorch` method does not support automatic differentiation of the cost function with respect to the network inputs unless the network is fully constructed within MATLAB. Note that NMPC requires an externally supplied gradient of the cost function for a successful solution.

---

## Usage

1. **Training Neural Networks**:
   - Navigate to the `train_side` folder.
   - Open the desired training script (e.g., `train_multistep_mlp.py`).
   - Adjust the `do_Train` flag as needed.
   - Execute the script to train the model or load an existing one and view validation results.

2. **Running NMPC Control**:
   - For MATLAB-based control, open the MATLAB script from the `control_matlab_side` folder.
   - Choose between the `_ros_` or `_ode_` version based on your setup.
   - Modify control parameters such as reference velocities or NMPC settings as required.
   - Run the script to observe the control performance.

---

## Requirements

- **Python Environment** (for training):
  - Python 3.1x
  - Pythorch
  - Scipy

- **MATLAB Environment** (for control):
  - (Optional) If using the `_ros_` scripts, install the ROS Toolbox for ROS integration
  - Docker (if running simulations with the provided Docker container)

---




# utils/normalizator.py

import torch
import os

class Normalizer:
    def __init__(self, min_vals_vel, max_vals_vel, min_vals_thrust, max_vals_thrust):
        # Initialize velocity attributes
        self.min_vals_vel = min_vals_vel
        self.max_vals_vel = max_vals_vel
        self.range_vals_vel = self.max_vals_vel - self.min_vals_vel

        # Initialize thrust attributes
        self.thrust_bias = -1
        self.vel_bias = -1
        self.min_vals_thrust = min_vals_thrust
        self.max_vals_thrust = max_vals_thrust
        self.range_vals_thrust = self.max_vals_thrust - self.min_vals_thrust
    def normalize_vel_11(self, vel):
        min_vals = self.min_vals_vel.to(vel.device)
        range_vals = self.range_vals_vel.to(vel.device)
        # Convert the bias to a tensor
        vel_bias = torch.tensor(self.vel_bias, device=vel.device, dtype=vel.dtype)
        return 2 * (vel - min_vals) / range_vals + vel_bias

    def normalize_thrust_11(self, thrust):
        min_vals = self.min_vals_thrust.to(thrust.device)
        range_vals = self.range_vals_thrust.to(thrust.device)
        # Convert the bias to a tensor
        thrust_bias = torch.tensor(self.thrust_bias, device=thrust.device, dtype=thrust.dtype)
        return 2 * (thrust - min_vals) / range_vals + thrust_bias
    def denormalize_vel_11(self, vel):
        return ((vel+1)*self.range_vals_vel/2)+self.min_vals_vel
    def denormalize_thrust_11(self, vel):
        return ((vel+1)*self.range_vals_thrust/2)+self.min_vals_thrust
    def normalize_all_11(self, input):
        vel = input[:, :3]
        thrust = input[:, 3:5]
        extra = torch.tensor([[0.1]], dtype=input.dtype, device=input.device)
        return torch.cat((self.normalize_vel_11(vel), self.normalize_thrust_11(thrust), extra), dim=1)
    def print_all_attributes(self):
        print(f"min_vals_vel: {self.min_vals_vel}")
        print(f"max_vals_vel: {self.max_vals_vel}")
        print(f"range_vals_vel: {self.range_vals_vel}")
        print(f"min_vals_thrust: {self.min_vals_thrust}")
        print(f"max_vals_thrust: {self.max_vals_thrust}")
        print(f"range_vals_thrust: {self.range_vals_thrust}")
    
    

import numpy as np
import torch
import scipy.io

def load_data(data_path=r'datas/data1gazebointerpolated.mat'):
    data_train = scipy.io.loadmat(data_path)
    thrust_obs = torch.tensor(data_train['simulated_thrusts'], dtype=torch.float32) 
    thrust_obs = thrust_obs[:,:]  
    vel_obs = torch.tensor(data_train['simulated_vel'], dtype=torch.float32) 
    Ts = torch.tensor(0.1, dtype=torch.float32)  
 
    thrust_obs = thrust_obs.T 
    vel_obs = vel_obs.T  

    # Print shapes and types
    #print(f'thrust_obs shape: {thrust_obs.shape}, type: {type(thrust_obs)}')
    #print(f'vel_obs shape: {vel_obs.shape}, type: {type(vel_obs)}')
    #print(f'Ts shape: {Ts.shape}, type: {type(Ts)}')

    return vel_obs, thrust_obs

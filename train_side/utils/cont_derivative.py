import numpy as np

def cont_derivatives(x, thrust):
    dx = np.zeros_like(x)
    dx[0] = -0.55245 * x[0] + (-0.00653252) * (x[2]**2) + 0.869158 * x[1] * x[2] + \
            (-0.0833071) * abs(x[0]) * x[0] + 0.0320901 * np.sum(thrust)
    dx[1] = -0.57293 * x[1] + (-0.0726335) * x[2] + (-0.119536) * abs(x[1]) * x[1] + \
            0.0896787 * abs(x[2]) * x[2] + (-0.000300261) * (thrust[0] - thrust[1]) + \
            (-1.06866) * x[0] * x[2] + (-0.0177022) * x[0] * x[1]
    dx[2] = 0.028676 * x[1] + (-0.49859) * x[2] + 0.0112123 * abs(x[1]) * x[1] + \
            (-0.435785) * abs(x[2]) * x[2] + 0.0325477 * (thrust[0] - thrust[1]) + \
            0.0600072 * x[0] * x[2] + 0.0077453 * x[0] * x[1]
    return dx

def ode4_step(x0, thrust, dt):
    k1 = cont_derivatives(x0, thrust)
    k2 = cont_derivatives(x0 + 0.5 * dt * k1, thrust)
    k3 = cont_derivatives(x0 + 0.5 * dt * k2, thrust)
    k4 = cont_derivatives(x0 + dt * k3, thrust)
    return x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def ode4_solution_init_x_and_thrust_array(x_initial, thrust_array, dt, t_final):
    num_steps = int(t_final / dt)
    x_values = np.zeros((num_steps + 1, len(x_initial)))
    x_values[0] = x_initial

    for t in range(num_steps):
        x_values[t + 1] = ode4_step(x_values[t], thrust_array[t], dt)
    
    return x_values

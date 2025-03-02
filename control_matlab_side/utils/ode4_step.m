function x_new = ode4_step(x0, thrust, dt)
    % ODE4_STEP Perform a single Runge-Kutta 4th order integration step.
    %
    %   x_new = ODE4_STEP(x0, thrust, dt) computes the new state vector
    %   x_new after a time step dt using the RK4 method.
    %
    %   Parameters:
    %       x0      - Current state vector (3x1 vector).
    %       thrust  - Thrust inputs (2x1 vector).
    %       dt      - Time step size (scalar).
    %
    %   Returns:
    %       x_new  - Updated state vector after time step dt (3x1 vector).

    % Compute the RK4 coefficients
    k1 = cont_derivatives(x0, thrust);
    k2 = cont_derivatives(x0 + 0.5 * dt * k1, thrust);
    k3 = cont_derivatives(x0 + 0.5 * dt * k2, thrust);
    k4 = cont_derivatives(x0 + dt * k3, thrust);

    % Update the state vector
    x_new = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end

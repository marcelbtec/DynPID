import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------
# PID Controller module
# -------------------------------
class PIDController(nn.Module):
    """
    A learnable PID (Proportional-Integral-Derivative) controller implemented as a PyTorch module.
    
    This controller implements the discrete-time PID control law with learnable gains:
        u[k] = Kp * e[k] + Ki * Σ(e[i] * dt) + Kd * (e[k] - e[k-1])/dt
    
    Attributes:
        Kp (nn.Parameter): Proportional gain
        Ki (nn.Parameter): Integral gain
        Kd (nn.Parameter): Derivative gain
        integral (torch.Tensor): Current integral term
        prev_error (torch.Tensor): Previous error for derivative calculation
    
    Args:
        init_Kp (float, optional): Initial proportional gain. Defaults to 0.05.
        init_Ki (float, optional): Initial integral gain. Defaults to 0.05.
        init_Kd (float, optional): Initial derivative gain. Defaults to 0.0.
    """
    def __init__(self, init_Kp=0.05, init_Ki=0.05, init_Kd=0.0):
        super(PIDController, self).__init__()
        self.Kp = nn.Parameter(torch.tensor(init_Kp, dtype=torch.float32))
        self.Ki = nn.Parameter(torch.tensor(init_Ki, dtype=torch.float32))
        self.Kd = nn.Parameter(torch.tensor(init_Kd, dtype=torch.float32))
        self.reset()
        
    def reset(self):
        """
        Reset the controller's internal state (integral and previous error).
        Should be called at the start of each new control sequence.
        """
        self.integral = None
        self.prev_error = None
        
    def forward(self, measurement, setpoint):
        """
        Compute the control output using the PID control law.
        
        Args:
            measurement (torch.Tensor): Current system measurement
            setpoint (torch.Tensor): Desired setpoint value
            
        Returns:
            torch.Tensor: Control output signal
        """
        dt = 0.1  # time step
        if self.integral is None:
            self.integral = torch.zeros_like(measurement)
        if self.prev_error is None:
            self.prev_error = torch.zeros_like(measurement)
        error = setpoint - measurement
        self.integral = self.integral + error * dt
        self.integral = torch.clamp(self.integral, -5.0, 5.0)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

# -----------------------------------------
# Adaptive PID Network Controller V2 with Internal Setpoint Adaptation
# -----------------------------------------
class AdaptivePIDNetworkControllerV2(nn.Module):
    """
    An adaptive PID controller network with dynamic setpoint adaptation.
    
    This controller combines multiple PID controllers with neural networks for
    setpoint generation and adaptation. It can handle changing setpoints and
    adapt its control strategy in real-time.
    
    The controller architecture consists of:
    1. A setpoint generation network
    2. Multiple PID controllers
    3. A dynamic setpoint adaptation mechanism
    
    Args:
        num_controllers (int): Number of parallel PID controllers
        input_dim (int): Dimension of input features
        steps (int, optional): Number of inner-loop iterations. Defaults to 25.
        alpha (float, optional): Setpoint adaptation rate (0-1). Defaults to 0.5.
    """
    def __init__(self, num_controllers, input_dim, steps=25, alpha=0.5):
        super(AdaptivePIDNetworkControllerV2, self).__init__()
        self.num_controllers = num_controllers
        self.steps = steps
        self.alpha = alpha
        
        # Feedforward (static setpoint) network: maps the input [current_state, desired_setpoint]
        # to a vector (one per controller).
        self.setpoint_network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_controllers)
        )
        
        # Dynamic setpoint network: updates the setpoint based on [static_setpoint, measurement].
        self.dynamic_setpoint_network = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        
        # Register PID controllers (each with learnable Kp, Ki, Kd).
        self.controllers = nn.ModuleList([PIDController() for _ in range(num_controllers)])
    
    def forward(self, x):
        """
        Forward pass of the controller network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim)
            
        Returns:
            torch.Tensor: Aggregated control signal
        """
        ext_setpoint = x[:, 1]  # external setpoint from input
        static_setpoints = self.setpoint_network(x)  # shape: (batch, num_controllers)
        controller_outputs = []
        for i, controller in enumerate(self.controllers):
            controller.reset()  # reset PID state for each new input
            measurement = torch.zeros_like(static_setpoints[:, i])
            dynamic_setpoint = static_setpoints[:, i]  # initialize dynamic setpoint
            for _ in range(self.steps):
                measurement = controller(measurement, dynamic_setpoint)
                inp = torch.stack([static_setpoints[:, i], measurement], dim=1)
                dynamic_update = self.dynamic_setpoint_network(inp).squeeze(1)
                dynamic_setpoint = (1 - self.alpha) * dynamic_update + self.alpha * ext_setpoint
            controller_outputs.append(measurement.unsqueeze(1))
        combined = torch.cat(controller_outputs, dim=1)  # shape: (batch, num_controllers)
        aggregated_output = combined.mean(dim=1)
        return aggregated_output

    def forward_with_full_dynamics(self, x, steps=None):
        """
        Forward pass with full dynamics recording.
        
        Records the internal dynamics of the controller, including:
        - Measurement evolution
        - Setpoint adaptation
        - Dynamic updates
        
        Args:
            x (torch.Tensor): Input tensor
            steps (int, optional): Number of steps to record. Defaults to None.
            
        Returns:
            tuple: (aggregated_output, measurement_dynamics, setpoint_dynamics, dynamic_update_dynamics)
        """
        if steps is None:
            steps = self.steps
        ext_setpoint = x[:, 1]
        static_setpoints = self.setpoint_network(x)
        measurement_dynamics = []
        setpoint_dynamics = []
        dynamic_update_dynamics = []
        controller_outputs = []
        for i, controller in enumerate(self.controllers):
            controller.reset()
            measurement = torch.zeros_like(static_setpoints[:, i])
            dynamic_setpoint = static_setpoints[:, i]
            meas_dyn = []
            sp_dyn = []
            dyn_up_dyn = []
            for _ in range(steps):
                measurement = controller(measurement, dynamic_setpoint)
                meas_dyn.append(measurement.unsqueeze(1))
                sp_dyn.append(dynamic_setpoint.unsqueeze(1))
                inp = torch.stack([static_setpoints[:, i], measurement], dim=1)
                dynamic_update = self.dynamic_setpoint_network(inp).squeeze(1)
                dyn_up_dyn.append(dynamic_update.unsqueeze(1))
                dynamic_setpoint = (1 - self.alpha) * dynamic_update + self.alpha * ext_setpoint
            measurement_dynamics.append(torch.cat(meas_dyn, dim=1))
            setpoint_dynamics.append(torch.cat(sp_dyn, dim=1))
            dynamic_update_dynamics.append(torch.cat(dyn_up_dyn, dim=1))
            controller_outputs.append(measurement.unsqueeze(1))
        combined = torch.cat(controller_outputs, dim=1)
        aggregated_output = combined.mean(dim=1)
        return aggregated_output, measurement_dynamics, setpoint_dynamics, dynamic_update_dynamics

# -------------------------------
# Differentiable simulation for training
# -------------------------------
def simulate_controller_train(controller, x0, r_initial, r_new, T, disturbance_step):
    """
    Differentiable simulation of the plant dynamics for training.
    
    Simulates a first-order system with the dynamics:
        x[t+1] = a * x[t] + u[t]
    where u[t] is the controller output.
    
    Args:
        controller (AdaptivePIDNetworkControllerV2): The controller to simulate
        x0 (float): Initial state
        r_initial (float): Initial setpoint
        r_new (float): New setpoint after disturbance
        T (int): Total simulation time steps
        disturbance_step (int): Time step at which setpoint changes
        
    Returns:
        tuple: (state_history, setpoint_history)
    """
    a = 0.9  # plant coefficient
    # Initialize state as a differentiable tensor.
    x = torch.tensor([x0], dtype=torch.float32)
    xs = []
    rs = []
    for t in range(T):
        r = r_initial if t < disturbance_step else r_new
        rs.append(r)
        # Construct input tensor: shape (1, 2)
        inp = torch.cat([x, torch.tensor([r], dtype=torch.float32, device=x.device)], dim=0).unsqueeze(0)
        u = controller(inp)  # differentiable control signal
        x = a * x + u        # update state
        xs.append(x)
    xs = torch.stack(xs).squeeze()  # shape: (T,)
    rs = torch.tensor(rs, dtype=torch.float32)
    return xs, rs

# -------------------------------
# Simulation for evaluation (non-differentiable)
# -------------------------------
def simulate_controller(controller, x0, r_initial, r_new, T, disturbance_step):
    """
    Non-differentiable simulation of the plant dynamics for evaluation.
    
    Similar to simulate_controller_train but without gradient tracking.
    Used for evaluation and visualization.
    
    Args:
        controller (AdaptivePIDNetworkControllerV2): The controller to simulate
        x0 (float): Initial state
        r_initial (float): Initial setpoint
        r_new (float): New setpoint after disturbance
        T (int): Total simulation time steps
        disturbance_step (int): Time step at which setpoint changes
        
    Returns:
        tuple: (state_history, setpoint_history, control_history)
    """
    a = 0.9  # plant coefficient
    x = x0
    xs, rs, us = [], [], []
    for t in range(T):
        r = r_initial if t < disturbance_step else r_new
        rs.append(r)
        inp = torch.tensor([[x, r]], dtype=torch.float32)
        with torch.no_grad():
            u = controller(inp).item()
        us.append(u)
        x = a * x + u
        xs.append(x)
    return np.array(xs), np.array(rs), np.array(us)

# -------------------------------
# Main: Train, then evaluate and analyze controller dynamics
# -------------------------------
if __name__ == "__main__":
    # Simulation parameters
    T = 150
    x0 = 0.0
    r_initial = 1.0   # initial setpoint
    r_new = 0.5       # new setpoint after disturbance
    disturbance_step = 70 

    # Instantiate the controller.
    controller_v2 = AdaptivePIDNetworkControllerV2(num_controllers=3, input_dim=2, steps=20, alpha=0.8)
    
    # ---- Training: Use differentiable simulation to update the controller parameters ----
    num_epochs = 80
    optimizer = torch.optim.Adam(controller_v2.parameters(), lr=0.008)
    loss_fn = nn.MSELoss()
    
    # Record the PID gain history over training
    kp_history = []
    ki_history = []
    kd_history = []
    loss_history = []
    
    controller_v2.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        xs_train, rs_train = simulate_controller_train(controller_v2, x0, r_initial, r_new, T, disturbance_step)
        loss = loss_fn(xs_train, rs_train)
        loss.backward()
        optimizer.step()
        
        kp_history.append([c.Kp.item() for c in controller_v2.controllers])
        ki_history.append([c.Ki.item() for c in controller_v2.controllers])
        kd_history.append([c.Kd.item() for c in controller_v2.controllers])
        loss_history.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # ---- Evaluation: Check the closed-loop response with the trained controller ----
    controller_v2.eval()
    xs, rs, us = simulate_controller(controller_v2, x0, r_initial, r_new, T, disturbance_step)
    steps_range = np.arange(T)
    plt.figure(figsize=(10, 6))
    plt.plot(steps_range, xs, marker='o', label='Plant State')
    plt.plot(steps_range, rs, marker='x', linestyle='--', label='Reference')
    plt.axvline(x=disturbance_step, color='red', linestyle=':', label='Reference Change')
    plt.xlabel('Time Step')
    plt.ylabel('State / Setpoint')
    plt.title('Closed-Loop Response (After Training)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ---- Inner-Loop Dynamics: Detailed view for one sample input ----
    sample_input = torch.tensor([[x0, r_initial]], dtype=torch.float32)
    with torch.no_grad():
        agg_out, meas_dyn_list, sp_dyn_list, dyn_up_list = controller_v2.forward_with_full_dynamics(sample_input, steps=15)
    
    static_setpoints = controller_v2.setpoint_network(sample_input).detach().cpu().numpy().flatten()
    num_channels = controller_v2.num_controllers
    steps_inner = np.arange(1, 16)
    
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 3*num_channels), sharex=True)
    if num_channels == 1:
        axs = [axs]
    
    for i in range(num_channels):
        meas_dyn = meas_dyn_list[i].detach().cpu().numpy().flatten()
        sp_dyn = sp_dyn_list[i].detach().cpu().numpy().flatten()
        dyn_up = dyn_up_list[i].detach().cpu().numpy().flatten()
        axs[i].plot(steps_inner, meas_dyn, marker='o', label='PID Measurement')
        axs[i].plot(steps_inner, sp_dyn, marker='x', label='Dynamic Setpoint')
        axs[i].plot(steps_inner, dyn_up, marker='s', label='Dynamic Update')
        axs[i].hlines(static_setpoints[i], steps_inner[0], steps_inner[-1], colors='gray', linestyles='--', label='Static Setpoint')
        axs[i].set_ylabel('Value')
        axs[i].set_title(f'Controller Channel {i+1}')
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel('Inner-Loop Iteration')
    plt.tight_layout()
    plt.show()
    
    # ---- Plot the training dynamics of PID gains ----
    epochs = np.arange(1, num_epochs+1)
    kp_history = np.array(kp_history)  # shape: (num_epochs, num_controllers)
    ki_history = np.array(ki_history)
    kd_history = np.array(kd_history)
    
    plt.figure(figsize=(10, 6))
    for i in range(controller_v2.num_controllers):
        plt.plot(epochs, kp_history[:, i], label=f'Kp - Controller {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Kp Value')
    plt.title('Kp Dynamics During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for i in range(controller_v2.num_controllers):
        plt.plot(epochs, ki_history[:, i], label=f'Ki - Controller {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Ki Value')
    plt.title('Ki Dynamics During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for i in range(controller_v2.num_controllers):
        plt.plot(epochs, kd_history[:, i], label=f'Kd - Controller {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Kd Value')
    plt.title('Kd Dynamics During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ---- Final Gains: Plot final learned PID gains ----
    final_Kp = [controller.Kp.item() for controller in controller_v2.controllers]
    final_Ki = [controller.Ki.item() for controller in controller_v2.controllers]
    final_Kd = [controller.Kd.item() for controller in controller_v2.controllers]
    channels = np.arange(1, num_channels+1)
    
    width = 0.2
    plt.figure(figsize=(8, 5))
    plt.bar(channels - width, final_Kp, width=width, label='Kp')
    plt.bar(channels, final_Ki, width=width, label='Ki')
    plt.bar(channels + width, final_Kd, width=width, label='Kd')
    plt.xlabel('Controller Channel')
    plt.ylabel('Gain Value')
    plt.title('Final Learned PID Gains')
    plt.xticks(channels)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Optionally, plot the loss evolution over epochs.
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, num_epochs+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

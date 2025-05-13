# Adaptive PID Neural Network Controller

This repository implements an advanced PID (Proportional-Integral-Derivative) controller that combines traditional control theory with neural networks for enhanced performance and adaptability. The implementation features a novel architecture that enables dynamic setpoint adaptation and real-time learning capabilities.

## Overview

The implementation consists of two main components:

1. **PIDController**: A learnable PID controller implemented as a PyTorch module
2. **AdaptivePIDNetworkControllerV2**: A sophisticated controller network that combines multiple PID controllers with neural networks for setpoint adaptation

## Mathematical Foundation

### PID Control Law
The discrete-time PID control law is implemented as:

\[ u[k] = K_p \cdot e[k] + K_i \cdot \sum_{i=0}^{k} e[i] \cdot \Delta t + K_d \cdot \frac{e[k] - e[k-1]}{\Delta t} \]

where:
- \(u[k]\) is the control output at time step k
- \(e[k]\) is the error (setpoint - measurement)
- \(K_p, K_i, K_d\) are learnable gains
- \(\Delta t\) is the time step (0.1s in this implementation)

### System Dynamics
The controlled system is modeled as a first-order system:

\[ x[t+1] = a \cdot x[t] + u[t] \]

where:
- \(x[t]\) is the system state
- \(a\) is the system coefficient (0.9 in this implementation)
- \(u[t]\) is the control input

## Architecture

### PIDController
- Implements the standard PID control law with learnable gains
- Features anti-windup protection for the integral term
- Maintains internal state for integral and derivative calculations

### AdaptivePIDNetworkControllerV2
The controller network consists of three main components:

1. **Setpoint Generation Network**:
   - Input: [current_state, desired_setpoint]
   - Architecture: 16 → 8 → num_controllers neurons
   - Output: Initial setpoints for each controller

2. **Dynamic Setpoint Network**:
   - Input: [static_setpoint, measurement]
   - Architecture: 2 → 3 → 1 neurons
   - Output: Dynamic setpoint updates

3. **Multiple PID Controllers**:
   - Parallel PID controllers with independent gains
   - Aggregated output for final control signal

## Features

- **Learnable PID Gains**: All controller parameters are optimized through backpropagation
- **Dynamic Setpoint Adaptation**: Real-time adjustment of setpoints based on system response
- **Multiple Control Channels**: Parallel PID controllers for enhanced control capabilities
- **Anti-windup Protection**: Prevents integral term saturation
- **Differentiable Simulation**: Enables end-to-end training of the entire control system

## Usage

### Basic Usage
```python
# Initialize the controller
controller = AdaptivePIDNetworkControllerV2(
    num_controllers=3,
    input_dim=2,
    steps=20,
    alpha=0.8
)

# Training setup
optimizer = torch.optim.Adam(controller.parameters(), lr=0.008)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    xs_train, rs_train = simulate_controller_train(
        controller, x0, r_initial, r_new, T, disturbance_step
    )
    loss = loss_fn(xs_train, rs_train)
    loss.backward()
    optimizer.step()
```

### Evaluation
```python
# Evaluate the controller
controller.eval()
xs, rs, us = simulate_controller(
    controller, x0, r_initial, r_new, T, disturbance_step
)
```

## Visualization

The implementation includes comprehensive visualization capabilities:

1. **Closed-Loop Response**: System state vs. reference trajectory
2. **Inner-Loop Dynamics**: Detailed view of controller behavior
3. **Gain Evolution**: Training dynamics of PID gains
4. **Loss Progression**: Training loss over epochs

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.19.2
- Matplotlib >= 3.3.2

## Performance Metrics

The controller's performance is evaluated using:

1. **Tracking Error**: Mean squared error between system state and reference
2. **Response Time**: Time to reach and maintain setpoint
3. **Overshoot**: Maximum deviation from setpoint
4. **Steady-State Error**: Final error after system stabilization

## Applications

This implementation is particularly suitable for:

1. **Process Control**: Temperature, pressure, or flow control in industrial processes
2. **Robotics**: Position and velocity control of robotic systems
3. **Power Systems**: Voltage and frequency regulation
4. **Chemical Reactors**: Temperature and concentration control

## Future Work

Potential extensions and improvements:

1. **Reinforcement Learning Integration**: Combine with RL for optimal control
2. **Model Predictive Control**: Add predictive capabilities
3. **Multi-Objective Optimization**: Consider multiple performance metrics
4. **Online Adaptation**: Real-time parameter adjustment
5. **Distributed Control**: Extend to multi-agent systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{adaptive_pid_controller,
  author = {Marcel Blattner},
  title = {Adaptive PID Neural Network Controller},
  year = {2024},
  url = {https://github.com/marcelbtec/DynPID}
}


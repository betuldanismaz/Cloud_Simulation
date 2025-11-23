# CloudSim: RL-Based Cloud Auto-Scaling Simulation

CloudSim is a simulation environment for experimenting with cloud resource auto-scaling using both rule-based and reinforcement learning (RL) approaches. The project features a custom OpenAI Gymnasium environment and uses Deep Q-Networks (DQN) to learn optimal scaling policies for cloud servers.

## Features

- **Custom Cloud Environment**: Simulates VM scaling, CPU load, and dynamic request patterns.
- **Rule-Based Baseline**: Implements a threshold-based scaling agent for comparison.
- **Reinforcement Learning Agent**: Trains a DQN agent to optimize scaling decisions.
- **Performance Evaluation**: Compares RL and rule-based agents with detailed metrics and plots.

## Project Structure

```
cloud_env.py              # Custom Gymnasium environment for cloud scaling
train.py                  # RL agent training script (DQN)
evaluate.py               # Evaluate and visualize RL agent performance
evaluate_comparison.py    # Compare RL agent with rule-based baseline
```

## Requirements

- Python 3.8+
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- numpy
- matplotlib

Install dependencies with:

```sh
pip install stable-baselines3 gymnasium numpy matplotlib
```

## Usage

### 1. Train the RL Agent

Run the training script to train a DQN agent on the cloud environment:

```sh
python train.py
```

This will save the trained model as `dqn_cloud_autoscale.zip`.

### 2. Evaluate RL Agent

Visualize the RL agent's scaling performance:

```sh
python evaluate.py
```

A plot will show CPU usage and VM count over time.

### 3. Compare with Rule-Based Agent

Compare the RL agent with a classic threshold-based agent:

```sh
python evaluate_comparison.py
```

This script prints a detailed performance report and displays comparison plots.

## Environment Details

The environment (`cloud_env.py`) models:

- **State**: `[CPU usage %, VM count, incoming requests]`
- **Actions**: `0=Scale In`, `1=Do Nothing`, `2=Scale Out`
- **Reward**: Penalizes SLA violations (CPU > 80%), resource cost, and inefficiency (CPU < 40%). Rewards efficient scaling in.

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)

---

**Author:**  
Your Name  
2025

# SafeNav-RL

**Safety-Constrained Reinforcement Learning for Assistive Robot Navigation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gym](https://img.shields.io/badge/Gym-0.26+-green.svg)](https://gymnasium.farama.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

SafeNav-RL is a modular reinforcement learning framework for **safe, uncertainty-aware navigation** of assistive robots (e.g., smart wheelchairs) in dynamic environments with obstacles.

This project extends trajectory-level safety analysis (see [SafeTraj-Prototype](https://github.com/pouyapd/SafeTraj-Prototype)) by training RL agents that are **intrinsically safety-constrained** — not just evaluated post-hoc, but rewarded and penalized in a way that builds collision-avoidance into the learned policy.

### Research Motivation

Existing neural trajectory predictors (e.g., Social Force, LSTM-based models) may fail silently in novel scenarios. Rather than only detecting failures after the fact, this project asks:

> *Can we train a navigation policy that learns to be safe from first principles, with explicit collision constraints and uncertainty-aware decision making?*

This is directly relevant to assistive robotics, autonomous wheelchairs, and any safety-critical mobile robot deployment.

---

## Key Features

| Feature | Description |
|---|---|
| **PPO with Safety Layer** | Proximal Policy Optimization with a differentiable safety projection that enforces collision-avoidance constraints |
| **Curriculum Learning** | Progressive difficulty — obstacle density and goal distance increase as the agent improves |
| **Uncertainty-Aware Rewards** | Risk scoring integrated into reward shaping based on proximity and trajectory history |
| **Domain Randomization** | Randomized obstacle layouts, robot start positions, and sensor noise for robust sim-to-real transfer |
| **ROS2 Integration Scaffold** | Ready-to-deploy ROS2 node that wraps the trained policy for real robot deployment |
| **Research-Grade Evaluation** | Metrics: success rate, collision rate, trajectory smoothness, goal-reach time |

---

## Architecture

```
SafeNav-RL/
├── configs/
│   └── default_config.yaml       # All hyperparameters centralized
├── env/
│   ├── navigation_env.py         # Custom Gym-compatible environment
│   ├── obstacle_map.py           # Randomized obstacle generation
│   └── curriculum.py             # Progressive difficulty scheduler
├── models/
│   ├── actor_critic.py           # PPO Actor-Critic networks (PyTorch)
│   └── safety_layer.py           # Differentiable safety projection layer
├── training/
│   ├── ppo_trainer.py            # Full PPO training pipeline
│   ├── rollout_buffer.py         # Experience collection buffer
│   └── callbacks.py              # Logging, checkpointing, early stopping
├── evaluation/
│   ├── evaluator.py              # Metrics computation
│   └── visualizer.py             # Trajectory plots and animation
├── ros2_integration/
│   ├── nav_agent_node.py         # ROS2 node wrapping trained policy
│   └── topic_definitions.py      # ROS2 message/topic definitions
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   └── visualize_trajectory.py   # Visualization entry point
└── tests/
    ├── test_env.py
    └── test_models.py
```

---

## Installation

```bash
git clone https://github.com/pouyapd/SafeNav-RL.git
cd SafeNav-RL
pip install -r requirements.txt
```

For ROS2 deployment (optional):
```bash
# Requires ROS2 Humble: https://docs.ros.org/en/humble/Installation.html
pip install rclpy
```

---

## Quick Start

### Train
```bash
python scripts/train.py --config configs/default_config.yaml
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --episodes 50
```

### Visualize a trajectory
```bash
python scripts/visualize_trajectory.py --checkpoint checkpoints/best_model.pt
```

---

## Environment

The robot operates in a bounded 2D workspace with randomly placed circular obstacles.

**Observation space (7D):**
- `(x, y)` — robot position
- `θ` — heading angle
- `d_goal` — normalized distance to goal
- `Δθ_goal` — angle error to goal
- `risk_score` — local collision risk (from proximity sensor simulation)

**Action space (continuous, 2D):**
- `v ∈ [0, 1]` — linear velocity
- `ω ∈ [-1, 1]` — angular velocity

**Robot dynamics (differential-drive):**
```
x_{t+1} = x_t + v·cos(θ)·Δt
y_{t+1} = y_t + v·sin(θ)·Δt
θ_{t+1} = θ_t + ω·Δt
```

**Reward function:**
```
r = -0.01                         # step penalty (efficiency)
  + α · (d_prev - d_curr)         # distance shaping
  - β · risk_score                 # safety penalty
  + 10.0  [if goal reached]        # success bonus
  - 5.0   [if collision]           # collision penalty
```

---

## Safety Layer

The safety layer projects actions into a constraint-satisfying set using a differentiable quadratic program (QP):

```
min  ||a - a_nominal||²
s.t. h(s, a) ≥ 0   ∀ obstacles in range
```

Where `h(s, a)` is a Control Barrier Function (CBF) approximation that ensures the robot maintains a safe distance from obstacles. This is inspired by CBF-based safety filters used in real robotic control.

---

## Curriculum Learning

Training uses three stages of increasing difficulty:

| Stage | Obstacles | Goal Distance | Max Steps |
|---|---|---|---|
| 1 — Beginner | 2–4 | 3–5 m | 200 |
| 2 — Intermediate | 4–8 | 5–8 m | 300 |
| 3 — Expert | 8–15 | 5–12 m | 400 |

Stage advancement requires 70% success rate over the last 100 episodes.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Success Rate** | % of episodes where goal is reached without collision |
| **Collision Rate** | % of episodes ending in collision |
| **Avg. Path Length** | Mean trajectory length normalized by straight-line distance |
| **Avg. Risk Score** | Mean proximity risk over trajectory |
| **Goal Reach Time** | Average steps to reach goal (successful episodes only) |

---

## ROS2 Deployment

The trained policy can be deployed on a real robot using the provided ROS2 node scaffold:

```bash
# After sourcing your ROS2 workspace
ros2 run safenav_rl nav_agent_node --ros-args -p model_path:=checkpoints/best_model.pt
```

**Topics:**
- Subscribes: `/odom` (nav_msgs/Odometry), `/scan` (sensor_msgs/LaserScan), `/goal_pose` (geometry_msgs/PoseStamped)
- Publishes: `/cmd_vel` (geometry_msgs/Twist)

---

## Connection to SafeTraj-Prototype

This project is part of a broader research line on **safety-critical autonomous navigation**:

1. [SafeTraj-Experiments](https://github.com/pouyapd/SafeTraj-Experiments) — 
   *MSc thesis*: trajectory-level evaluation of pretrained DNN-LNA neural models 
   for autonomous wheelchair navigation (University of Genoa / CNR-IEIIT / REXASI-PRO)
2. [SafeTraj-Prototype](https://github.com/pouyapd/SafeTraj-Prototype) — 
   *Toolkit*: modular Python framework for trajectory behaviour analysis and risk scoring
3. **SafeNav-RL** (this repo) — 
   *Extension*: RL navigation agent with intrinsic safety constraints, 
   building on the failure analysis findings


Together they form a pipeline from failure analysis → safety-aware policy learning.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{safenav_rl,
  author = {Bathaei Pourmand, Pouya},
  title = {SafeNav-RL: Safety-Constrained RL for Assistive Robot Navigation},
  year = {2025},
  url = {https://github.com/pouyapd/SafeNav-RL}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Developed as part of research in safety-critical AI and autonomous navigation at the University of Genoa.*

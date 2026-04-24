# Deep RL Robot Arm

A modern reinforcement learning environment for training robotic manipulation skills using the **Franka Panda 7-DOF arm** with a 2-finger gripper in a simulated office workspace.

## 🎯 What's This About?

This project combines **MuJoCo physics simulation** with **Gymnasium** (gym-compatible) to create a challenging RL environment where an agent learns to autonomously reach and grasp objects on a desk. It's perfect for exploring RL algorithms like PPO, SAC, or TD3 applied to continuous control.

**The Challenge:** The robot starts at a fixed base position on the desk corner and must learn to navigate its 7-DOF arm to reach a randomly-placed cup on the tabletop surface, then grasp it.

---

## 🤖 Environment Details

### Robot
- **Franka Panda 7-DOF arm** with 2-finger parallel gripper
- Base position: corner of the desk (stable, on-table placement)
- Action space: 8 dimensions (7 arm joint velocities + 1 gripper control)
- Actuator ranges: Joint limits ±2.8973 rad; gripper 0–0.04 m

### Observation Space (19D)
- Joint positions (7) + joint velocities (7)
- Target (cup) XYZ position (3)
- Gripper opening state (1) [normalized 0–1]

### Reward Function
- **Distance reward:** `-1.0 * ||end_effector - target||`
- **Reaching bonuses:** +0.5 @ 20cm, +1.0 @ 10cm, +3.0 @ 5cm
- **Grasp incentive:** +0.2 × (gripper closure) when within 5cm
- **Control penalty:** -0.0001 × ||actions|| (smooth motion)
- **Collision penalty:** -0.1 × collision_count
- **Self-collision penalty:** -0.5 per frame (arm hitting itself)

### Workspace
- Office-like scene with walls, desk, obstacles (bookshelf, monitor stand, cabinet)
- Desk surface: 1.0m wide × 0.6m deep at height 0.75m
- Cup spawns randomly on tabletop (0.20m minimum clearance from robot base)
- Procedurally generated checker-pattern floor with ceiling lighting

---

## 🧠 Reinforcement Learning Concepts

### Why This Environment?

1. **Curriculum Learning Opportunity**
   - Start with large sparse rewards (reach target distance)
   - Progressively add grasp/manipulation objectives
   - Add obstacle avoidance constraints later

2. **Exploration vs. Exploitation**
   - Continuous action space requires sophisticated exploration
   - Random cup spawning prevents reward hacking from memorizing positions

3. **Sim-to-Real Transfer**
   - MuJoCo physics are reasonably realistic for tabletop manipulation
   - Gripper control and contact dynamics matter for grasping

4. **Multi-Objective Learning**
   - Reaching (position control)
   - Grasping (force/closure control)
   - Collision avoidance (implicit in reward)

### Algorithms to Try
- **PPO** (Proximal Policy Optimization): Good baseline, stable, sample-efficient
- **SAC** (Soft Actor-Critic): Excellent for continuous control with exploration regularization
- **TD3** (Twin Delayed DDPG): Alternative actor-critic with reduced overestimation bias

### Key RL Challenges Here
- **Sparse rewards:** Cup reaching is deceptively hard without shaping
- **Exploration:** 8D action space is large; random actions rarely reach the target
- **Sample efficiency:** Real grasping requires many interactions to learn
- **Convergence:** Long-horizon tasks (reach + grasp) need careful reward tuning

---

## 🚀 Quick Start

### Setup
```bash
cd arm_rl
python -m pip install -r requirements.txt  # (if applicable)
```

### Run Training
```bash
# Default: headless training
python train.py --episodes 2000

# With rendering to visualize
python train.py --episodes 500 --render
```

### Inspect Environment
```python
from env.panda_env import PandaEnv
import numpy as np

env = PandaEnv(render=False)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Take a random step
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward:.3f}")
```

---

## 🎬 Training Progress Visualization

### Before Training (Initial Behavior)
Random exploration—the agent has not learned any meaningful strategy yet. The arm flails randomly without coordination, unable to reach the target cup.

https://github.com/BrianLi9897/deep_rl_robot/blob/main/videos/initial_training.webm

**What you see:**
- Jerky, uncoordinated arm movements
- No clear strategy for reaching the cup
- Random gripper open/close
- Typical reward: **~-5 to -10 per episode** (very far from target)

### After Training (~1000–2000 episodes)
The agent has learned a coherent reaching strategy. It smoothly moves the end-effector toward the cup, approaches carefully, and begins to grasp.

https://github.com/BrianLi9897/deep_rl_robot/blob/main/videos/after_training_sometime.webm

**What you see:**
- Smooth, purposeful arm trajectories
- Direct path planning toward the target
- Controlled approach and positioning
- Gripper closes near the target
- Typical reward: **~-0.5 to -1.5 per episode** (much closer to target)

**Learning Highlights:**
- ✅ Joint coordination emerges (no more random flailing)
- ✅ Distance to target decreases by ~5–10x
- ✅ Grasp success rate improves significantly
- ✅ Obstacle avoidance is learned implicitly (negative collisions in reward)

---

## 📊 Training Tips

1. **Reward Scaling**
   - The distance reward dominates early; consider scaling by exploration bonus for variance

2. **Hyperparameters to Tune**
   - Learning rate: Start at 3e-4 for PPO, 1e-4 for SAC
   - Batch size: 64–256 depending on available memory
   - Network size: 2–3 hidden layers of 256–512 units

3. **Curriculum Ideas**
   - Phase 1: Remove obstacles, just reach the cup
   - Phase 2: Add obstacles but keep cup at fixed positions
   - Phase 3: Randomize cup position each episode
   - Phase 4: Introduce grasping task (close gripper at target)

4. **Debugging**
   - Plot joint limits violations (actuator_ctrlrange clipping)
   - Watch for collisions with desk legs or walls
   - Monitor gripper never opens past 0.04m (physically constrained)

5. **Evaluation**
   - Success metric: distance to target < 0.05m (5cm)
   - Grasp metric: gripper closed within 5cm radius

---

## 📁 Project Structure
```
arm_rl/
├── env/
│   └── panda_env.py              # Gym environment wrapper
├── assets/
│   └── panda/
│       ├── scene.xml             # Office world with desk, obstacles
│       └── panda.xml             # Franka Panda robot MJCF model
├── videos/
│   ├── initial_training.webm      # Random exploration (pre-training)
│   └── after_training_sometime.webm  # Learned reaching strategy
├── train.py                       # RL training script
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

---

## 🔗 References

- **MuJoCo:** https://mujoco.readthedocs.io/ — Physics engine and XML format
- **Gymnasium:** https://gymnasium.farama.org/ — RL environment interface (successor to OpenAI Gym)
- **Franka Robot:** https://frankaemika.github.io/ — Official Panda arm docs
- **RL Algorithms:**
  - PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
  - SAC: Haarnoja et al., "Soft Actor-Critic Algorithms" (2018)

---

## 💡 Future Enhancements

- [ ] Multi-object reaching (multiple cups)
- [ ] Hierarchical RL (learn reach skill, then grasp skill)
- [ ] Domain randomization (mass, friction, object size randomization)
- [ ] Real-world validation with actual Franka Panda
- [ ] Imitation learning from human demonstrations

---

**Happy training! 🦾**
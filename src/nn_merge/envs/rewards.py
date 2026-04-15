"""Custom reward wrappers for MuJoCo environments.

Each wrapper overrides the environment's reward signal. The underlying MuJoCo
state is accessible via self.unwrapped.data (positions, velocities, forces).

To add a new reward:
  1. Subclass gymnasium.RewardWrapper
  2. Override step() to compute your reward from the MuJoCo state
  3. Add it to the REWARDS dict at the bottom of this file
"""

import gymnasium as gym
import numpy as np


class ForwardTarget(gym.RewardWrapper):
    def __init__(self, env, speed_target: float = 1.0, torque_penalty: float = 0.1):
        super().__init__(env)
        self.speed_target = speed_target
        self.torque_penalty = torque_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0.0)
        speed_reward = -abs(forward_velocity - self.speed_target)
        torque_cost = self.torque_penalty * np.sum(np.square(action))
        contact_reward = info.get("reward_contact", 0.0)
        
        reward = speed_reward - torque_cost + contact_reward + 1.0
        return obs, reward, terminated, truncated, info

class HealthyV5Clone(gym.RewardWrapper):
    def __init__(self, env, speed_target: float = 1.0, torque_penalty: float = 0.1,
                 max_healthy_z: float = 1.0):
        super().__init__(env)
        self.speed_target = speed_target
        self.torque_penalty = torque_penalty
        self.max_healthy_z = max_healthy_z

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0.0)
        speed_reward = -abs(forward_velocity - self.speed_target)
        torque_cost = self.torque_penalty * np.sum(np.square(action))
        contact_reward = info.get("reward_contact", 0.0)

        torso_z = float(self.unwrapped.data.qpos[2])
        healthy_reward = 1.0 if torso_z <= self.max_healthy_z else 0.0

        reward = speed_reward - torque_cost + contact_reward + healthy_reward
        return obs, reward, terminated, truncated, info

class ForwardReward(gym.RewardWrapper):
    """Reward only forward (positive x) velocity. No survival bonus, no penalties."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0.0)
        reward = forward_velocity
        return obs, reward, terminated, truncated, info


class SpinReward(gym.RewardWrapper):
    """Reward angular velocity around the z-axis (yaw). Encourages spinning."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # Ant's qvel[2] is the z-axis angular velocity
        angular_velocity = self.unwrapped.data.qvel[2]
        reward = abs(angular_velocity)
        return obs, reward, terminated, truncated, info


class EnergyEfficientReward(gym.RewardWrapper):
    """Reward moderate forward speed while heavily penalizing large torques."""

    def __init__(self, env, speed_target: float = 1.0, torque_penalty: float = 0.1):
        super().__init__(env)
        self.speed_target = speed_target
        self.torque_penalty = torque_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0.0)
        # Reward being close to target speed, penalize deviation
        speed_reward = -abs(forward_velocity - self.speed_target)
        # Penalize control effort
        torque_cost = self.torque_penalty * np.sum(np.square(action))
        # Match Ant-v5's healthy_reward so the agent isn't incentivized to terminate early
        reward = speed_reward - torque_cost + 1.0
        return obs, reward, terminated, truncated, info


REWARDS: dict[str, type[gym.RewardWrapper]] = {
    "forward": ForwardReward,
    "spin": SpinReward,
    "energy_efficient": EnergyEfficientReward,
    "forward_target": ForwardTarget,
    "healthy_v5_clone": HealthyV5Clone
}

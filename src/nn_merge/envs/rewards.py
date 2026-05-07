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


class ForwardTarget(gym.Wrapper):
    def __init__(self, env, speed_target: float = 2.5, torque_penalty: float = 0.1, healthy_reward: float = 2.0, death_penalty: float = 100.0):
        super().__init__(env)
        self.speed_target = speed_target
        # Scale torque penalty by target speed: slower agents need more precision/gentleness,
        # while faster agents need more power.
        self.torque_penalty = torque_penalty * (0.5 / max(speed_target, 0.1))
        self.healthy_reward = healthy_reward
        self.death_penalty = death_penalty
        
        # Add target speed to observation space so the policy can distinguish roles
        if isinstance(self.observation_space, gym.spaces.Box):
            low = np.append(self.observation_space.low, -np.inf)
            high = np.append(self.observation_space.high, np.inf)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _get_obs(self, obs):
        return np.append(obs, self.speed_target)

    def reset(self, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # 1. Ultimate Deterrence: Penalize early termination above all else
        if terminated and not truncated:
            return self._get_obs(obs), -self.death_penalty, terminated, truncated, info
            
        forward_velocity = info.get("x_velocity", 0.0)
        y_velocity = info.get("y_velocity", 0.0)
        torso_z = float(self.unwrapped.data.qpos[2])
        
        # Triangle/Linear speed reward: 1.0 at target, 0.0 at v=0.
        # This provides a constant gradient incentive to move, preventing the 
        # "lazy" local optimum caused by the flattened gradient of a quadratic.
        abs_error = abs(forward_velocity - self.speed_target)
        speed_reward = max(0.0, 1.0 - abs_error / max(self.speed_target, 0.1))
        
        # 3. Forward Pressure (scaled down to avoid chaotic charging)
        forward_reward = 1.0 * forward_velocity
        
        # 4. Stability: Penalize lateral movement
        lateral_penalty = 1.0 * abs(y_velocity) 
        
        # 5. Uprightness: Significant penalty for being on back
        is_upright = (0.28 < torso_z < 0.8)
        current_healthy_reward = self.healthy_reward if is_upright else -2.0
        
        # 6. Symmetry & Leg Usage: Force all 4 legs to contribute equally
        action_arr = np.array(action)
        leg_work = np.array([
            np.sum(np.abs(action_arr[0:2])), # Front-Left
            np.sum(np.abs(action_arr[2:4])), # Front-Right
            np.sum(np.abs(action_arr[4:6])), # Back-Left
            np.sum(np.abs(action_arr[6:8]))  # Back-Right
        ])
        
        # Variance of work across 4 legs (penalizes 'dead' or dragging legs)
        usage_variance = np.var(leg_work)
        
        # Side-to-side and Front-to-back balance for stability
        lr_diff = abs((leg_work[0] + leg_work[2]) - (leg_work[1] + leg_work[3]))
        fb_diff = abs((leg_work[0] + leg_work[1]) - (leg_work[2] + leg_work[3]))
        
        symmetry_penalty = 2.0 * usage_variance + 0.2 * (lr_diff + fb_diff)
        
        torque_cost = self.torque_penalty * np.sum(np.square(action))
        
        # Master Reward Equation
        reward = (10.0 * speed_reward) + forward_reward + current_healthy_reward - torque_cost - lateral_penalty - symmetry_penalty
        
        return self._get_obs(obs), reward, terminated, truncated, info

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

class HalfCheetahForwardTarget(gym.RewardWrapper):
    def __init__(self, env, speed_target: float = 2.0, control_penalty: float = 0.1,
                 upright_weight: float = 0.3):
        super().__init__(env)
        self.speed_target = speed_target
        self.control_penalty = control_penalty
        self.upright_weight = upright_weight

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0.0)
        speed_reward = max(0.0, 1.0 - abs(forward_velocity - self.speed_target) / max(self.speed_target, 0.1))
        control_cost = self.control_penalty * np.sum(np.square(action))
        pitch = float(self.unwrapped.data.qpos[2])
        upright_reward = self.upright_weight * np.cos(pitch) if abs(pitch) > 0.98 else 0
        reward = speed_reward - control_cost + upright_reward
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


class DynamicTarget(ForwardTarget):
    """Switches between two target speeds at a specific step count.
    
    Useful for evaluating if a single policy (merged or multi-task) can
    dynamically adapt its behavior mid-episode.
    """
    def __init__(self, env, target1: float = 0.5, target2: float = 1.5, switch_step: int = 500, **kwargs):
        super().__init__(env, speed_target=target1, **kwargs)
        self.target1 = target1
        self.target2 = target2
        self.switch_step = int(switch_step)
        self.current_step = 0

    def step(self, action):
        # Update speed_target before calling super().step() so the observation 
        # (which appends speed_target) reflects the new goal immediately.
        if self.current_step >= self.switch_step:
            self.speed_target = self.target2
        
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_step = 0
        self.speed_target = self.target1
        return super().reset(**kwargs)


REWARDS: dict[str, type[gym.Wrapper]] = {
    "forward": ForwardReward,
    "spin": SpinReward,
    "energy_efficient": EnergyEfficientReward,
    "forward_target": ForwardTarget,
    "dynamic_target": DynamicTarget,
    "half_cheetah_fwd_tgt": HalfCheetahForwardTarget
}

from nn_merge.envs.rewards import REWARDS

import gymnasium as gym


def make_env(env_id: str, reward_name: str = "default", **reward_kwargs) -> gym.Env:
    """Create an environment with an optional custom reward wrapper."""
    env = gym.make(env_id)
    if reward_name != "default":
        if reward_name not in REWARDS:
            available = ", ".join(REWARDS.keys())
            raise ValueError(f"Unknown reward {reward_name!r}. Available: {available}")
        env = REWARDS[reward_name](env, **reward_kwargs)
    return env

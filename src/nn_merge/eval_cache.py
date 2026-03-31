"""Persistent JSON cache for evaluation results.

Cache key format: "{resolved_model_path}|{reward_name}|seed{seed}"

The seed is the eval environment seed (controls observation noise / episode
initialization). Including it in the key means:
- The same model+reward evaluated with different seeds produces separate entries
- Re-running with the same seed hits the cache and is skipped
"""

import json
import os
from datetime import datetime
from pathlib import Path

DEFAULT_CACHE_PATH = "models/eval_cache.json"


def load_cache(path: str = DEFAULT_CACHE_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_cache(cache: dict, path: str = DEFAULT_CACHE_PATH) -> None:
    """Atomically write cache to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def _reward_suffix(reward_name: str, reward_kwargs: dict | None = None) -> str:
    if not reward_kwargs:
        return reward_name
    kw = ",".join(f"{k}={v}" for k, v in sorted(reward_kwargs.items()))
    return f"{reward_name}[{kw}]"


def make_model_key(model_path: str, reward_name: str, seed: int, reward_kwargs: dict | None = None) -> str:
    resolved = str(Path(model_path).resolve())
    suffix = _reward_suffix(reward_name, reward_kwargs)
    return f"{resolved}|{suffix}|seed{seed}"


def make_merged_key(source_paths: list[str], strategy: str, reward_name: str, seed: int, reward_kwargs: dict | None = None) -> str:
    resolved = sorted(str(Path(p).resolve()) for p in source_paths)
    joined = ",".join(resolved)
    suffix = _reward_suffix(reward_name, reward_kwargs)
    return f"merged:{strategy}:{joined}|{suffix}|seed{seed}"


def get_entry(cache: dict, key: str) -> list[float] | None:
    entry = cache.get(key)
    return entry["episode_rewards"] if entry else None


def set_entry(
    cache: dict,
    key: str,
    episode_rewards: list[float],
    model_path: str,
    reward_name: str,
    seed: int,
) -> None:
    cache[key] = {
        "model_path": model_path,
        "reward_name": reward_name,
        "seed": seed,
        "episode_rewards": list(episode_rewards),
        "n_episodes": len(episode_rewards),
        "timestamp": datetime.now().isoformat(),
    }

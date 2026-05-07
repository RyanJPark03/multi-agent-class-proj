import argparse
from dotenv import load_dotenv
import os
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from nn_merge.envs import make_env
from nn_merge.utils import load_model
from nn_merge.eval_cache import (
    DEFAULT_CACHE_PATH,
    get_entry,
    load_cache,
    make_model_key,
    save_cache,
    set_entry,
)


def main():
    load_dotenv()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, default="models/fast_and_slow_ants/fast_ant.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--reward", type=str, default="default",
                        help="Reward wrapper name (must match training)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="Record MP4 videos (works headless)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory to save recorded videos "
                             "(default: <model_dir>/<model_stem>_eval_videos)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-kwargs", nargs="*", default=[],
                        help="Reward kwargs (e.g., speed_target=1.5)")
    parser.add_argument("--env-kwargs", nargs="*", default=[],
                        help="Kwargs passed to gym.make (e.g. terminate_when_unhealthy=False)")
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Setting CUDA_VISIBLE_DEVICES to: {args.gpu}")
        
    if args.record and args.video_dir is None:
        model_path = Path(args.model)
        args.video_dir = str(model_path.parent / (model_path.stem + "_eval_videos"))

    def _parse_val(v):
        if isinstance(v, str):
            if v.lower() == "true": return True
            if v.lower() == "false": return False
            if v.lower() in ("none", "null"): return None
        try:
            if "." in v: return float(v)
            return int(v)
        except ValueError: return v

    reward_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.reward_kwargs}
    env_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.env_kwargs}

    if args.render:
        env_kwargs["render_mode"] = "human"
        env = make_env(args.env_id, args.reward, env_kwargs=env_kwargs, **reward_kwargs)
        env = Monitor(env)
        model = load_model(args.model, device="cpu")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=args.episodes, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    elif args.record:
        env_kwargs["render_mode"] = "rgb_array"
        env = make_env(args.env_id, args.reward, env_kwargs=env_kwargs, **reward_kwargs)
        env = Monitor(env)
        env = RecordVideo(env, video_folder=args.video_dir,
                          episode_trigger=lambda _: True)
        model = load_model(args.model, device="cpu")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=args.episodes, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        env.close()
        print(f"Videos saved to {args.video_dir}/")

    else:
        cache_key = make_model_key(args.model, args.reward, args.seed)
        episode_rewards = None
        episode_lengths = None

        if not args.no_cache:
            cache = load_cache(args.cache)
            episode_rewards, episode_lengths = get_entry(cache, cache_key)
            if episode_rewards is not None:
                print(f"Cache hit (seed={args.seed}, {len(episode_rewards)} episodes)")

        if episode_rewards is None:
            env = make_env(args.env_id, args.reward, env_kwargs=env_kwargs, **reward_kwargs)
            env = Monitor(env)
            env.reset(seed=args.seed)
            model = load_model(args.model, device="cpu")
            episode_rewards, episode_lengths = evaluate_policy(
                model, env, n_eval_episodes=args.episodes,
                deterministic=True, return_episode_rewards=True,
            )
            if not args.no_cache:
                set_entry(cache, cache_key, episode_rewards, args.model, args.reward, args.seed,
                          episode_lengths=episode_lengths)
                save_cache(cache, args.cache)

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        if episode_lengths:
            mean_len = sum(episode_lengths) / len(episode_lengths)
            std_len = (sum((l - mean_len) ** 2 for l in episode_lengths) / len(episode_lengths)) ** 0.5
            print(f"Mean length: {mean_len:.1f} +/- {std_len:.1f}")


if __name__ == "__main__":
    main()

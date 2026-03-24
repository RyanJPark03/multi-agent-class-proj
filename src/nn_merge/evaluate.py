import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from nn_merge.envs import make_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--reward", type=str, default="default",
                        help="Reward wrapper name (must match training)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="Record MP4 videos (works headless)")
    parser.add_argument("--video-dir", type=str, default="models/videos",
                        help="Directory to save recorded videos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.render:
        env = gym.make(args.env_id, render_mode="human")
    elif args.record:
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=args.video_dir,
                          episode_trigger=lambda _: True)
    else:
        env = make_env(args.env_id, args.reward)

    model = PPO.load(args.model)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if args.record:
        env.close()
        print(f"Videos saved to {args.video_dir}/")


if __name__ == "__main__":
    main()

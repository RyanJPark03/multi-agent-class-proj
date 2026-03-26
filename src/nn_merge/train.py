import argparse
import os
import subprocess
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from nn_merge.envs import make_env


class WandbMetricsCallback(BaseCallback):
    """Logs all SB3 training metrics to W&B each rollout."""

    def _on_rollout_end(self) -> bool:
        import wandb
        metrics = {k: v for k, v in self.logger.name_to_value.items()}
        if metrics:
            wandb.log(metrics, step=self.num_timesteps)
        return True

    def _on_step(self) -> bool:
        return True


def auto_select_gpus():
    """Pick GPUs with the most free memory. 2 if >4 available, else 1."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            idx, free_mb = line.split(", ")
            gpus.append((int(idx), int(free_mb)))
        gpus.sort(key=lambda g: g[1], reverse=True)
        n_take = 2 if len(gpus) > 4 else 1
        selected = [str(g[0]) for g in gpus[:n_take]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
        print(f"Auto-selected GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on a MuJoCo task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--reward", type=str, default="default",
                        help="Reward wrapper name (see nn_merge/envs/rewards.py)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Comma-separated CUDA GPUs to use (e.g., '0' or '0,1')")
    parser.add_argument("--wandb-project", type=str, default="nn-merge")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--reward-kwargs", nargs="*", default=[],
                        help="Reward wrapper kwargs as key=value pairs (e.g. speed_target=3.0)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for training: cpu, cuda, or auto")
    args = parser.parse_args()

    # Parse reward kwargs
    reward_kwargs = {}
    for kv in args.reward_kwargs:
        key, val = kv.split("=", 1)
        try:
            val = float(val)
        except ValueError:
            pass
        reward_kwargs[key] = val

    if args.device == "cpu":
        pass  # no GPU needed
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        auto_select_gpus()

    save_path = args.save_path or f"models/{args.env_id.lower()}_seed{args.seed}"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # W&B setup
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        run_name = args.run_name or f"{args.env_id}_{args.reward}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "env_id": args.env_id,
                "reward": args.reward,
                "reward_kwargs": reward_kwargs,
                "seed": args.seed,
                "timesteps": args.timesteps,
                "hidden_size": args.hidden_size,
                "device": args.device,
            },
        )

    env = make_env(args.env_id, args.reward, **reward_kwargs)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[args.hidden_size, args.hidden_size],
            vf=[args.hidden_size, args.hidden_size],
        )
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    callback = WandbMetricsCallback() if use_wandb else None
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    # Save parameter summary
    from nn_merge.inspect_model import save_param_summary
    param_path = f"{save_path}_params.txt"
    save_param_summary(save_path, param_path)
    print(f"Parameter summary saved to {param_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

import argparse
from dotenv import load_dotenv
import os
import subprocess
from pathlib import Path

from stable_baselines3 import PPO, SAC
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


class CheckpointCallback(BaseCallback):
    """Save model checkpoints at regular intervals during training."""

    def __init__(self, save_freq: int, checkpoint_dir: str, save_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_wandb = save_wandb

    def _init_callback(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.checkpoint_dir, f"step_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"Checkpoint saved: {path}.zip")
            if self.save_wandb:
                import wandb
                artifact = wandb.Artifact(
                    f"checkpoint-step-{self.num_timesteps}",
                    type="model",
                    metadata={"timesteps": self.num_timesteps},
                )
                artifact.add_file(f"{path}.zip")
                wandb.log_artifact(artifact)
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
    load_dotenv()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

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
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for training: cpu, cuda, or auto")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Save a checkpoint every N timesteps (default: timesteps/10)")
    parser.add_argument("--save-wandb-checkpoints", action="store_true",
                        help="Upload checkpoints as W&B artifacts")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                        help="RL algorithm to use: ppo or sac")
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

    if args.algo == "ppo":
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
    elif args.algo == "sac":
        policy_kwargs = dict(
            net_arch=dict(
                pi=[args.hidden_size, args.hidden_size],
                qf=[args.hidden_size, args.hidden_size],
            )
        )
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    callbacks = []
    if use_wandb:
        callbacks.append(WandbMetricsCallback())
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else args.timesteps // 10
    if checkpoint_freq > 0:
        checkpoint_dir = os.path.join(str(Path(save_path).parent), "checkpoints", Path(save_path).name)
        callbacks.append(CheckpointCallback(
            save_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir,
            save_wandb=args.save_wandb_checkpoints and use_wandb,
            verbose=1,
        ))
    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)
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

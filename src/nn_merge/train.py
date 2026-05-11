import argparse
from dotenv import load_dotenv
import os
import subprocess
import torch.multiprocessing as mp
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from nn_merge.envs import make_env
from nn_merge.cadmm.dinno import DiNNOCallback
from nn_merge.cadmm.config import DiNNOConfig

# ... (WandbMetricsCallback, CheckpointCallback, auto_select_gpus remain the same) ...
class WandbMetricsCallback(BaseCallback):
    def __init__(self, prefix: str = "", verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix

    def _on_rollout_end(self) -> bool:
        import wandb
        metrics = {f"{self.prefix}{k}": v for k, v in self.logger.name_to_value.items()}
        if metrics:
            wandb.log(metrics)
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

def main():
    load_dotenv()
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    parser = argparse.ArgumentParser(description="Train a PPO/SAC agent on a MuJoCo task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--reward", type=str, default="forward_target")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="nn-merge")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--reward-kwargs", nargs="*", default=[])
    parser.add_argument("--env-kwargs", nargs="*", default=[],
                        help="Kwargs passed to gym.make (e.g. terminate_when_unhealthy=False)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--dinno", action="store_true")
    parser.add_argument("--dinno-targets", nargs="+", type=float, default=[0.5, 1.5],
                        help="Target speeds for each DiNNO agent (one per agent).")
    parser.add_argument("--rho-schedule-steps", type=int, default=None,
                        help="Steps over which rho ramps from rho_init to rho_final. "
                             "Defaults to timesteps // 10 so the schedule shape stays "
                             "consistent across training budgets.")

    # CHANGED: We now take a SINGLE base model to prevent the permutation mismatch problem
    parser.add_argument("--load-base-model", type=str, default=None,
                        help="Path to a single pretrained baseline model to initialize BOTH agents.")
    args = parser.parse_args()

    if args.rho_schedule_steps is None:
        args.rho_schedule_steps = max(1, args.timesteps // 10)
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Setting CUDA_VISIBLE_DEVICES to: {args.gpu}")

    reward_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.reward_kwargs}
    env_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.env_kwargs}

    save_path = args.save_path or f"models/{args.env_id.lower()}_seed{args.seed}"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb

    if not args.dinno:
        # Standard Single Agent Training
        # Ensure a speed target is set (defaulting to 0.5 for the slow agent baseline)
        if "speed_target" not in reward_kwargs:
            reward_kwargs["speed_target"] = 0.5
        
        env = make_env(args.env_id, args.reward, env_kwargs=env_kwargs, **reward_kwargs)

        if use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name or f"{args.env_id}_{args.reward}_single_seed{args.seed}",
                config=vars(args)
            )

        callbacks = _get_callbacks(args, use_wandb, save_path)
        
        if args.load_base_model:
            model = _load_model(args, args.load_base_model, env, args.device)
        else:
            model = _create_model(args, env, device=args.device)
            
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(save_path)
        
        if use_wandb:
            wandb.finish()
    else:
        config = DiNNOConfig()
        manager = mp.Manager()
        shared_registry = manager.dict()

        # Both agents MUST start with the same initialization to avoid permutation collapse.
        # If args.load_base_model is provided via CLI, it will be used for both.
        # Otherwise, they will initialize with the same random weights due to shared seeding.
        agent_configs = [
            {"agent_idx": i, "target_speed": float(t), "load_model": args.load_base_model}
            for i, t in enumerate(args.dinno_targets)
        ]

        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        n_gpus = len(gpu_ids)

        processes = []
        for i, cfg in enumerate(agent_configs):
            is_cuda = "cuda" in args.device or args.device == "auto"
            dev = f"cuda:{i % n_gpus}" if is_cuda and n_gpus > 0 else "cpu"
            
            p = mp.Process(
                target=train_worker,
                args=(cfg["agent_idx"], args, dev, cfg["target_speed"], shared_registry, save_path, use_wandb, cfg["load_model"])
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        print("Parallel DiNNO training complete.")

def train_worker(agent_idx, args, device, target_speed, shared_registry, save_path, use_wandb, load_model_path=None):
    from stable_baselines3.common.utils import set_random_seed
    # Ensure both workers start with the same RNG state.
    # This guarantees identical randomized policy initialization.
    set_random_seed(args.seed)

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    
    agent_name = f"agent_{agent_idx}"
    prefix = f"agent{agent_idx}/"
    
    reward_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.reward_kwargs}
    reward_kwargs["speed_target"] = target_speed
    env_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1]) for kv in args.env_kwargs}

    env = make_env(args.env_id, args.reward, env_kwargs=env_kwargs, **reward_kwargs)
    config = DiNNOConfig()
    config.rho_schedule_steps = args.rho_schedule_steps

    # CRITICAL FIX: target_params MUST be "actor". Never apply consensus to the Critic
    # when the agents have different reward functions.
    cb_dinno = DiNNOCallback(
        node_id=agent_name,
        target_params="actor",
        rho_init=config.rho_init,
        rho_final=config.rho_final,
        rho_schedule_steps=config.rho_schedule_steps,
        registry=shared_registry,
        communication_freq=config.communication_freq
    )
    
    agent_save_path = f"{save_path}_agent{agent_idx}"
    
    if use_wandb:
        import wandb
        base_name = args.run_name or f"{args.env_id}_{args.reward}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=f"{base_name}_agent{agent_idx}",
            group=args.run_name or f"{args.env_id}_dinno_parallel",
            config={
                "env_id": args.env_id,
                "reward": args.reward,
                "reward_kwargs": reward_kwargs,
                "seed": args.seed,
                "timesteps": args.timesteps,
                "hidden_size": args.hidden_size,
                "device": device,
                "dinno": True,
                "agent_idx": agent_idx,
            },
            reinit=True
        )

    callbacks = [cb_dinno] + _get_callbacks(args, use_wandb, agent_save_path, prefix=prefix)
    
    # Priority: Load the EXACT SAME model from disk if provided, 
    # otherwise initialize with shared randomized weights.
    if load_model_path:
        model = _load_model(args, load_model_path, env, device)
    else:
        model = _create_model(args, env, device=device)
        
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(agent_save_path)
    
    if use_wandb:
        wandb.finish()

def _get_callbacks(args, use_wandb, save_path, prefix=""):
    callbacks = []
    if use_wandb:
        callbacks.append(WandbMetricsCallback(prefix=prefix))
    # Checkpoints every 10% of training
    checkpoint_freq = args.timesteps // 10
    if checkpoint_freq > 0:
        checkpoint_dir = os.path.join(str(Path(save_path).parent), "checkpoints", Path(save_path).name)
        callbacks.append(CheckpointCallback(
            save_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir,
            save_wandb=use_wandb,
            verbose=1,
        ))
    return callbacks

def _parse_val(v):
    if isinstance(v, str):
        if v.lower() == "true": return True
        if v.lower() == "false": return False
        if v.lower() in ("none", "null"): return None
    try: return float(v)
    except (ValueError, TypeError): return v

def _create_model(args, env, device):
    if args.algo == "ppo":
        policy_kwargs = dict(net_arch=dict(pi=[args.hidden_size, args.hidden_size], vf=[args.hidden_size, args.hidden_size]))
        return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=args.seed, device=device, verbose=1)
    elif args.algo == "sac":
        policy_kwargs = dict(net_arch=dict(pi=[args.hidden_size, args.hidden_size], qf=[args.hidden_size, args.hidden_size]))
        return SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=args.seed, device=device, verbose=1)

def _load_model(args, path, env, device):
    kwargs = dict(env=env, device=device)
    if args.algo == "ppo": return PPO.load(path, **kwargs)
    elif args.algo == "sac": return SAC.load(path, **kwargs)

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
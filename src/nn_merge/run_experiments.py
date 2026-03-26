import argparse
import subprocess
import sys
import threading
from pathlib import Path

import yaml


def get_available_gpus() -> list[int]:
    """Query nvidia-smi for GPU indices, sorted by free memory descending."""
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
        return [g[0] for g in gpus]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def build_train_command(experiment: dict, defaults: dict) -> list[str]:
    """Build a `python -m nn_merge.train` command from experiment config."""
    cfg = {**defaults, **experiment}
    cmd = [sys.executable, "-m", "nn_merge.train"]

    name = cfg.get("name", f"{cfg.get('reward', 'default')}_seed{cfg.get('seed', 0)}")
    save_path = cfg.get("save_path", f"models/{name}")

    cmd.extend(["--save-path", save_path])
    cmd.extend(["--seed", str(cfg.get("seed", 0))])
    cmd.extend(["--timesteps", str(cfg.get("timesteps", 1_000_000))])
    cmd.extend(["--env-id", cfg.get("env_id", "Ant-v5")])
    cmd.extend(["--hidden-size", str(cfg.get("hidden_size", 64))])

    reward = cfg.get("reward", "default")
    cmd.extend(["--reward", reward])

    reward_kwargs = cfg.get("reward_kwargs", {})
    if reward_kwargs:
        kv_pairs = [f"{k}={v}" for k, v in reward_kwargs.items()]
        cmd.extend(["--reward-kwargs"] + kv_pairs)

    if cfg.get("wandb_project"):
        cmd.extend(["--wandb-project", cfg["wandb_project"]])
    if cfg.get("run_name"):
        cmd.extend(["--run-name", cfg["run_name"]])
    elif name:
        cmd.extend(["--run-name", name])
    if cfg.get("no_wandb", False):
        cmd.append("--no-wandb")

    cmd.extend(["--device", cfg.get("device", "cpu")])

    return cmd


def stream_output(proc: subprocess.Popen, label: str):
    """Stream process stdout with a prefix label."""
    for line in iter(proc.stdout.readline, ""):
        print(f"[{label}] {line}", end="", flush=True)


def run_experiment(cmd: list[str], label: str, gpu: str | None, semaphore: threading.Semaphore):
    """Run a single experiment subprocess."""
    env = None
    if gpu is not None:
        import os
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}

    with semaphore:
        print(f"[{label}] Starting (GPU={gpu or 'auto'}): {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        stream_output(proc, label)
        proc.wait()
        status = "DONE" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{label}] {status}")


def main():
    parser = argparse.ArgumentParser(description="Run multiple training experiments in parallel")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent experiments (default: number of experiments)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    max_parallel = args.max_parallel or len(experiments)
    semaphore = threading.Semaphore(max_parallel)

    # Assign GPUs round-robin
    gpus = get_available_gpus()

    threads = []
    for i, exp in enumerate(experiments):
        label = exp.get("name", f"exp_{i}")
        cmd = build_train_command(exp, defaults)
        gpu = str(gpus[i % len(gpus)]) if gpus else None

        t = threading.Thread(target=run_experiment, args=(cmd, label, gpu, semaphore))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()

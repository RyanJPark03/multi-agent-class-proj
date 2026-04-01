import argparse
import os
import signal
import subprocess
import sys
import threading
import time

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


def build_train_command(experiment: dict, defaults: dict, output_dir: str) -> list[str]:
    """Build a `python -m nn_merge.train` command from experiment config."""
    cfg = {**defaults, **experiment}
    cmd = [sys.executable, "-m", "nn_merge.train"]

    name = cfg.get("name", f"{cfg.get('reward', 'default')}_seed{cfg.get('seed', 0)}")
    save_path = cfg.get("save_path", f"{output_dir}/{name}")

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

    checkpoint_freq = cfg.get("checkpoint_freq", 0)
    if checkpoint_freq:
        cmd.extend(["--checkpoint-freq", str(checkpoint_freq)])
    if cfg.get("save_wandb_checkpoints", False):
        cmd.append("--save-wandb-checkpoints")

    return cmd


# Track all running processes for cleanup on Ctrl+C
_active_procs: list[subprocess.Popen] = []
_lock = threading.Lock()


def run_experiment(cmd: list[str], label: str, gpu: str | None, semaphore: threading.Semaphore,
                   threads_per_exp: int = 1):
    """Run a single experiment subprocess."""
    env = {**os.environ,
           "OMP_NUM_THREADS": str(threads_per_exp),
           "MKL_NUM_THREADS": str(threads_per_exp)}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    with semaphore:
        print(f"[{label}] Starting (GPU={gpu or 'auto'}, {threads_per_exp} CPU threads)")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        with _lock:
            _active_procs.append(proc)

        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                # Only print summary lines, skip verbose SB3 table formatting
                if line.startswith("|") or line.startswith("-") or not line:
                    continue
                print(f"[{label}] {line}", flush=True)
            proc.wait()
        finally:
            with _lock:
                if proc in _active_procs:
                    _active_procs.remove(proc)

        status = "DONE" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{label}] {status}")


def kill_all():
    """Kill all active child processes."""
    with _lock:
        for proc in _active_procs:
            try:
                proc.terminate()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run multiple training experiments in parallel")
    parser.add_argument("--config", type=str, nargs="+", required=True,
                        help="Path(s) to YAML experiment config(s)")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent experiments (default: total number of experiments)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full SB3 training output")
    args = parser.parse_args()

    from pathlib import Path

    # Collect experiments from all config files
    all_jobs: list[tuple[str, list[str]]] = []  # (label, cmd)
    all_devices_list: list[str] = []
    for config_path in args.config:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        defaults = config.get("defaults", {})
        experiments = config["experiments"]

        config_stem = Path(config_path).stem
        output_dir = f"models/{config_stem}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for exp in experiments:
            label = exp.get("name", f"{config_stem}_{len(all_jobs)}")
            cmd = build_train_command(exp, defaults, output_dir)
            all_jobs.append((label, cmd))
            all_devices_list.append({**defaults, **exp}.get("device", "cpu"))

    max_parallel = args.max_parallel or len(all_jobs)
    semaphore = threading.Semaphore(max_parallel)

    # Limit CPU: half the cores, split evenly across concurrent experiments
    total_cpus = os.cpu_count() or 1
    threads_per_exp = max(1, total_cpus // 2 // max_parallel)

    # Assign GPUs round-robin (only if any experiment uses cuda)
    gpus = get_available_gpus() if any(d != "cpu" for d in all_devices_list) else []

    print(f"Running {len(all_jobs)} experiments from {len(args.config)} config(s) "
          f"(max parallel: {max_parallel}, {threads_per_exp} CPU threads/exp, "
          f"{total_cpus} cores total)")

    threads = []
    for i, (label, cmd) in enumerate(all_jobs):
        device = all_devices_list[i]
        gpu = str(gpus[i % len(gpus)]) if gpus and device != "cpu" else None

        t = threading.Thread(target=run_experiment,
                             args=(cmd, label, gpu, semaphore, threads_per_exp),
                             daemon=True)
        t.start()
        threads.append(t)
        # Stagger launches so parallel wandb.init() calls don't race
        if i < len(all_jobs) - 1:
            time.sleep(2)

    try:
        for t in threads:
            t.join()
        print("\nAll experiments complete.")
    except KeyboardInterrupt:
        print("\nInterrupted — stopping all experiments...")
        kill_all()
        sys.exit(1)


if __name__ == "__main__":
    main()

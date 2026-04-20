"""Violin plot comparing model performance across rewards and seeds.

For each (model, reward, eval_seed) triple, evaluates the policy and caches the
per-episode rewards. Groups models by stripping `_seed\\d+$` from their filename
stem so models trained with different seeds form one violin. Merged models are
shown as separate violins at the right of each subplot.

Reward specs support inline kwargs using colon-separated key=value pairs:
  forward_target:speed_target=2.0
  forward_target:speed_target=2.0,torque_penalty=0.05
"""

import argparse
import re
from pathlib import Path

import json
import zipfile

import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from nn_merge.eval_cache import (
    DEFAULT_CACHE_PATH,
    get_entry,
    load_cache,
    make_merged_key,
    make_model_key,
    save_cache,
    set_entry,
)
from nn_merge.envs import make_env
from nn_merge.merging.strategies import MERGE_STRATEGIES


def _load_model(path: str, device: str = "cpu") -> BaseAlgorithm:
    """Load an SB3 model, auto-detecting the algorithm from the saved zip."""
    import stable_baselines3 as sb3

    zip_path = path if path.endswith(".zip") else path + ".zip"
    with zipfile.ZipFile(zip_path) as zf:
        data = json.loads(zf.read("data"))

    algo_map = {}
    for attr in dir(sb3):
        cls = getattr(sb3, attr)
        if isinstance(cls, type) and issubclass(cls, BaseAlgorithm) and cls is not BaseAlgorithm:
            algo_map[attr.upper()] = cls

    policy_class = data.get("policy_class", {})
    if isinstance(policy_class, dict):
        policy_class = policy_class.get("__module__", "")
    for name in algo_map:
        if name in policy_class.upper():
            return algo_map[name].load(path, device=device)

    return sb3.PPO.load(path, device=device)


def parse_reward_spec(spec: str) -> tuple[str, dict]:
    """Parse 'name' or 'name:k=v,k=v' into (name, kwargs)."""
    if ":" not in spec:
        return spec, {}
    name, kw_str = spec.split(":", 1)
    kwargs = {}
    for pair in kw_str.split(","):
        k, v = pair.split("=", 1)
        try:
            kwargs[k] = float(v)
        except ValueError:
            kwargs[k] = v
    return name, kwargs


def _group_name(model_path: str) -> str:
    stem = Path(model_path).stem
    return re.sub(r"_seed\d+$", "", stem)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'path' or 'name=path' into (label, path)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    return _group_name(spec), spec


def _run_eval(
    model: BaseAlgorithm,
    env_id: str,
    reward_name: str,
    reward_kwargs: dict,
    seed: int,
    n_episodes: int,
    cache: dict,
    cache_key: str,
    no_cache: bool,
) -> list[float]:
    if not no_cache:
        cached_rewards, _ = get_entry(cache, cache_key)
        if cached_rewards is not None:
            print(f"  Cache hit: {cache_key}")
            return cached_rewards

    env = make_env(env_id, reward_name, **reward_kwargs)
    env = Monitor(env)
    env.reset(seed=seed)
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=n_episodes,
        deterministic=True, return_episode_rewards=True,
    )
    env.close()

    if not no_cache:
        set_entry(cache, cache_key, rewards, cache_key, reward_name, seed,
                  episode_lengths=lengths)

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate models and plot violin comparisons")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model paths. Optional 'name=path' syntax to label "
                             "(e.g. fast=models/exp/step_500000.zip)")
    parser.add_argument("--rewards", nargs="+", default=["default"],
                        help="Reward specs: 'name' or 'name:key=val,key=val'")
    parser.add_argument("--strategies", nargs="+", default=["weight_average"])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=str, default="models/eval_plot.png")
    parser.add_argument("--outlier-sidecar", type=str, default=None,
                        help="Path to outlier sidecar JSON (see nn_merge.outliers). "
                             "Filters episode rewards by per-(group,reward) min/max bounds.")
    parser.add_argument("--pairwise", action="store_true",
                        help="Merge every pair of models separately instead of all at once")
    parser.add_argument("--merged-save-dir", type=str, default=None,
                        help="Directory to save merged models (e.g. models/my_exp). "
                             "Each strategy saved as <dir>/merged_<strategy>.zip")
    args = parser.parse_args()

    # Parse reward specs into (name, kwargs, label) tuples
    reward_specs = []
    for spec in args.rewards:
        name, kwargs = parse_reward_spec(spec)
        label = spec  # use full spec as column label
        reward_specs.append((name, kwargs, label))

    cache = load_cache(args.cache) if not args.no_cache else {}

    # results[reward_label][group_label] = list of episode rewards (all seeds pooled)
    results: dict[str, dict[str, list[float]]] = {r[2]: {} for r in reward_specs}

    # --- Evaluate individual models ---
    model_specs = [parse_model_spec(s.strip()) for s in args.models if s.strip()]
    loaded_models: dict[str, BaseAlgorithm] = {}
    for label, model_path in model_specs:
        if not Path(model_path).exists() and not Path(model_path + ".zip").exists():
            raise FileNotFoundError(f"Model not found: {model_path!r}")
        print(f"Evaluating {label} ({model_path})...")
        model = _load_model(model_path, device="cpu")
        loaded_models[label] = model

        for reward_name, reward_kwargs, reward_label in reward_specs:
            for seed in args.eval_seeds:
                key = make_model_key(model_path, reward_name, seed, reward_kwargs)
                rewards = _run_eval(
                    model, args.env_id, reward_name, reward_kwargs, seed,
                    args.episodes, cache, key, args.no_cache,
                )
                results[reward_label].setdefault(label, []).extend(rewards)

    # --- Merge and evaluate ---
    import copy
    from itertools import combinations

    labels = list(loaded_models.keys())
    if args.pairwise:
        merge_groups = list(combinations(range(len(labels)), 2))
    else:
        merge_groups = [tuple(range(len(labels)))]

    for strategy_name in args.strategies:
        if strategy_name not in MERGE_STRATEGIES:
            print(f"Unknown strategy {strategy_name!r}, skipping.")
            continue

        for idx_group in merge_groups:
            group_labels = [labels[i] for i in idx_group]
            group_paths = [model_specs[i][1] for i in idx_group]
            state_dicts = [loaded_models[l].policy.state_dict() for l in group_labels]

            group_label = f"merged({'+'.join(group_labels)})"
            print(f"Merging {group_label} with {strategy_name}...")
            merged_sd = MERGE_STRATEGIES[strategy_name](state_dicts)
            shell = copy.deepcopy(loaded_models[group_labels[0]])
            shell.policy.load_state_dict(merged_sd)

            if args.merged_save_dir:
                save_dir = Path(args.merged_save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"merged_{'_'.join(group_labels)}_{strategy_name}"
                shell.save(str(save_path))
                print(f"  Merged model saved to {save_path}.zip")

            for reward_name, reward_kwargs, reward_label in reward_specs:
                for seed in args.eval_seeds:
                    key = make_merged_key(group_paths, strategy_name, reward_name, seed, reward_kwargs)
                    rewards = _run_eval(
                        shell, args.env_id, reward_name, reward_kwargs, seed,
                        args.episodes, cache, key, args.no_cache,
                    )
                    results[reward_label].setdefault(group_label, []).extend(rewards)

    if not args.no_cache:
        save_cache(cache, args.cache)

    # --- Apply outlier sidecar ---
    if args.outlier_sidecar:
        from nn_merge.outliers import apply_sidecar, load_sidecar
        sidecar = load_sidecar(args.outlier_sidecar)
        for reward_label, group_dict in results.items():
            for group_label, rewards in group_dict.items():
                bounds = sidecar.get(f"{group_label}|{reward_label}")
                if bounds:
                    before = len(rewards)
                    group_dict[group_label] = apply_sidecar(rewards, bounds)
                    after = len(group_dict[group_label])
                    print(f"  outlier filter [{group_label}|{reward_label}]: {before} → {after}")

    # --- Plot ---
    n_cols = len(reward_specs)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6), squeeze=False)

    for ax, (_, _, reward_label) in zip(axes[0], reward_specs):
        groups = results[reward_label]
        # Individual model groups first, then merged_* last
        labels = sorted(
            groups.keys(),
            key=lambda k: (k.startswith("merged_"), k),
        )
        data = [groups[label] for label in labels]
        positions = list(range(len(labels)))

        if any(len(d) > 1 for d in data):
            parts = ax.violinplot(data, positions=positions, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)

        for pos, d in zip(positions, data):
            ax.scatter([pos] * len(d), d, color="black", alpha=0.35, s=12, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"reward: {reward_label}")
        ax.set_ylabel("Episode reward")

    fig.suptitle(f"Model comparison ({args.env_id})", y=1.01)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()

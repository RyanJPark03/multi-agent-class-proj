"""Per-component breakdown of speed-target rewards (Ant or HalfCheetah).

Rolls each model out for N episodes at each speed target, logs every term
of the underlying reward into info["components"], and plots one subplot
per component plus a combined total.

Supported envs: Ant-v5 (ForwardTarget, 7 components) and HalfCheetah-v5
(HalfCheetahForwardTarget, 3 components). The script auto-selects which
wrapper and component set to use based on --env-id.
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from nn_merge.envs.rewards import ForwardTarget, HalfCheetahForwardTarget
from nn_merge.utils import load_model


class AntBreakdownTarget(ForwardTarget):
    """Ant ForwardTarget with per-component logging into info['components']."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        comps = {k: 0.0 for k in ANT_KEYS}

        if terminated and not truncated:
            comps["death"] = -self.death_penalty
            info["components"] = comps
            return self._get_obs(obs), -self.death_penalty, terminated, truncated, info

        vx = info.get("x_velocity", 0.0)
        vy = info.get("y_velocity", 0.0)
        z = float(self.unwrapped.data.qpos[2])

        abs_err = abs(vx - self.speed_target)
        speed_r = max(0.0, 1.0 - abs_err / max(self.speed_target, 0.1))

        is_up = (0.28 < z < 0.8)
        healthy_r = self.healthy_reward if is_up else -2.0

        a = np.asarray(action)
        leg_work = np.array([
            np.sum(np.abs(a[0:2])),
            np.sum(np.abs(a[2:4])),
            np.sum(np.abs(a[4:6])),
            np.sum(np.abs(a[6:8])),
        ])
        usage_var = float(np.var(leg_work))
        lr_diff = abs((leg_work[0] + leg_work[2]) - (leg_work[1] + leg_work[3]))
        fb_diff = abs((leg_work[0] + leg_work[1]) - (leg_work[2] + leg_work[3]))
        sym_pen = 2.0 * usage_var + 0.2 * (lr_diff + fb_diff)
        torque_c = self.torque_penalty * float(np.sum(np.square(a)))

        comps["speed"] = 10.0 * speed_r
        comps["forward"] = float(vx)
        comps["healthy"] = healthy_r
        comps["torque"] = -torque_c
        comps["lateral"] = -1.0 * abs(vy)
        comps["symmetry"] = -sym_pen

        reward = sum(comps.values())
        info["components"] = comps
        return self._get_obs(obs), reward, terminated, truncated, info


class HalfCheetahBreakdownTarget(HalfCheetahForwardTarget):
    """HalfCheetahForwardTarget with per-component logging."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        comps = {k: 0.0 for k in HC_KEYS}

        vx = info.get("x_velocity", 0.0)
        a = np.asarray(action)
        pitch = float(self.unwrapped.data.qpos[2])

        speed_r = max(0.0, 1.0 - abs(vx - self.speed_target)
                      / max(self.speed_target, 0.1))
        control_c = self.control_penalty * float(np.sum(np.square(a)))
        upright_r = (self.upright_weight * float(np.cos(pitch))
                     if abs(pitch) > 0.98 else 0.0)

        comps["speed"] = speed_r
        comps["control"] = -control_c
        comps["upright"] = upright_r

        reward = sum(comps.values())
        info["components"] = comps
        return self._get_obs(obs), reward, terminated, truncated, info


ANT_KEYS = ("speed", "forward", "healthy", "torque", "lateral", "symmetry", "death")
HC_KEYS = ("speed", "control", "upright")

ENV_PROFILES = {
    "Ant-v5": {"wrapper": AntBreakdownTarget, "keys": ANT_KEYS},
    "HalfCheetah-v5": {"wrapper": HalfCheetahBreakdownTarget, "keys": HC_KEYS},
}


def rollout(model, env, n_episodes, base_seed, keys):
    """Return list of per-episode dicts (component sums + meta)."""
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        totals = {k: 0.0 for k in keys}
        steps = 0
        vx_sum = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            for k in keys:
                totals[k] += info["components"][k]
            vx_sum += info.get("x_velocity", 0.0)
            steps += 1
            if terminated or truncated:
                break
        totals["steps"] = steps
        totals["mean_vx"] = vx_sum / max(steps, 1)
        episodes.append(totals)
    return episodes


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    return Path(spec).stem, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path or path (defaults name to file stem)")
    parser.add_argument("--targets", nargs="+", type=float,
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-id", type=str, default="Ant-v5",
                        choices=sorted(ENV_PROFILES.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default="models/reward_breakdown.png")
    parser.add_argument("--reward-kwargs", nargs="*", default=[],
                        help="kwargs forwarded to the breakdown wrapper, e.g. "
                             "upright_weight=0.6")
    args = parser.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"

    profile = ENV_PROFILES[args.env_id]
    Wrapper = profile["wrapper"]
    keys = profile["keys"]

    def _parse_val(v):
        if isinstance(v, str):
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        try:
            if "." in v: return float(v)
            return int(v)
        except (ValueError, TypeError):
            return v
    extra_kwargs = {kv.split("=", 1)[0]: _parse_val(kv.split("=", 1)[1])
                    for kv in args.reward_kwargs}

    specs = [parse_model_spec(s) for s in args.models]
    plot_keys = list(keys) + ["total"]
    results = {name: {k: {"mean": [], "std": []} for k in plot_keys}
               for name, _ in specs}
    meta = {name: {"steps_mean": [], "vx_mean": []} for name, _ in specs}

    for name, path in specs:
        print(f"Loading {name} ({path})")
        model = load_model(path, device="cpu")
        for target in args.targets:
            env = gym.make(args.env_id)
            env = Wrapper(env, speed_target=float(target), **extra_kwargs)
            episodes = rollout(model, env, args.episodes,
                               base_seed=args.seed, keys=keys)
            env.close()
            totals_per_ep = [sum(ep[k] for k in keys) for ep in episodes]
            for k in keys:
                vals = [ep[k] for ep in episodes]
                results[name][k]["mean"].append(float(np.mean(vals)))
                results[name][k]["std"].append(float(np.std(vals)))
            results[name]["total"]["mean"].append(float(np.mean(totals_per_ep)))
            results[name]["total"]["std"].append(float(np.std(totals_per_ep)))

            steps_mean = float(np.mean([ep["steps"] for ep in episodes]))
            vx_mean = float(np.mean([ep["mean_vx"] for ep in episodes]))
            meta[name]["steps_mean"].append(steps_mean)
            meta[name]["vx_mean"].append(vx_mean)
            print(f"  target={target:.2f}  total={np.mean(totals_per_ep):8.1f}  "
                  f"ep_len={steps_mean:5.0f}  mean_vx={vx_mean:5.2f}")

    n = len(plot_keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, key in zip(axes.flat, plot_keys):
        for i, (name, _) in enumerate(specs):
            means = np.array(results[name][key]["mean"])
            stds = np.array(results[name][key]["std"])
            ax.plot(args.targets, means, marker="o", color=cmap(i),
                    label=name, linewidth=2)
            ax.fill_between(args.targets, means - stds, means + stds,
                            color=cmap(i), alpha=0.15)
        ax.set_title(key)
        ax.set_xlabel("speed_target")
        ax.set_ylabel("episode sum")
        ax.set_xticks(args.targets)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="best", frameon=False)
    fig.suptitle(f"Reward-component breakdown ({args.env_id}, "
                 f"{args.episodes} episodes)", y=1.0)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\nEpisode length / mean vx (per target):")
    header = "  target  " + "  ".join(f"{n:>20s}" for n, _ in specs)
    print(header)
    for j, t in enumerate(args.targets):
        cells = []
        for name, _ in specs:
            cells.append(f"len={meta[name]['steps_mean'][j]:4.0f} "
                         f"vx={meta[name]['vx_mean'][j]:+5.2f}")
        print(f"  {t:5.2f}   " + "  ".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()

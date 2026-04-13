"""CLI outlier sidecar editor for the eval cache.

Reads pre-computed episode rewards from the eval cache, prints per-group
statistics, and lets you set min/max thresholds via CLI flags. Thresholds
are written to a sidecar JSON file that `nn_merge.plot` reads to filter
outliers before plotting.

Sidecar format (human-editable):

    {
      "fast|forward_target:speed_target=1.25": {"min": 100.0, "max": 600.0},
      "merged_weight_average|forward_target:speed_target=2.0": {"max": 800.0}
    }

Either bound is optional; missing bounds mean "no clipping on that side".
The key is "<group_label>|<reward_label>" — the same labels used on the
plot's x-axis and column titles.

Typical workflow:

    # 1. Inspect: print stats for all (group, reward) combos
    python -m nn_merge.outliers stats \\
        --models fast=models/.../fast.zip slow=models/.../slow.zip \\
        --rewards "forward_target:speed_target=1.25"

    # 2. Set bounds for a specific group
    python -m nn_merge.outliers set "fast|forward_target:speed_target=1.25" \\
        --min 100 --max 600

    # 3. Show the current sidecar
    python -m nn_merge.outliers show

    # 4. Plot with the sidecar applied
    python -m nn_merge.plot ... --outlier-sidecar models/eval_outliers.json
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev

from nn_merge.eval_cache import (
    DEFAULT_CACHE_PATH,
    get_entry,
    load_cache,
    make_merged_key,
    make_model_key,
)

DEFAULT_SIDECAR_PATH = "models/eval_outliers.json"


def load_sidecar(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_sidecar(sidecar: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in sidecar.items() if v}
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def apply_sidecar(rewards: list[float], bounds: dict | None) -> list[float]:
    if not bounds:
        return list(rewards)
    lo = bounds.get("min", float("-inf"))
    hi = bounds.get("max", float("inf"))
    return [r for r in rewards if lo <= r <= hi]


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return float("nan")
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _collect_groups(
    cache: dict,
    model_specs: list[tuple[str, str]],
    reward_specs: list[tuple[str, dict, str]],
    eval_seeds: list[int],
    strategies: list[str],
) -> dict[tuple[str, str], list[float]]:
    """Pool cached episode rewards by (group_label, reward_label)."""
    groups: dict[tuple[str, str], list[float]] = {}

    for label, model_path in model_specs:
        for reward_name, reward_kwargs, reward_label in reward_specs:
            for seed in eval_seeds:
                key = make_model_key(model_path, reward_name, seed, reward_kwargs)
                rewards, _ = get_entry(cache, key)
                if rewards is None:
                    print(f"  cache miss: {label} / {reward_label} / seed={seed}")
                    continue
                groups.setdefault((label, reward_label), []).extend(rewards)

    paths = [p for _, p in model_specs]
    for strategy in strategies:
        merged_label = f"merged_{strategy}"
        for reward_name, reward_kwargs, reward_label in reward_specs:
            for seed in eval_seeds:
                key = make_merged_key(paths, strategy, reward_name, seed, reward_kwargs)
                rewards, _ = get_entry(cache, key)
                if rewards is None:
                    continue
                groups.setdefault((merged_label, reward_label), []).extend(rewards)

    return groups


def _print_stats(groups: dict[tuple[str, str], list[float]], sidecar: dict) -> None:
    if not groups:
        print("No data found in cache for the requested models/rewards.")
        return

    header = f"{'group|reward':<60} {'n':>5} {'min':>8} {'p5':>8} {'p50':>8} {'p95':>8} {'max':>8} {'mean':>8} {'std':>8}  bounds"
    print(header)
    print("-" * len(header))
    for (group, reward) in sorted(groups.keys()):
        data = groups[(group, reward)]
        s = sorted(data)
        key = f"{group}|{reward}"
        bounds = sidecar.get(key, {})
        bstr = ""
        if bounds:
            parts = []
            if "min" in bounds:
                parts.append(f"min={bounds['min']:g}")
            if "max" in bounds:
                parts.append(f"max={bounds['max']:g}")
            bstr = "  [" + ", ".join(parts) + "]"
        kept = apply_sidecar(data, bounds)
        dropped = len(data) - len(kept)
        if dropped:
            bstr += f" (drops {dropped})"
        print(
            f"{key:<60} {len(s):>5} {s[0]:>8.1f} {_percentile(s, 0.05):>8.1f} "
            f"{_percentile(s, 0.50):>8.1f} {_percentile(s, 0.95):>8.1f} {s[-1]:>8.1f} "
            f"{mean(s):>8.1f} {pstdev(s):>8.1f}{bstr}"
        )


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", required=True,
                   help="Same syntax as nn_merge.plot ('path' or 'name=path')")
    p.add_argument("--rewards", nargs="+", default=["default"])
    p.add_argument("--strategies", nargs="+", default=["weight_average"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)


def main():
    parser = argparse.ArgumentParser(description="CLI outlier sidecar editor")
    parser.add_argument("--sidecar", type=str, default=DEFAULT_SIDECAR_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="Print per-group stats from the eval cache")
    _add_data_args(p_stats)

    p_set = sub.add_parser("set", help="Set min/max bounds for one group")
    p_set.add_argument("key", help="'<group_label>|<reward_label>'")
    p_set.add_argument("--min", type=float, default=None)
    p_set.add_argument("--max", type=float, default=None)

    p_clear = sub.add_parser("clear", help="Remove bounds for one group")
    p_clear.add_argument("key", help="'<group_label>|<reward_label>'")

    sub.add_parser("show", help="Print the current sidecar")

    args = parser.parse_args()
    sidecar = load_sidecar(args.sidecar)

    if args.cmd == "stats":
        # Lazy-import plot helpers so this stays usable even if matplotlib isn't installed.
        from nn_merge.plot import parse_model_spec, parse_reward_spec
        cache = load_cache(args.cache)
        model_specs = [parse_model_spec(s.strip()) for s in args.models if s.strip()]
        reward_specs = []
        for spec in args.rewards:
            name, kwargs = parse_reward_spec(spec)
            reward_specs.append((name, kwargs, spec))
        groups = _collect_groups(cache, model_specs, reward_specs, args.eval_seeds, args.strategies)
        _print_stats(groups, sidecar)
        return

    if args.cmd == "set":
        if args.min is None and args.max is None:
            parser.error("set requires at least one of --min / --max")
        bounds = dict(sidecar.get(args.key, {}))
        if args.min is not None:
            bounds["min"] = args.min
        if args.max is not None:
            bounds["max"] = args.max
        sidecar[args.key] = bounds
        save_sidecar(sidecar, args.sidecar)
        print(f"{args.key} -> {bounds}")
        print(f"Saved sidecar to {args.sidecar}")
        return

    if args.cmd == "clear":
        if sidecar.pop(args.key, None) is None:
            print(f"No entry for {args.key!r}")
        else:
            save_sidecar(sidecar, args.sidecar)
            print(f"Cleared {args.key}")
        return

    if args.cmd == "show":
        if not sidecar:
            print(f"(empty sidecar at {args.sidecar})")
        else:
            print(json.dumps(sidecar, indent=2))
        return


if __name__ == "__main__":
    main()

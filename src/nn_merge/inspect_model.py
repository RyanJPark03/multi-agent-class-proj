import argparse

import torch
from stable_baselines3 import PPO


def format_param_summary(state_dict: dict[str, torch.Tensor], layer_filter: str | None = None) -> str:
    lines = []
    header = f"{'Layer':<50s} {'Shape':<20s} {'Params':>8s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
    lines.append(header)
    lines.append("-" * len(header))

    total_params = 0
    for name, param in state_dict.items():
        if layer_filter and layer_filter not in name:
            continue
        n = param.numel()
        total_params += n
        shape_str = str(list(param.shape))
        lines.append(
            f"{name:<50s} {shape_str:<20s} {n:>8d} "
            f"{param.float().mean().item():>10.6f} {param.float().std().item():>10.6f} "
            f"{param.float().min().item():>10.6f} {param.float().max().item():>10.6f}"
        )

    lines.append("-" * len(header))
    lines.append(f"Total parameters: {total_params:,}")
    return "\n".join(lines)


def save_param_summary(model_path: str, save_path: str, layer_filter: str | None = None):
    model = PPO.load(model_path)
    summary = format_param_summary(model.policy.state_dict(), layer_filter)
    with open(save_path, "w") as f:
        f.write(summary + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect model parameters")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=str, default=None,
                        help="Filter to layers containing this string")
    parser.add_argument("--values", action="store_true",
                        help="Print full parameter values (can be very long)")
    args = parser.parse_args()

    model = PPO.load(args.model)
    sd = model.policy.state_dict()

    print(format_param_summary(sd, args.layer))

    if args.values:
        print()
        for name, param in sd.items():
            if args.layer and args.layer not in name:
                continue
            print(f"--- {name} ---")
            print(param)
            print()


if __name__ == "__main__":
    main()

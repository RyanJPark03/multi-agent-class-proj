import argparse

import nn_merge.merging.strategies as strategies
from nn_merge.utils import load_model


def main():
    parser = argparse.ArgumentParser(description="Merge multiple trained models")
    parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to saved SB3 models"
    )
    parser.add_argument("--strategy", type=str, default="weight_average")
    parser.add_argument("--save-path", type=str, default="models/merged")
    args = parser.parse_args()

    strategy_fn = getattr(strategies, args.strategy)

    models = [load_model(path) for path in args.models]
    state_dicts = [m.policy.state_dict() for m in models]

    merged_sd = strategy_fn(state_dicts)

    # Use the first model as a shell and load merged weights
    shell = models[0]
    shell.policy.load_state_dict(merged_sd)
    shell.save(args.save_path)
    print(f"Merged {len(models)} models -> {args.save_path}.zip")


if __name__ == "__main__":
    main()

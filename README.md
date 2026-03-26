# Neural Network Merging Experiments

Experiment codebase for training small RL policies on MuJoCo tasks and merging their weights.

## Setup

Requires Python 3.10+ and MuJoCo.

```bash
uv venv && uv pip install -e .
wandb login  # one-time setup for logging
```

## Usage

After installation, you can run commands via `python -m`, the installed CLI entry points, or the shell scripts in `scripts/`.

### Train

Train PPO agents with different seeds or reward functions to produce models for merging:

```bash
# Same task, different seeds
python -m nn_merge.train --seed 0 --timesteps 500000
python -m nn_merge.train --seed 1 --timesteps 500000

# Different reward functions
python -m nn_merge.train --seed 0 --reward forward --timesteps 500000
python -m nn_merge.train --seed 0 --reward spin --timesteps 500000
python -m nn_merge.train --seed 0 --reward energy_efficient --timesteps 500000
```

Training metrics are logged to Weights & Biases. Use `--no-wandb` to disable.

| Argument | Default | Description |
|---|---|---|
| `--seed` | `0` | Random seed |
| `--timesteps` | `1000000` | Total training timesteps |
| `--save-path` | `models/{env}_seed{seed}` | Output path (without `.zip`) |
| `--env-id` | `Ant-v5` | Gymnasium environment |
| `--hidden-size` | `64` | Hidden layer size (2 layers) |
| `--reward` | `default` | Reward wrapper name (see below) |
| `--gpu` | auto | Comma-separated CUDA GPUs (e.g. `0` or `0,1`) |
| `--wandb-project` | `nn-merge` | W&B project name |
| `--run-name` | auto | W&B run name |
| `--no-wandb` | off | Disable W&B logging |
| `--reward-kwargs` | none | Reward wrapper params (e.g. `speed_target=3.0`) |

A `_params.txt` file with model parameter summary is saved alongside each model after training.

### Inspect

Inspect a trained model's parameters:

```bash
python -m nn_merge.inspect_model --model models/ant-v5_seed0
python -m nn_merge.inspect_model --model models/ant-v5_seed0 --layer policy_net
python -m nn_merge.inspect_model --model models/ant-v5_seed0 --values  # full tensor values
```

### Experiment Runner

Run multiple experiments in parallel from a YAML config:

```bash
python -m nn_merge.run_experiments --config experiments/example.yaml
python -m nn_merge.run_experiments --config experiments/example.yaml --max-parallel 2
```

Each experiment runs as a separate process with its own GPU assigned automatically. See `experiments/example.yaml` for the config format:

```yaml
defaults:
  env_id: Ant-v5
  timesteps: 2000000
  hidden_size: 64

experiments:
  - name: fast_ant
    reward: forward_target
    reward_kwargs:
      speed_target: 3.0
    seed: 0

  - name: spinner
    reward: spin
    seed: 0
```

The `defaults` section provides base values that individual experiments can override.

### Merge

Merge multiple trained models into one:

```bash
python -m nn_merge.merge --models models/ant-v5_seed0 models/ant-v5_seed1 models/ant-v5_seed2
```

| Argument | Default | Description |
|---|---|---|
| `--models` | (required) | Paths to saved models |
| `--strategy` | `weight_average` | Merge function name from `nn_merge.merging.strategies` |
| `--save-path` | `models/merged` | Output path |

### Evaluate

```bash
python -m nn_merge.evaluate --model models/ant-v5_seed0
python -m nn_merge.evaluate --model models/merged --reward forward

# Record MP4 videos (works headless / inside Docker)
python -m nn_merge.evaluate --model models/ant-v5_seed0 --record
```

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Path to saved model |
| `--episodes` | `10` | Number of eval episodes |
| `--env-id` | `Ant-v5` | Gymnasium environment |
| `--reward` | `default` | Reward wrapper (must match training) |
| `--render` | off | Live GUI rendering (requires display) |
| `--record` | off | Save MP4 videos to `--video-dir` |
| `--video-dir` | `models/videos` | Directory for recorded videos |
| `--seed` | `0` | Eval seed |

## Custom Rewards

Reward wrappers live in `src/nn_merge/envs/rewards.py`. Built-in options:

| Name | Description |
|---|---|
| `default` | Ant-v5's built-in reward (forward velocity + survival - control cost) |
| `forward` | Pure forward velocity, no penalties |
| `forward_target` | Reward proximity to target speed, penalize torque. kwargs: `speed_target`, `torque_penalty` |
| `spin` | Angular velocity around z-axis |
| `energy_efficient` | Target moderate speed, penalize large torques. kwargs: `speed_target`, `torque_penalty` |

Reward wrappers that accept kwargs can be configured from the CLI:

```bash
python -m nn_merge.train --reward forward_target --reward-kwargs speed_target=3.0 torque_penalty=0.05
```

To add a new reward, subclass `gymnasium.RewardWrapper` in `rewards.py` and add it to the `REWARDS` dict:

```python
class MyReward(gym.RewardWrapper):
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = ...  # your logic here (access MuJoCo state via self.unwrapped.data)
        return obs, reward, terminated, truncated, info

REWARDS["my_reward"] = MyReward
```

Then train with `--reward my_reward`.

## Adding Merge Strategies

Write a function in `src/nn_merge/merging/strategies.py` matching this signature:

```python
def my_strategy(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    ...
```

Then use it with `python -m nn_merge.merge --strategy my_strategy`.

## Docker

Build the image:

```bash
docker build -t nn-merge .
```

Run with GPU support, mounting `models/` so outputs persist on the host:

```bash
docker run --gpus all --env-file .env -v $(pwd)/models:/app/models -v $(pwd)/experiments:/app/experiments -v $(pwd)/src:/app/src -it nn-merge
```

This drops you into a bash shell inside the container. MuJoCo is configured for headless EGL rendering automatically. Run `wandb login` inside the container to enable logging.

## Project Structure

```
├── src/nn_merge/
│   ├── train.py              # PPO training script
│   ├── merge.py              # Model merging script
│   ├── evaluate.py           # Evaluation script
│   ├── inspect_model.py      # Parameter inspection
│   ├── run_experiments.py    # Parallel experiment runner
│   ├── envs/
│   │   ├── __init__.py       # make_env() factory
│   │   └── rewards.py        # Custom reward wrappers
│   └── merging/
│       └── strategies.py     # Merge strategy implementations
├── experiments/              # YAML experiment configs
│   └── example.yaml
├── scripts/                  # Shell wrappers
├── Dockerfile
└── pyproject.toml
```

## Notes

- **Ant is hard.** 500K timesteps may not be enough for convergence. Try `HalfCheetah-v5` for faster iteration.
- **Headless rendering:** set `MUJOCO_GL=egl` if you don't have a display.
- **SB3 model format:** saved as `.zip` files. Pass paths without the extension — SB3 appends it automatically.
- **Value function weights** are included in the merge. They're unused during evaluation (`model.predict` only uses the policy network), but matter if you fine-tune a merged model further.

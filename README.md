# Neural Network Merging Experiments

Experiment codebase for training small RL policies on MuJoCo tasks and merging their weights.

## Setup

Requires Python 3.10+ and MuJoCo. This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv venv                  # create .venv/ from pyproject.toml
uv pip install -e .      # install the package in editable mode
source .venv/bin/activate
wandb login              # one-time setup for logging (or use --no-wandb)
```

You can also run any command without activating the venv by prefixing with `uv run`, e.g. `uv run python -m nn_merge.train ...`.

MuJoCo ships via the `mujoco` pip dependency — no separate system install needed. On headless machines (no display), set `MUJOCO_GL=egl` so rendering/recording works.

## GPU Usage

Training uses PyTorch + Stable-Baselines3 and will run on CUDA if available, CPU otherwise. GPU selection happens automatically:

- Explicitly specify with `--gpu 0` or `--gpu 3`. In DiNNO training, if multiple GPUs are visible (e.g., `--gpu 0,1`), the script will automatically assign each agent to its own device.
- Force CPU with `--device cpu`.
- [run_experiments.py](src/nn_merge/run_experiments.py) assigns GPUs round-robin across concurrent experiments.

Note: PPO on small MLP policies is often CPU-bound (env stepping dominates). A GPU helps most when running several experiments in parallel or using larger networks. The runner also caps total CPU threads to half the machine's cores and divides them evenly across concurrent experiments.

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
| `--gpu` | auto | Comma-separated CUDA GPUs (e.g. `0` or `3`) |
| `--wandb-project` | `nn-merge` | W&B project name |
| `--run-name` | auto | W&B run name |
| `--no-wandb` | off | Disable W&B logging |
| `--reward-kwargs` | none | Reward wrapper params (e.g. `speed_target=3.0 torque_penalty=0.05`) |
| `--env-kwargs` | none | Kwargs passed to `gym.make` (e.g. `terminate_when_unhealthy=False`) |
| `--checkpoint-freq` | `timesteps/10` | Save a checkpoint every N timesteps (0 to disable) |
| `--save-wandb-checkpoints` | off | Upload checkpoints as W&B artifacts |
| `--algo` | `ppo` | RL algorithm (`ppo` or `sac`) |
| `--dinno` | off | Enable parallel DiNNO consensus training |
| `--load-base-model` | none | Path to a single base model to initialize both DiNNO agents |

A `_params.txt` file with model parameter summary is saved alongside each model after training.

Checkpoints are saved by default (10 evenly spaced) under `checkpoints/<model_name>/` next to the final model. Use `--save-wandb-checkpoints` to additionally upload each checkpoint as a versioned W&B artifact.

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

# Multiple config files — all experiments pooled together
python -m nn_merge.run_experiments --config experiments/ants.yaml experiments/cheetahs.yaml
```

CPU threads are automatically capped to half the machine's cores, split evenly across concurrent experiments.

Each experiment runs as a separate process with its own GPU assigned automatically. See `experiments/example.yaml` for the config format:

```yaml
defaults:
  env_id: Ant-v5
  timesteps: 2000000
  hidden_size: 64
  checkpoint_freq: 200000  # optional, defaults to timesteps/10
  max_threads: 1           # optional, caps CPU threads/exp (else half-cores auto-split)
  env_kwargs:              # optional, forwarded to gym.make
    terminate_when_unhealthy: false

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

`env_kwargs` is forwarded verbatim to `gym.make(env_id, **env_kwargs)`. Note that
these kwargs are env-specific: e.g. `terminate_when_unhealthy` is an Ant flag and
will raise on HalfCheetah, which has no "unhealthy" state. Use it to keep
episodes running to the time limit (otherwise an Ant that falls over ends its
episode immediately, starving low-speed targets of training data).

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
| `--video-dir` | auto | Directory for videos (defaults to `<model>_eval_videos/`) |
| `--reward-kwargs` | none | Override reward params (e.g. `speed_target=1.5`) |
| `--env-kwargs` | none | Kwargs passed to `gym.make` (e.g. `terminate_when_unhealthy=False`) |
| `--gpu` | none | Explicitly set GPU for evaluation |
| `--seed` | `0` | Eval environment seed (observation noise) |
| `--cache` | `models/eval_cache.json` | Path to evaluation cache |
| `--no-cache` | off | Skip cache read/write |

Results are cached by `(model, reward, seed)` — re-running the same combination skips evaluation.

Example for running model with dynamic speed target:

```bash
python src/nn_merge/evaluate.py \
    --model path/to/merged_model.zip \
    --reward dynamic_target \
    --reward-kwargs target1=0.5 target2=1.5 switch_step=500 \
    --record

```

### Plot

Evaluate models across rewards and seeds, then produce a violin plot:

```bash
python -m nn_merge.plot \
  --models models/my_exp/ant_model1 models/my_exp/ant_model2 \
  --rewards forward spin default \
  --eval-seeds 0 1 2 3 4 \
  --output models/eval_plot.png
```

Rewards that take kwargs can be specified inline using `name:key=value` syntax:

```bash
python -m nn_merge.plot \
  --models models/fast_and_slow_ants/fast_ant.zip models/fast_and_slow_ants/slow_ant.zip \
  --rewards "forward_target:speed_target=1.25" "forward_target:speed_target=2.0" \
  --output models/merge_comparison.png
```

Models with the same base name (e.g. `fast_ant_seed0`, `fast_ant_seed1` → `fast_ant`) are grouped into one violin. Each column is a different reward. A merged model (weight average by default) is shown as a separate violin on the right.

To label models explicitly (useful when comparing checkpoints whose filenames are identical, like `step_500000.zip` from different runs), use `name=path` syntax:

```bash
python -m nn_merge.plot \
  --models fast=models/fast_ant/checkpoints/step_500000.zip \
           slow=models/slow_ant/checkpoints/step_500000.zip \
  --rewards "forward_target:speed_target=1.25" "forward_target:speed_target=2.0" \
  --output models/merge_comparison.png
```

| Argument | Default | Description |
|---|---|---|
| `--models` | (required) | Paths to saved models |
| `--rewards` | `default` | Reward specs: `name` or `name:key=val,key=val` |
| `--strategies` | `weight_average` | Merge strategies to include |
| `--eval-seeds` | `0 1 2 3 4` | Eval seeds (determine violin distribution) |
| `--episodes` | `20` | Episodes per (model, reward, seed) |
| `--env-id` | `Ant-v5` | Gymnasium environment |
| `--cache` | `models/eval_cache.json` | Shared eval cache |
| `--output` | `models/eval_plot.png` | Figure save path |
| `--merged-save-dir` | none | Save merged model(s) as `<dir>/merged_<strategy>.zip` |
| `--outlier-sidecar` | none | JSON file of per-group min/max bounds (see Outliers) |
| `--no-cache` | off | Skip cache |

### Outliers

Set per-group min/max bounds to filter outliers from the plot. The tool reads
cached episode rewards (no re-evaluation) and edits a sidecar JSON file via
CLI subcommands:

```bash
# 1. Inspect — prints n / min / p5 / p50 / p95 / max / mean / std per group
python -m nn_merge.outliers stats \
  --models fast=models/fast_and_slow_ants/fast_ant.zip \
           slow=models/fast_and_slow_ants/slow_ant.zip \
  --rewards "forward_target:speed_target=1.25" "forward_target:speed_target=2.0"

# 2. Set bounds for one (group, reward) combo
python -m nn_merge.outliers set "fast|forward_target:speed_target=1.25" \
  --min 100 --max 600

# 3. Show / clear entries
python -m nn_merge.outliers show
python -m nn_merge.outliers clear "fast|forward_target:speed_target=1.25"
```

`stats` re-prints existing bounds and shows how many episodes each filter would
drop, so you can iterate. The sidecar is a plain JSON file (default
`models/eval_outliers.json`) and can also be hand-edited:

```json
{
  "fast|forward_target:speed_target=1.25": {"min": 100.0, "max": 600.0},
  "merged_weight_average|forward_target:speed_target=2.0": {"max": 800.0}
}
```

Keys are `"<group_label>|<reward_label>"` — the same labels shown on the plot.
Either bound is optional. Pass `--outlier-sidecar models/eval_outliers.json` to
`nn_merge.plot` to apply the filter when rendering.

## Custom Rewards

Reward wrappers live in `src/nn_merge/envs/rewards.py`. Built-in options:

| Name | Description |
|---|---|
| `default` | Ant-v5's built-in reward (forward velocity + survival - control cost) |
| `forward` | Pure forward velocity, no penalties |
| `forward_target` | Multi-task reward. Appends `speed_target` to the observation. Penalty is calculated as relative error squared. kwargs: `speed_target`, `torque_penalty`, `healthy_reward` |
| `dynamic_target` | Mid-episode goal switching. Switches from `target1` to `target2` at `switch_step`. |
| `spin` | Angular velocity around z-axis |
| `energy_efficient` | Target moderate speed, penalize large torques. kwargs: `speed_target`, `torque_penalty` |
| `healthy_v5_clone` | Configurable speed target with standard Healthy check |

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

## Project Structure

```
├── src/nn_merge/
│   ├── train.py              # PPO training script
│   ├── merge.py              # Model merging script
│   ├── evaluate.py           # Evaluation script (cached)
│   ├── plot.py               # Violin plot across models/rewards
│   ├── outliers.py           # Interactive outlier sidecar editor
│   ├── eval_cache.py         # Eval result cache (JSON)
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
└── pyproject.toml
```

## Distributed Optimization (DiNNO / CADMM)

Support for distributed consensus training using the DiNNO (Distributed Neural Network Optimization) algorithm. This allows multiple RL agents to train on local data/rewards while being regularized to converge toward a shared global policy.

### Core Components (`src/nn_merge/cadmm/dinno.py`)

- **`DiNNOManager`**: tracks local parameter snapshots ($\theta_k$), dual variables ($p$), and calculates the DiNNO penalty gradients.
- **`DiNNOCallback`**: Stable Baselines 3 `BaseCallback` that integrates consensus logic into algorithms like PPO/SAC using **PyTorch Gradient Hooks**. By using hooks, we append the consensus term $\nabla \mathcal{L}_{cons} = p + \rho \sum (\theta - \bar{\theta})$ to the RL gradients automatically during the backward pass.
- **Target Parameters**: In multi-task DiNNO (where agents have different speed targets), it is critical to set `target_params="actor"` so consensus is only enforced on the policy/action network and not the value function/critic, which must diverge to capture different reward scales.
- **Observation Space**: The `forward_target` reward wrapper automatically appends the `speed_target` to the observation space. This allows a single merged policy to behave differently based on the input goal.

### Example
The following snippet demonstrates how to set up two SAC agents to synchronize their policies through a shared registry:

```python
import gymnasium as gym
from stable_baselines3 import SAC
from nn_merge.cadmm.dinno import DiNNOCallback

# 1. Create a shared registry for parameter exchange
shared_registry = {}

# 2. Setup agents on their respective environments/devices
model0 = SAC("MlpPolicy", gym.make("Ant-v5"), device="cuda:0")
model1 = SAC("MlpPolicy", gym.make("Ant-v5"), device="cuda:1")

# 3. Initialize DiNNOCallbacks for each agent
callback0 = DiNNOCallback(
    node_id="agent_0", 
    rho=5.0,                  # Penalty strength
    registry=shared_registry, 
    communication_freq=100    # Sync snapshots every 100 steps
)

callback1 = DiNNOCallback(
    node_id="agent_1", 
    rho=5.0, 
    registry=shared_registry, 
    communication_freq=100
)

# 4. Starting concurrent or sequential training
# In a real setup, these might run in separate processes
model0.learn(total_timesteps=50000, callback=callback0)
model1.learn(total_timesteps=50000, callback=callback1)
```

### Mathematical Logic
The DiNNO objective modifies the standard RL loss by adding an augmented Lagrangian penalty:
$$\mathcal{L}_{total} = \mathcal{L}_{RL} + \theta^T p + \frac{\rho}{2} \sum_{j \in N_i} || \theta - \frac{\theta_i + \theta_j}{2} ||^2$$

The gradient hooks inject the derivative of this penalty directly into the optimizer. Communication happens at the end of every rollout (PPO) or every $N$ steps (SAC), where agents snapshot their current weights to the registry and update their dual variables ($p$) based on the disagreement with their neighbors.

## CLI Example

```bash
python src/nn_merge/train.py --dinno --load-base-model models/fast_and_slow_ants_v4/slow_ant_0.5  --algo sac   --timesteps 1000000 --gpu 3
```

## Notes

- **Ant is hard.** 500K timesteps may not be enough for convergence. Try `HalfCheetah-v5` for faster iteration.
- **Headless rendering:** set `MUJOCO_GL=egl` if you don't have a display.
- **SB3 model format:** saved as `.zip` files. Pass paths without the extension — SB3 appends it automatically.
- **Value function weights** are included in the merge. They're unused during evaluation (`model.predict` only uses the policy network), but matter if you fine-tune a merged model further.

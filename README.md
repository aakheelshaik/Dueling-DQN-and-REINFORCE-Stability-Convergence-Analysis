# Dueling DQN and REINFORCE Stability Analysis

This project compares reinforcement learning algorithms on classic Gymnasium control benchmarks. The notebook implements Dueling Deep Q-Networks and REINFORCE variants from scratch, trains them across multiple random seeds, and plots return statistics to compare stability and convergence.

## Project Contents

- `dueling_dqn_reinforce_analysis.ipynb` - main notebook containing all implementations, experiment runners, plots, and analysis.
- `,gitignore` - ignore file currently listing `.venv`.

## Algorithms

The notebook compares four agents:

- Dueling DQN Type 1
- Dueling DQN Type 2
- REINFORCE
- REINFORCE with value baseline

The DQN agents use:

- Experience replay
- Epsilon-greedy exploration
- Target network updates with Polyak averaging
- PyTorch neural network models

The REINFORCE agents use:

- Policy-gradient learning
- Discounted returns
- Optional learned value baseline for variance reduction

## Environments

Experiments are run on:

- `CartPole-v1`
- `Acrobot-v1`

Each algorithm is evaluated across 5 random seeds by default.

## Requirements

The notebook installs missing Python packages automatically, but you can also install them manually:

```bash
pip install gymnasium numpy torch matplotlib tqdm
```

Recommended Python environment:

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab

## How to Run

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install gymnasium numpy torch matplotlib tqdm jupyter
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open and run:

```text
dueling_dqn_reinforce_analysis.ipynb
```

Run the notebook cells from top to bottom. The experiment cells train all agents and generate summary statistics and plots.

## Experiment Defaults

CartPole experiments use 200 episodes per seed. Acrobot experiments use 300 episodes per seed.

Default shared settings include:

- Discount factor: `gamma = 0.99`
- Hidden dimension: `128`
- DQN learning rate: `2e-4`
- REINFORCE learning rate: `1e-3`
- DQN replay buffer size: `20000`
- DQN batch size: `64`
- Seeds: `[0, 1, 2, 3, 4]`

## Outputs

The notebook produces:

- Per-agent training returns
- Mean and standard deviation of returns across seeds
- Final-performance summary statistics
- Plots comparing learning stability and convergence

## Notes

- Training can take several minutes because each algorithm is run across multiple seeds and environments.
- Results may vary slightly depending on package versions, hardware, and random seeds.
- The repository currently contains notebook-based code only; there is no separate Python package or command-line entry point.

# power-grid-rl-agents

> **Suggested repository name:** `power-grid-rl-agents`
>
> This name is concise, descriptive, and follows standard GitHub naming conventions (lowercase, hyphen-separated). It clearly communicates that the repository contains reinforcement learning agents for power grid management.

Reinforcement learning agents for power grid management using the [Grid2Op](https://grid2op.readthedocs.io/) environment. Agents are trained to maintain grid stability and N-1 reliability by optimising a combined `L2RPNReward` and `N1Reward`. Two core algorithms are implemented and iteratively improved across multiple experiment branches:

- **Deep Q-Network (DQN)** – via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **Successive Representation (SR) Agent** – a custom implementation

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Branch Overview](#branch-overview)
- [Installation](#installation)
  - [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

---

## Project Overview

The goal of this project is to develop, evaluate, and iteratively improve RL agents for the task of automated power grid operation. The environment is the `l2rpn_case14_sandbox` scenario from Grid2Op, wrapped as a Gymnasium-compatible environment.

Key challenges addressed:
- **N-1 reliability** – the grid must remain stable after the loss of any single component.
- **Continuous operation** – agents must take topological actions (bus switching, line reconnection) to prevent cascading failures.
- **Exploration strategies** – multiple exploration approaches (ε-greedy, Boltzmann, UCB) are compared.

---

## Repository Structure

```
power-grid-rl-agents/
├── baseline.ipynb              # Baseline DQN agent (Stable-Baselines3 MlpPolicy)
├── baseline.zip                # Saved weights for the baseline DQN agent
├── CombinedScaledRewards.png   # Reward plot comparing agent performance
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files excluded from version control
└── README.md                   # This file
```

Each experiment branch (see [Branch Overview](#branch-overview)) follows the same structure and adds its own notebook, saved model, and reward plot.

---

## Branch Overview

This project uses a **branch-per-experiment** workflow. Each branch contains a self-contained notebook and saved model for a specific improvement or ablation:

| Branch | Description |
|---|---|
| `main` | Stable, production-ready baseline DQN agent |
| `DQN-with-new-exploration-strategy` | Boltzmann (softmax) exploration for DQN |
| `DQN-with-UCB` | Upper Confidence Bound (UCB) exploration for DQN |
| `Reward-Shaping` | Custom reward shaping to guide DQN behaviour |
| `DQN-with-Seeds-for-reproducibility` | Fixed random seeds for reproducible experiments |
| `minimize-observations` | Reduced observation space (critical features only) |
| `Successive-Representation` | SR agent baseline and tuned variants |
| `Final-code` | Final selected models and consolidated results |

> **GitHub best practice tip:** Long-lived experiment branches should be merged into `main` (or a dedicated `develop` branch) via pull requests once validated, and then deleted. Use branch naming conventions such as `feat/`, `exp/`, or `fix/` prefixes for clarity (e.g. `exp/boltzmann-exploration`).

---

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip`

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Samuel-Mbah/Power-grid-assignment.git
    cd Power-grid-assignment
    ```
    > If the repository has been renamed to `power-grid-rl-agents`, replace the URL accordingly.

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate        # Linux / macOS
    # venv\Scripts\activate         # Windows
    ```

3. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Grid2Op scenario data** (handled automatically on first run, or follow the [Grid2Op documentation](https://grid2op.readthedocs.io/en/latest/)).

### Dependencies

| Package | Purpose |
|---|---|
| `grid2op` | Power grid simulation environment |
| `lightsim2grid` | Fast backend for Grid2Op |
| `gymnasium` | RL environment interface (OpenAI Gym successor) |
| `stable-baselines3` | DQN implementation |
| `torch` | Deep learning backend for Stable-Baselines3 |
| `numpy` | Numerical computation |
| `matplotlib` | Reward visualisation |

---

## Usage

Open `baseline.ipynb` in Jupyter and run all cells in order. The notebook is self-contained and covers environment setup, agent training, evaluation, and plotting.

To run the key steps in a standalone Python script, first copy the `Gym2OpEnv` class definition from the notebook into a file (e.g. `env_wrapper.py`), then:

```python
from stable_baselines3 import DQN
from env_wrapper import Gym2OpEnv   # copy Gym2OpEnv from baseline.ipynb

# Initialise environment
env = Gym2OpEnv()

# Train a new agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("baseline")

# Load a pre-trained agent and evaluate
model = DQN.load("baseline", env=env)
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

To experiment with a different strategy, check out the corresponding branch:

```bash
git checkout DQN-with-UCB
```

---

## Results

The plot below compares the scaled episode rewards across evaluation runs for the baseline agent:

![Combined Scaled Rewards](CombinedScaledRewards.png)

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature-name`.
3. Commit your changes with clear messages: `git commit -m "feat: add X"`.
4. Push to your fork and open a pull request against `main`.
5. Ensure your branch is up to date with `main` before opening a PR.

Please keep one experiment or feature per branch and include a reward plot with your results.



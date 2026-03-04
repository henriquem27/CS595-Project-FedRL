# Federated Reinforcement Learning with Differential Privacy

A research project investigating how **Federated Learning (FL)** and **Differential Privacy (DP)** can be combined with **Reinforcement Learning (RL)** to train cooperative agents across heterogeneous environments without sharing raw training data. The project uses the Lunar Lander control task as a testbed, where agents operating under different planetary gravities and wind conditions collaboratively learn a shared landing policy.

Developed as part of **CS 595 - Applied Federated Learning** at Illinois Institute of Technology. All experiments were conducted on **Chameleon Cloud** infrastructure.

---

## Motivation

Traditional RL assumes centralized training, but real-world scenarios often involve multiple agents operating in distinct environments (e.g., autonomous vehicles in different cities, robots on different terrain). Sharing raw experience data between these agents raises privacy concerns and may be infeasible due to bandwidth constraints.

This project asks: **Can federated learning enable RL agents in heterogeneous environments to collaboratively learn a robust shared policy, and what is the privacy-utility tradeoff when differential privacy is applied to the weight updates?**

---

## Approach

### Reinforcement Learning

- **Algorithm:** Proximal Policy Optimization (PPO) via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **Environment:** [Gymnasium LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) with configurable gravity, wind power, and turbulence
- **Policy Network:** Multi-Layer Perceptron (MLPPolicy)
- **Vectorized Training:** 64 parallel sub-environments per client using `SubprocVecEnv` for throughput

### Federated Learning

The system implements **Federated Averaging (FedAvg)** across heterogeneous clients:

1. A central server broadcasts the current global model weights to a random subset of clients.
2. Each client trains locally on its own environment configuration for a fixed number of timesteps.
3. Clients send their updated weights back to the server.
4. The server computes the element-wise average of all received weights to produce the new global model.

**Client heterogeneity** is introduced through three planetary environments with varying gravity, each spawning additional derived clients with randomized wind power:

| Environment | Gravity (m/s^2) | Wind Power | Purpose |
|-------------|-----------------|------------|---------|
| Moon | -1.62 | 0 - 15 | Low gravity |
| Earth | -9.8 | 0 - 15 | Standard gravity |
| Mars | -3.73 | 0 - 15 | Medium gravity |

In the final configuration (**v5**), 21 total clients (3 base + 18 derived) participate, with approximately half selected per round.

### Differential Privacy

Privacy is enforced at the client update level using the **Laplace mechanism with L2 norm clipping**:

1. **Delta computation:** Compute the difference between updated and received weights.
2. **Norm clipping:** Clip the delta vector to a fixed L2 sensitivity bound.
3. **Noise injection:** Add Laplace noise scaled by `sensitivity / epsilon` to each weight.
4. **Aggregation:** The server averages the noisy deltas and applies them to the global model.

Four epsilon values are tested (500, 1000, 500, 100) to characterize the privacy-utility tradeoff, where lower epsilon provides stronger privacy guarantees but introduces more noise.

### Gradual Weight Adjustment (v4)

Version 4 introduced momentum-based weight blending to stabilize federated updates:

- **Standard FL:** `updated = (1 - alpha) * old + alpha * new` with `alpha = 0.3`
- **DP-FL:** Server-side learning rate scaling that dampens the impact of DP noise

This yielded 10-60% performance improvement for DP-FL configurations and smoother learning curves overall.

---

## Experimental Setup

### Infrastructure

All experiments were run on **Chameleon Cloud**, an NSF-funded configurable experimental environment for large-scale computer science research. Docker containers were used for reproducible execution.

### Training Configuration (v5 - Final)

| Parameter | Value |
|-----------|-------|
| Federated rounds | 125 |
| Local steps per client per round | 10,000 |
| Parallel sub-environments per client | 64 |
| Total clients | 21 (3 base + 18 derived) |
| Clients selected per round | ~11 |
| DP sensitivity | 0.2 |
| DP epsilon values tested | 100, 500, 1000, 5000 |

### Experiments Run

| Experiment | Description |
|------------|-------------|
| **Baseline FL** | Standard FedAvg, no privacy |
| **DP-FL (eps=5000)** | Minimal noise, near-baseline fidelity |
| **DP-FL (eps=1000)** | Light privacy |
| **DP-FL (eps=500)** | Moderate privacy |
| **DP-FL (eps=100)** | Strong privacy, significant noise |
| **Single Agent** | Independent training per environment (no federation) |

### Evaluation

Trained models are evaluated across 7 test scenarios (200 episodes each), including the three base environments with specific wind settings and a turbulence scenario unseen during training. Two key metrics are reported:

- **Mean reward per scenario:** Measures landing performance (200+ considered "solved").
- **KNN client identification accuracy on weight vectors:** Measures whether an adversary can determine which environment a client trained on by inspecting its model weights. Lower accuracy implies better privacy.

---

## Results

### Performance Comparison (Mean Reward)

| Scenario | Baseline FL | DP eps=5000 | DP eps=500 | DP eps=100 | Single Earth | Single Mars | Single Moon |
|----------|:-----------:|:-----------:|:----------:|:----------:|:------------:|:-----------:|:-----------:|
| Earth (no wind) | 254 | 274 | 256 | -379 | 241 | 191 | 33 |
| Earth (wind=6) | 200 | 234 | 217 | -344 | 229 | 143 | -10 |
| Mars (wind=5) | 248 | 249 | 241 | -541 | 155 | 188 | 141 |
| Mars (wind=8) | 251 | 247 | 244 | -586 | 151 | 174 | 130 |
| Moon (wind=15) | 72 | 104 | 112 | -935 | 11 | 34 | 33 |
| Moon (wind=7) | 134 | 146 | 129 | -1115 | 55 | 57 | 102 |
| Turbulence | 180 | 232 | 181 | -345 | 209 | 103 | -31 |
| **Average** | **191** | **213** | **197** | **-606** | **150** | **127** | **57** |

### Privacy Analysis (KNN Client Identification)

| Experiment | KNN Accuracy |
|------------|:------------:|
| Baseline FL (no DP) | 88% |
| DP eps=5000 | 82% |
| DP eps=500 | 78% |
| DP eps=100 | 36% |

Lower identification accuracy means an adversary is less able to distinguish which environment a client belongs to by examining its weight updates, indicating stronger privacy protection. At eps=100, accuracy drops to near-random (33% for 3 classes), demonstrating effective privacy but at the cost of utility.

### Key Findings

1. **Federated models generalize better than single-agent models.** The FL baseline outperforms every single-agent model on average, demonstrating that cross-environment collaboration produces more robust policies.
2. **Moderate DP preserves utility.** At eps=500 and eps=5000, the DP-FL models perform comparably to or better than the non-private baseline, while reducing client identifiability.
3. **Strong DP degrades performance severely.** At eps=100, the Laplace noise overwhelms the learning signal, producing a non-functional policy.
4. **The privacy-utility sweet spot lies around eps=500-1000.** These settings reduce KNN identification accuracy by 6-10 percentage points while maintaining competitive reward.

---

## Project Structure

```
CS595-Project-FedRL/
├── moonlanderv5/          # Final experiment code (results in figures/)
│   ├── experiment.py      # Orchestrates all experiment runs
│   ├── fl_moon.py         # Standard Federated Learning training loop
│   ├── dp_moon.py         # Differential Privacy FL training loop
│   ├── single_moon.py     # Single-agent baseline training
│   ├── helpers.py         # Utilities: logging, environment wrappers, aggregation
│   └── requirements.txt
├── moonlanderv4/          # Gradual weight adjustment + Docker support
│   ├── (same core files)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── run.sh
│   ├── plotting.py        # Automated plot generation
│   └── generate_plots.py
├── moonlanderv1-v3/       # Earlier iterations (v1: initial, v2: batched, v3: persistent envs)
├── plotting.ipynb         # Jupyter notebook for all final figures and evaluation
├── data_loader.py         # Loads saved weight vectors for analysis
├── combine_logs.py        # Merges per-client CSV metrics
├── figures/               # Publication-ready plots and result CSVs
│   ├── round_learning_curve.pdf
│   ├── silo_reward.pdf
│   ├── catplot.pdf
│   ├── fl_trajectory.pdf
│   ├── dp_*trajectory.pdf
│   ├── results.csv
│   └── knn_acc.csv
└── sv_results/            # Stored experiment outputs for reproducibility
```

### Evolution Across Versions

| Feature | v1 | v2 | v3 | v4 | v5 |
|---------|:--:|:--:|:--:|:--:|:--:|
| Basic FL | x | x | x | x | x |
| Differential Privacy | | x | x | x | x |
| Persistent environment pool | | | x | x | x |
| Disk-based metric streaming | | | x | x | x |
| Gradual weight adjustment | | | | x | |
| Docker containerization | | | | x | |
| Parallel sub-envs | 1 | 1 | 32 | 32 | 64 |
| Total clients | 15 | 15 | 15 | 15 | 21 |

---

## Tools and Technologies

| Category | Tool |
|----------|------|
| RL Framework | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) |
| Environment | [Gymnasium](https://gymnasium.farama.org/) LunarLander-v3 (Box2D physics) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Federated Aggregation | Custom FedAvg implementation |
| Differential Privacy | Laplace mechanism with L2 norm clipping (custom) |
| Compute Infrastructure | [Chameleon Cloud](https://www.chameleoncloud.org/) |
| Containerization | Docker |
| Data Analysis | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn, UMAP |
| Weight Analysis | UMAP dimensionality reduction, KNN classification |

---

## Quick Start

### Run Experiments

```bash
cd moonlanderv5
pip install -r requirements.txt
python experiment.py
```

This runs the full suite: DP-FL at four epsilon levels, standard FL, and single-agent baselines. Expected runtime is 2-8 hours depending on hardware.

### Run with Docker (v4)

```bash
cd moonlanderv4
chmod +x run.sh
./run.sh
```

### Reproduce Figures

All plots in `figures/` were generated from the Jupyter notebook:

```bash
pip install jupyter umap-learn
jupyter notebook plotting.ipynb
```

### Run Individual Experiments

```bash
cd moonlanderv5
python fl_moon.py       # Standard FL only
python dp_moon.py       # DP-FL only
python single_moon.py   # Single-agent baseline only
```

---

## Requirements

```
gymnasium[box2d]>=0.26.0
stable-baselines3>=2.0.0
torch>=1.13.0
numpy>=1.21.0,<2.0.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

For plotting: `umap-learn`, `seaborn`, `jupyter`

---

## Citation

```bibtex
@misc{fedrl-lunarlander-2025,
  title   = {Federated Reinforcement Learning with Differential Privacy
             on Heterogeneous Lunar Lander Environments},
  author  = {CS595 Project Team},
  year    = {2025},
  url     = {https://github.com/henriquem27/CS595-Project-FedRL}
}
```

---

## License

See repository for license information.

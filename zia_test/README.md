# Federated Reinforcement Learning with Differential Privacy - Complete Analysis

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Differential Privacy Implementation](#differential-privacy-implementation)
5. [Experiment Pipeline](#experiment-pipeline)
6. [Technical Details](#technical-details)
7. [Results and Visualization](#results-and-visualization)

---

## Project Overview

This is a **Federated Reinforcement Learning (FedRL)** research project that trains PPO (Proximal Policy Optimization) agents on the LunarLander-v3 environment with **differential privacy** mechanisms. The project compares three training approaches:

1. **Single-Agent Training** - Independent agents (baseline)
2. **Standard Federated Learning** - Collaborative training without privacy
3. **Differentially-Private Federated Learning** - Collaborative training with privacy guarantees

### Research Questions
- Does federated learning improve performance over independent training?
- What is the cost of differential privacy on model performance?
- Can agents learn across heterogeneous environments (different planetary gravities)?

---

## Project Structure

```
CS595-Project-FedRL/
├── zia_test/              # Simple RL examples (CartPole)
│   ├── main.py           # Standalone policy gradient implementation
│   └── list.py           # Gymnasium environment listing utility
├── moonlander/           # Main experimental codebase
│   ├── helpers.py        # Core FL utilities and callbacks
│   ├── fl_moon.py        # Standard federated learning
│   ├── dp_moon.py        # Differential privacy FL
│   ├── single_moon.py    # Independent training baseline
│   ├── clustered.py      # Clustered FL (multiple global models)
│   ├── experiment.py     # Main experiment orchestration
│   ├── plotting.py       # Individual experiment visualization
│   └── comparison_plot.py # Cross-experiment comparison plots
├── moonlanderv1/         # Legacy version 1
├── moonlanderv2/         # Legacy version 2
├── sv_results/           # Results storage
└── requirements.txt      # Python dependencies
```

---

## Core Components

### 1. Helper Functions (`moonlander/helpers.py`)

#### **Federated Averaging (FedAvg)**
```python
average_state_dicts() / average_ordered_dicts()
```
- **Purpose**: Core aggregation mechanism for federated learning
- **Algorithm**: 
  1. Takes model weights from multiple clients
  2. Sums them element-wise across all parameters
  3. Divides by number of clients to get average
- **Mathematical Formula**: `W_global = (1/N) * Σ(W_client_i)` for i=1 to N

#### **Clustering Functions**
```python
cluster_and_average_models(state_dicts, n_clusters)
find_closest_model(client_state_dict, global_state_dicts)
```
- **Purpose**: Handles heterogeneous clients with different data distributions
- **How it works**:
  1. Flattens all model weights into vectors (converts tensors → 1D arrays)
  2. Runs K-Means clustering to find `n_clusters` groups
  3. Averages models within each cluster separately
  4. Returns one global model per cluster
- **Use case**: Moon/Earth/Mars have different physics, so one global model may not fit all

#### **WeightStorageCallback**
Custom Stable-Baselines3 callback that tracks training metrics:

**Tracked Data:**
- **Model Weights**: Flattened weight vectors at specified intervals
- **Episode Rewards**: Total reward per episode
- **Episode Lengths**: Steps per episode
- **Training Steps**: Global step counter
- **Epoch Numbers**: Federated round number

**Storage Format**: NumPy compressed archive (`.npz`)

---

### 2. Standard Federated Learning (`moonlander/fl_moon.py`)

#### **Function: `run_fl_experiment()`**

**Parameters:**
- `NUM_ROUNDS`: Number of federated communication rounds
- `LOCAL_STEPS`: Training steps per client per round
- `clients_per_round`: Number of clients selected each round (partial participation)
- `task_list`: Client configurations with environment settings
- `n_envs`: Number of parallel environments for vectorized training

#### **Training Algorithm:**

```
Initialize:
  - Create global PPO model
  - Create client models (one per task)
  - Each client has different environment (Moon: -1.6g, Earth: -9.8g, Mars: -3.73g)

For each round in NUM_ROUNDS:
  1. Client Selection:
     - Randomly sample 'clients_per_round' clients
     - Simulates realistic FL where not all clients participate
  
  2. Broadcast:
     - Send global_model.weights → selected clients
     - All selected clients start from same weights
  
  3. Local Training:
     For each selected client:
       - Create vectorized environment (16 parallel envs)
       - Load global weights
       - Train PPO for LOCAL_STEPS
       - Store updated weights
       - Close environment (prevent resource leaks)
  
  4. Aggregation:
     - Collect weights from all selected clients
     - avg_weights = FedAvg(client_weights)
  
  5. Update:
     - global_model.weights = avg_weights

Save:
  - Global model → models/fl_global_model_final.zip
  - Training data → federated_training_data.npz
```

#### **Environment Management:**
- **Lazy Loading**: Creates environments only when needed (saves memory)
- **Vectorization**: Uses `SubprocVecEnv` for parallel training (16 environments)
- **Proper Cleanup**: Closes environments after each round to prevent file descriptor leaks

**⚠️ No Privacy Protection**: Weights are shared directly without any noise or encryption.

---

### 3. Differential Privacy Federated Learning (`moonlander/dp_moon.py`)

#### **Function: `run_dp_fl_experiment()`**

**Additional DP Parameters:**
- `DP_SENSITIVITY`: Maximum allowed L1 norm for weight updates (default: 15.0)
- `DP_EPSILON`: Privacy budget - lower = more privacy, more noise (tested: 5, 10, 30)

#### **Differential Privacy Mechanism**

The implementation uses the **Laplace Mechanism** with **Gradient Clipping** to achieve ε-differential privacy.

##### **Step-by-Step DP Process:**

**1. Calculate Weight Delta (Update)**
```python
delta[key] = new_weights[key] - global_weights[key]
```
- For each parameter in the model, compute the difference
- This represents how much the client wants to change the global model
- Example: If global weight = 0.5 and client weight = 0.7, delta = 0.2

**2. Compute L1 Norm of Delta**
```python
total_l1_norm = Σ|delta[key]| for all parameters
```
- Measures the "magnitude" of the total update
- L1 norm = sum of absolute values of all parameters
- Example: If deltas are [0.2, -0.3, 0.1], L1 norm = 0.2 + 0.3 + 0.1 = 0.6

**3. Gradient Clipping (Sensitivity Bounding)**
```python
clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))
clipped_delta[key] = delta[key] * clip_factor
```
- **Purpose**: Ensures all updates have bounded sensitivity
- **How it works**:
  - If L1 norm ≤ SENSITIVITY: clip_factor = 1.0 (no clipping)
  - If L1 norm > SENSITIVITY: scale down to exactly SENSITIVITY
- **Example**: 
  - SENSITIVITY = 15.0, L1 norm = 30.0
  - clip_factor = 15.0 / 30.0 = 0.5
  - All deltas are multiplied by 0.5 (halved)
- **Critical for DP**: Without clipping, we'd need infinite noise to protect unbounded updates

**4. Add Laplace Noise**
```python
DP_SCALE = DP_SENSITIVITY / DP_EPSILON
noise = np.random.laplace(0, scale=DP_SCALE, size=delta.shape)
noisy_delta[key] = clipped_delta[key] + noise
```
- **Noise Distribution**: Laplace(μ=0, b=DP_SCALE)
- **Scale Calculation**: 
  - DP_SCALE = SENSITIVITY / ε
  - Lower ε → higher scale → more noise → more privacy
- **Example**:
  - SENSITIVITY = 15.0, ε = 10.0
  - DP_SCALE = 15.0 / 10.0 = 1.5
  - Noise is sampled from Laplace(0, 1.5)
- **Per-Parameter**: Noise is added to **every single weight** in the model

**5. Aggregate Noisy Updates**
```python
avg_noisy_delta = average_ordered_dicts(noisy_deltas_from_selected_clients)
new_global_weights[key] = global_weights[key] + avg_noisy_delta[key]
```
- Average the noisy deltas from all selected clients
- Update global model by **adding** the averaged noisy delta
- This is different from standard FL which averages full weights

---

#### **Privacy Guarantees**

##### **ε-Differential Privacy Definition:**
A mechanism M satisfies ε-differential privacy if for any two neighboring datasets D and D' (differing by one client), and any output set S:

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]
```

**Interpretation:**
- ε controls the privacy-utility trade-off
- Smaller ε = stronger privacy (outputs are more similar regardless of one client's data)
- Larger ε = weaker privacy but better utility

##### **Privacy Budget Analysis:**

**Per-Round Privacy:**
- Each round satisfies ε-DP due to Laplace mechanism
- Noise scale calibrated to sensitivity and epsilon

**Total Privacy (Composition):**
```
ε_total = NUM_ROUNDS × ε_per_round  (Basic Composition Theorem)
```

**Example:**
- NUM_ROUNDS = 2
- ε_per_round = 10.0
- ε_total = 2 × 10.0 = 20.0

**⚠️ Limitation**: Uses basic composition, not advanced composition theorems (which give tighter bounds).

##### **Privacy-Utility Trade-off:**

| Epsilon | Privacy Level | Noise Level | Expected Performance |
|---------|---------------|-------------|---------------------|
| ε = 5   | Strong        | High        | Degraded (high variance) |
| ε = 10  | Moderate      | Medium      | Moderate performance |
| ε = 30  | Weak          | Low         | Near-standard FL |

---

#### **Differences from Standard FL:**

| Aspect | Standard FL | DP-FL |
|--------|-------------|-------|
| **What's Shared** | Full model weights | Noisy weight deltas |
| **Aggregation** | Average weights | Average noisy deltas, then add to global |
| **Privacy** | None (server sees exact weights) | ε-DP (server cannot infer individual contributions) |
| **Computation** | Simple averaging | Clipping + noise + averaging |
| **Performance** | Best | Degraded by noise |

---

#### **Potential Privacy Issues & Improvements:**

**✅ Correct Elements:**
1. Gradient clipping bounds sensitivity
2. Laplace noise calibrated to ε and sensitivity
3. Per-parameter noise addition
4. Delta-based updates (only share changes)

**⚠️ Limitations:**

1. **Privacy Budget Composition:**
   - Current: Basic composition (ε_total = T × ε)
   - Better: Advanced composition or Rényi DP (tighter bounds)
   - Impact: Overestimates privacy loss

2. **Sensitivity Metric:**
   - Current: L1 norm (sum of absolute values)
   - Alternative: L2 norm with Gaussian noise (more common in DP-SGD)
   - Trade-off: L1 is simpler but may be looser bound

3. **No Privacy Accounting:**
   - Doesn't track cumulative privacy loss across rounds
   - No adaptive noise scaling based on remaining budget
   - Could implement privacy accountant (like TensorFlow Privacy)

4. **Client Selection Privacy:**
   - Random selection doesn't provide additional privacy
   - Could leak information about which clients participated
   - Solution: Secure aggregation or shuffling

5. **No Secure Aggregation:**
   - Server sees individual noisy updates before averaging
   - Better: Cryptographic secure aggregation (server only sees sum)

---

### 4. Single Agent Training (`moonlander/single_moon.py`)

**Purpose**: Baseline comparison - trains independent agents without federated learning.

#### **How it works:**
```
For each client in task_list:
  1. Create environment with client's settings
  2. Create independent PPO model
  3. Train for TOTAL_TIMESTEPS (no communication)
  4. Save training data
```

**Key Differences from FL:**
- No weight sharing between clients
- No aggregation step
- Each agent learns only from its own environment
- Serves as control group to measure FL benefits

**Expected Results:**
- Agents perform well on their own environment
- No knowledge transfer between environments
- Total computation = FL computation (same total steps)

---

### 5. Clustered Federated Learning (`moonlander/clustered.py`)

**Advanced FL variant** that maintains multiple global models instead of one.

#### **Key Concept:**
Instead of forcing all clients to converge to one model, maintain `N_CLUSTERS` global models (e.g., 3 for Moon/Earth/Mars).

#### **Training Algorithm:**

```
Initialize:
  - Create N_CLUSTERS global models
  - Create client models

For each round:
  1. Assignment Phase:
     For each client:
       - Compute L2 distance to all global models
       - Assign to closest global model
       - Load that global model's weights
  
  2. Local Training:
     - Clients train using their assigned global model
  
  3. Clustering:
     - Collect all client weights
     - Run K-Means clustering into N_CLUSTERS groups
     - Average weights within each cluster
  
  4. Update:
     - Each global_model[i] = average of cluster i
```

#### **Advantages:**
- Better handles non-IID (heterogeneous) data
- Clients with similar tasks converge to same global model
- More flexible than single global model

#### **When to Use:**
- Heterogeneous client distributions (like Moon/Earth/Mars)
- When one model can't fit all tasks well
- When you know there are distinct client groups

---

### 6. Experiment Orchestration (`moonlander/experiment.py`)

Main script that runs all experiments sequentially and logs results.

#### **Experiment Pipeline:**

```python
# 1. Setup Base Clients
task_list = [
    {'label': 'Client_1_Moon', 'gravity': -1.6, 'wind': 0.5},
    {'label': 'Client_2_Earth', 'gravity': -9.8, 'wind': 0.5},
    {'label': 'Client_3_Mars', 'gravity': -3.73, 'wind': 0.5},
]

# 2. Data Augmentation
add_derived_tasks(task_list, num_to_add_per_task=10)
# Creates 30 additional clients with random wind values
# Total: 33 clients

# 3. Run Experiments
experiments = [
    'Standard FL',
    'DP-FL (ε=30)',
    'DP-FL (ε=10)',
    'DP-FL (ε=5)',
    'Single Agent'
]

# 4. Log execution times
```

#### **Hyperparameters:**
```python
NUM_ROUNDS = 2              # Federated communication rounds
LOCAL_STEPS = 50,000        # Steps per client per round
CHECK_FREQ = 2,000          # Log weights every 2000 steps
clients_per_round = 16      # Half of 33 total clients
n_envs = 4                  # Parallel environments per client
```

#### **Total Training Steps:**
- FL: 2 rounds × 50,000 steps × 16 clients = 1,600,000 steps
- Single: 100,000 steps × 3 clients = 300,000 steps

---

## Differential Privacy Implementation

### Mathematical Foundation

#### **Laplace Mechanism**
For a function f with sensitivity Δf, the Laplace mechanism achieves ε-DP by:

```
M(D) = f(D) + Lap(Δf / ε)
```

Where:
- `Lap(b)` is Laplace distribution with scale b
- `Δf` is the maximum change in f when one record changes (sensitivity)
- `ε` is the privacy parameter

#### **Sensitivity Calculation**

**L1 Sensitivity:**
```
Δf = max_{D,D'} ||f(D) - f(D')||_1
```

In our case:
- f(D) = client's weight update
- We bound this by clipping to DP_SENSITIVITY
- Therefore: Δf = DP_SENSITIVITY

#### **Noise Scale:**
```
b = Δf / ε = DP_SENSITIVITY / DP_EPSILON
```

### Code Implementation Details

#### **Full DP Pipeline in Code:**

```python
# 1. Train locally
client.learn(total_timesteps=LOCAL_STEPS)
new_weights = client.policy.state_dict()

# 2. Compute delta
delta = OrderedDict()
for key in global_state_dict.keys():
    delta[key] = new_weights[key] - global_state_dict[key]

# 3. Calculate L1 norm
total_l1_norm = 0.0
for key in delta.keys():
    total_l1_norm += torch.sum(torch.abs(delta[key]))
total_l1_norm = total_l1_norm.item()

# 4. Clip to sensitivity
clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))

# 5. Add Laplace noise
noisy_delta = OrderedDict()
DP_SCALE = DP_SENSITIVITY / DP_EPSILON

for key in delta.keys():
    clipped_delta = delta[key] * clip_factor
    
    noise = torch.tensor(
        np.random.laplace(0, scale=DP_SCALE, size=clipped_delta.shape),
        dtype=clipped_delta.dtype,
        device=clipped_delta.device
    )
    
    noisy_delta[key] = clipped_delta + noise

# 6. Aggregate across clients
avg_noisy_delta = average_ordered_dicts(all_noisy_deltas)

# 7. Update global model
for key in global_state_dict.keys():
    new_global_state_dict[key] = global_state_dict[key] + avg_noisy_delta[key]
```

### Privacy Analysis

#### **What Information is Protected:**
1. **Individual Updates**: Server cannot determine exact contribution of any single client
2. **Participation**: Difficult to infer if specific client participated (with noise)
3. **Training Data**: Client's training episodes are protected by noise

#### **What Information Leaks:**
1. **Number of Clients**: Server knows how many clients participated
2. **Model Architecture**: Shared across all clients
3. **Approximate Updates**: With high ε, noise is small, updates are approximate

#### **Privacy Budget Depletion:**
```
Round 1: ε_1 consumed
Round 2: ε_2 consumed
...
Total: ε_total = Σ ε_i  (basic composition)
```

**Implication**: More rounds = more privacy loss. Need to balance:
- More rounds → better convergence
- Fewer rounds → better privacy

---

## Technical Details

### Environment Configurations

#### **LunarLander-v3 Settings:**

| Client Type | Gravity (m/s²) | Wind Power | Description |
|-------------|----------------|------------|-------------|
| Moon        | -1.6           | 0.0-15.0   | Low gravity, easier landing |
| Earth       | -9.8           | 0.0-15.0   | Standard gravity |
| Mars        | -3.73          | 0.0-15.0   | Medium gravity |

**Heterogeneity**: Different gravities create non-IID data distributions, making FL challenging.

### PPO Hyperparameters

Using Stable-Baselines3 default PPO settings:
- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Buffer Size**: 2048
- **Gamma**: 0.99
- **GAE Lambda**: 0.95

### Computational Resources

**Parallelization:**
- `n_envs = 4-16`: Parallel environments per client
- `SubprocVecEnv`: Each environment in separate process
- Speeds up training significantly

**Memory Management:**
- Lazy environment loading
- Explicit environment closing
- Prevents file descriptor exhaustion

---

## Results and Visualization

### Plotting Functions

#### **1. Learning Curves (`plotting.py`)**
```python
plot_learning_curves(data, output_filename, title)
```
- **X-axis**: Local client epochs (steps / PPO_BUFFER_SIZE)
- **Y-axis**: Smoothed episode reward (rolling window = 100)
- **Color**: By client type (Moon/Earth/Mars)
- **Shows**: Training progress over time

#### **2. t-SNE Visualization**
```python
plot_tsne_weights(data, output_filename, title)
```
- **Dimensionality Reduction**: Projects high-dimensional weights to 2D
- **Color**: Training timestep (darker = later)
- **Marker Shape**: Client type (Moon/Earth/Mars)
- **Shows**: How model weights evolve and cluster

#### **3. UMAP Visualization**
```python
plot_umap_weights(data, output_filename, title)
```
- Similar to t-SNE but preserves global structure better
- Faster for large datasets
- Shows weight space topology

#### **4. Comparison Plots (`comparison_plot.py`)**

**Combined Plot:**
- All experiments on one graph
- Color by experiment type (FL/Single/DP)
- Line style by client type

**Side-by-Side Plot:**
- Separate subplot per client
- Direct comparison of FL/Single/DP for each client
- Easier to see individual client performance

---

## Simple Test Code (`zia_test/main.py`)

### Purpose
Educational example showing basic RL concepts before complex FL.

### Implementation Details

#### **PolicyNetwork Class:**
```python
class PolicyNetwork:
    def __init__(self, state_size, action_size):
        self.weights = np.random.randn(state_size, action_size) * 0.01
```
- Simple linear policy: action_probs = softmax(state · weights)
- No hidden layers (minimal complexity)

#### **Policy Gradient Algorithm (REINFORCE):**
```python
def update(self, states, actions, rewards):
    # 1. Compute discounted rewards
    discounted_rewards = discount_rewards(rewards, gamma=0.99)
    
    # 2. Normalize rewards (reduce variance)
    normalized_rewards = (rewards - mean) / std
    
    # 3. Policy gradient update
    for state, action, reward in zip(states, actions, normalized_rewards):
        probs = predict(state)
        gradient = outer(state, probs)
        gradient[action] -= state  # Log-likelihood gradient
        weights -= learning_rate * gradient * reward
```

#### **Key Concepts Demonstrated:**
1. **Policy Gradient**: Update weights to increase probability of good actions
2. **Reward Discounting**: Future rewards worth less (γ = 0.99)
3. **Variance Reduction**: Normalize rewards to stabilize training
4. **Episode-based Learning**: Update after full episode

**Not used in main experiments** - just for learning RL basics.

---

## Summary

### What This Codebase Achieves:

1. ✅ **Federated Reinforcement Learning**: Implements FedAvg for RL (non-trivial, most FL is supervised)
2. ✅ **Differential Privacy**: Adds ε-DP guarantees to federated RL
3. ✅ **Heterogeneous Clients**: Handles different environment physics (Moon/Earth/Mars)
4. ✅ **Client Selection**: Partial participation (realistic FL scenario)
5. ✅ **Comprehensive Evaluation**: Compares FL vs Single vs DP-FL
6. ✅ **Rich Visualization**: Learning curves, t-SNE, UMAP, comparison plots

### Differential Privacy Status:

| Aspect | Status | Notes |
|--------|--------|-------|
| **Gradient Clipping** | ✅ Implemented | Bounds L1 norm to SENSITIVITY |
| **Laplace Noise** | ✅ Implemented | Calibrated to ε and sensitivity |
| **Per-Parameter Noise** | ✅ Implemented | Noise on every weight |
| **ε-DP Guarantee** | ✅ Provided | Per-round privacy |
| **Privacy Accounting** | ⚠️ Basic | Uses basic composition |
| **Secure Aggregation** | ❌ Not Implemented | Server sees individual updates |
| **Advanced Composition** | ❌ Not Implemented | Could tighten privacy bounds |

### Key Innovation:

**Combining three challenging areas:**
- Federated Learning (distributed training)
- Reinforcement Learning (sequential decision making)
- Differential Privacy (formal privacy guarantees)

In a **heterogeneous multi-environment setting** (different planetary gravities).

---

## Future Improvements

1. **Privacy Enhancements:**
   - Implement privacy accountant (track cumulative ε)
   - Use advanced composition theorems
   - Add secure aggregation
   - Implement DP-SGD style per-sample clipping

2. **Algorithm Improvements:**
   - Adaptive privacy budgets
   - Personalized federated learning
   - Better clustering algorithms
   - Momentum-based aggregation

3. **Evaluation:**
   - Formal privacy auditing
   - Attack simulations (membership inference)
   - More diverse environments
   - Longer training runs

4. **Engineering:**
   - Distributed training across machines
   - GPU acceleration
   - Checkpoint/resume functionality
   - Hyperparameter tuning

---

## References

### Federated Learning:
- McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)

### Differential Privacy:
- Dwork & Roth (2014): "The Algorithmic Foundations of Differential Privacy"
- Abadi et al. (2016): "Deep Learning with Differential Privacy" (DP-SGD)

### Reinforcement Learning:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" (PPO)
- Stable-Baselines3 Documentation

### Federated RL:
- Nadiger et al. (2019): "Federated Reinforcement Learning"
- Zhuo et al. (2019): "Federated Deep Reinforcement Learning"

---

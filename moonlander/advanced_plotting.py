import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# --- Configuration ---
STANDARD_FILE = 'federated_training_data.npz'
DP_FILE = 'dp_training_data.npz'
SINGLE_FILE = 'training_data.npz'  # <--- New file from single_moon.py
OUTPUT_DIR = "comparison_plots"


def load_data(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    return np.load(filename, allow_pickle=True)


def get_client_type(label):
    """
    Extracts the environment type from the label.
    e.g. "Client_1_Moon" -> "Moon"
    """
    try:
        return label.split('_')[-1]
    except:
        return label


def plot_three_way_pca(std_data, dp_data, single_data):
    """
    Projects ALL three experiments into the PCA space defined by Standard FL.
    This shows:
      1. The Generalization Cluster (Std FL)
      2. The Privacy Noise (DP FL)
      3. The Specialization Drift (Single Agent)
    """
    print("Generating 3-Way PCA Comparison...")

    # 1. Fit PCA on Standard FL (The Baseline)
    std_weights = std_data['weights']
    pca = PCA(n_components=2)
    pca.fit(std_weights)

    # 2. Transform all datasets
    std_pca = pca.transform(std_weights)
    dp_pca = pca.transform(dp_data['weights'])
    single_pca = pca.transform(single_data['weights'])

    # 3. Plot
    plt.figure(figsize=(18, 6))

    # Helper for plotting
    def plot_scatter(ax, data_pca, labels, title):
        types = [get_client_type(l) for l in labels]
        unique_types = sorted(list(set(types)))
        colors = plt.cm.get_cmap('tab10', len(unique_types))

        for i, t in enumerate(unique_types):
            indices = [idx for idx, x in enumerate(types) if x == t]
            if not indices:
                continue
            ax.scatter(
                data_pca[indices, 0],
                data_pca[indices, 1],
                label=t,
                color=colors(i),
                alpha=0.6
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Lock axis to standard scale for fair comparison
        ax.set_xlim(std_pca[:, 0].min() - 1, std_pca[:, 0].max() + 1)
        ax.set_ylim(std_pca[:, 1].min() - 1, std_pca[:, 1].max() + 1)

    # Subplot 1: Standard FL
    plot_scatter(plt.subplot(1, 3, 1), std_pca,
                 std_data['weight_labels'], "Standard FL (Shared Model)")

    # Subplot 2: DP FL
    plot_scatter(plt.subplot(1, 3, 2), dp_pca,
                 dp_data['weight_labels'], "DP FL (Noisy Shared Model)")

    # Subplot 3: Single Agent
    plot_scatter(plt.subplot(1, 3, 3), single_pca,
                 single_data['weight_labels'], "Single Agent (Independent)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_3way_pca.png"))
    print("Saved comparison_3way_pca.png")


def plot_performance_benchmark(std_data, dp_data, single_data):
    """
    Overlays learning curves. 
    Single Agent usually learns FASTER (high reward) because it specializes.
    FL usually learns SLOWER but generalizes better.
    """
    print("Generating Performance Benchmark Plot...")

    plt.figure(figsize=(12, 8))

    def get_smooth_curve(data):
        df = pd.DataFrame({
            'steps': data['ep_steps'],
            'reward': data['ep_rewards']
        })
        # Group by step to average across all clients
        df['step_group'] = (df['steps'] // 2000) * 2000
        mean_perf = df.groupby('step_group')['reward'].mean()
        return mean_perf.rolling(window=5, min_periods=1).mean()

    # Plot Single Agent (The "Ideal" Specialist)
    if single_data:
        curve = get_smooth_curve(single_data)
        plt.plot(curve.index, curve, label="Single Agent (Independent)",
                 color='green', linewidth=2.5, linestyle='-')

    # Plot Standard FL (The Collaborative Generalist)
    if std_data:
        curve = get_smooth_curve(std_data)
        plt.plot(curve.index, curve, label="Standard FL",
                 color='blue', linewidth=2.5, linestyle='--')

    # Plot DP FL (The Private Version)
    if dp_data:
        curve = get_smooth_curve(dp_data)
        plt.plot(curve.index, curve, label="DP FL (Private)",
                 color='orange', linewidth=2, linestyle=':')

    plt.title("Impact of Federation & Privacy on Learning Speed")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_benchmark_curve.png"))
    print("Saved comparison_benchmark_curve.png")


def plot_cluster_separability(std_data, dp_data):
    # ... (Keep this function from previous response, it is specific to FL comparison) ...
    # Only change: ensure it handles missing data gracefully
    if not std_data or not dp_data:
        return

    print("Generating Distinguishability Metric...")

    def get_score(data):
        weights = data['weights']
        labels = data['weight_labels']
        steps = data['weight_steps']
        last_step = np.max(steps)
        mask = (steps == last_step)

        final_weights = weights[mask]
        final_raw = labels[mask]
        final_groups = np.array([get_client_type(l) for l in final_raw])

        if len(np.unique(final_groups)) < 2 or len(final_weights) <= len(np.unique(final_groups)):
            return 0.0
        return silhouette_score(final_weights, final_groups)

    score_std = get_score(std_data)
    score_dp = get_score(dp_data)

    plt.figure(figsize=(8, 6))
    plt.bar(['Standard FL', 'DP FL'], [score_std, score_dp],
            color=['#1f77b4', '#ff7f0e'])
    plt.title('Differentiation (Silhouette Score)')
    plt.ylabel('Score (Higher = More Distinct)')
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_distinguishability.png"))
    print("Saved comparison_distinguishability.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    std_data = load_data(STANDARD_FILE)
    dp_data = load_data(DP_FILE)
    single_data = load_data(SINGLE_FILE)

    if std_data and dp_data and single_data:
        plot_three_way_pca(std_data, dp_data, single_data)
        plot_performance_benchmark(std_data, dp_data, single_data)
        plot_cluster_separability(std_data, dp_data)
    else:
        print("Error: Could not load all three datasets.")
        print(
            f"Standard: {std_data is not None}, DP: {dp_data is not None}, Single: {single_data is not None}")
        print("Please run single_moon.py, fl_moon.py, and dp_moon.py first.")

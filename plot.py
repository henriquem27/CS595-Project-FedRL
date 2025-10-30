import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA
import sys


def plot_all_data(data_file='training_data.npz'):
    """
    Loads data from the .npz file and generates two plots:
    1. A side-by-side PCA vs. UMAP of the model weights.
    2. A learning curve of the episode rewards.
    """

    # -----------------------------------------------------------------
    # 1. Load the Data
    # -----------------------------------------------------------------
    print(f"Loading data from {data_file}...")
    try:
        data = np.load(data_file)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.", file=sys.stderr)
        print(
            "Please run your training script first to generate the file.", file=sys.stderr)
        return

    # Extract all the arrays
    X_weights = data['weights']
    y_weight_labels = data['weight_labels']
    weight_steps = data['weight_steps']

    ep_rewards = data['ep_rewards']
    ep_labels = data['ep_labels']
    ep_steps = data['ep_steps']

    print(f"Loaded {X_weights.shape[0]} weight vectors.")
    print(f"Loaded {ep_rewards.shape[0]} episode reward records.")

    # -----------------------------------------------------------------
    # 2. Plot 1: Weight Clustering (PCA vs. UMAP)
    # -----------------------------------------------------------------
    print("Running PCA and UMAP...")

    # --- Run PCA ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_weights)

    # Create a DataFrame for plotting PCA
    pca_df = pd.DataFrame(
        X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Agent'] = y_weight_labels
    pca_df['Step'] = weight_steps

    # --- Run UMAP ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_weights)

    # Create a DataFrame for plotting UMAP
    umap_df = pd.DataFrame(X_umap, columns=['UMAP 1', 'UMAP 2'])
    umap_df['Agent'] = y_weight_labels
    umap_df['Step'] = weight_steps

    # --- Create the side-by-side plot ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig1.suptitle('High-Dimensional Weight Clustering Analysis', fontsize=16)

    # Plot PCA
    sns.scatterplot(
        data=pca_df,
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Agent',
        style='Agent',
        size='Step',
        palette='bright',
        alpha=0.7,
        ax=ax1
    )
    ax1.set_title('PCA Projection of Model Weights')

    # Plot UMAP
    sns.scatterplot(
        data=umap_df,
        x='UMAP 1',
        y='UMAP 2',
        hue='Agent',
        style='Agent',
        size='Step',
        palette='bright',
        alpha=0.7,
        ax=ax2
    )
    ax2.set_title('UMAP Projection of Model Weights')

    print("Generated clustering plot.")

    # -----------------------------------------------------------------
    # 3. Plot 2: Learning Curves (Episode Rewards)
    # -----------------------------------------------------------------
    print("Generating learning curves...")

    # Create a DataFrame for the episode data
    # This is perfect for seaborn's lineplot
    episode_df = pd.DataFrame({
        'Step': ep_steps,
        'Reward': ep_rewards,
        'Agent': ep_labels
    })

    # --- Create the line plot ---
    # seaborn.lineplot will automatically aggregate data by step
    # and plot the mean with a confidence interval.
    fig2, ax_reward = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=episode_df,
        x='Step',
        y='Reward',
        hue='Agent',
        style='Agent',
        markers=True,
        dashes=False,
        ax=ax_reward
    )

    ax_reward.set_title('Learning Curve: Episode Reward over Time')
    ax_reward.set_xlabel('Training Timesteps')
    ax_reward.set_ylabel('Episode Reward')
    ax_reward.legend(title='Agent')

    print("Generated learning curve plot.")

    # -----------------------------------------------------------------
    # 4. Show All Plots
    # -----------------------------------------------------------------
    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == '__main__':
    plot_all_data()
